"""Service metier des memoires utilisateur stockees dans Supabase.

Ce module appartient a la couche ``core``: il contient la logique metier
et les acces a Supabase pour la table ``user_memories``. Les DTO Pydantic
restent dans ``models.user_memory`` afin d'eviter tout couplage inverse
entre les couches.
"""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from core.supabase_client import get_supabase_admin_client
from models.user_memory import UserMemory, UserMemoryCreate, UserMemorySource, UserMemoryUpdate

logger = logging.getLogger("makenbrain.user_memories")

USER_MEMORIES_TABLE = "user_memories"


class UserMemoryServiceError(RuntimeError):
    """Erreur racine pour les operations de memoire utilisateur.

    Les exceptions techniques du SDK Supabase sont encapsulees dans ce
    type afin de presenter un contrat stable aux routes et aux autres
    services applicatifs.
    """


class UserMemoryNotFoundError(UserMemoryServiceError):
    """Signale qu'aucune memoire n'existe dans le perimetre utilisateur.

    Une memoire appartenant a un autre utilisateur est volontairement
    traitee comme introuvable pour ne jamais reveler l'existence de
    donnees hors scope.
    """


class UserMemoryService:
    """Service applicatif pour les memoires rattachees a un utilisateur.

    Chaque operation applique explicitement le filtre ``user_id`` afin
    de garantir l'isolation multi-utilisateur meme lorsque le client
    Supabase admin contourne les politiques RLS.
    """

    def __init__(self, client: Any | None = None) -> None:
        """Cree le service avec un client Supabase optionnel.

        L'injection d'un client permet de tester la logique metier sans
        declencher de connexion reelle a Supabase.
        """
        self._client = client or get_supabase_admin_client()

    def create_memory(
        self,
        user_id: UUID | str,
        content: str,
        source: UserMemorySource | str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Cree une memoire utilisateur et retourne un dict normalise."""
        memory = UserMemoryCreate(
            user_id=user_id,
            content=content,
            source=source,
            metadata=metadata or {},
        )
        payload = memory.model_dump(mode="json")

        try:
            response = self._client.table(USER_MEMORIES_TABLE).insert(payload).execute()
        except Exception as exc:  # noqa: BLE001 - Supabase SDK exceptions vary by version.
            logger.exception("Creation memoire utilisateur Supabase echouee.")
            raise UserMemoryServiceError(f"Creation memoire utilisateur impossible : {exc}") from exc

        row = self._first_row(response)
        if row is None:
            raise UserMemoryServiceError("Creation memoire utilisateur impossible : aucune ligne retournee.")
        return self._to_memory_dict(row)

    def get_user_memories(self, user_id: UUID | str) -> list[dict[str, Any]]:
        """Retourne toutes les memoires appartenant a un utilisateur."""
        try:
            response = (
                self._client.table(USER_MEMORIES_TABLE)
                .select("*")
                .eq("user_id", str(user_id))
                .execute()
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lecture memoires utilisateur Supabase echouee.")
            raise UserMemoryServiceError(f"Lecture memoires utilisateur impossible : {exc}") from exc

        return [self._to_memory_dict(row) for row in self._rows(response)]

    def get_memory(self, user_id: UUID | str, memory_id: UUID | str) -> dict[str, Any]:
        """Retourne une memoire uniquement si elle appartient a l'utilisateur."""
        try:
            response = (
                self._client.table(USER_MEMORIES_TABLE)
                .select("*")
                .eq("id", str(memory_id))
                .eq("user_id", str(user_id))
                .limit(1)
                .execute()
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lecture memoire utilisateur Supabase echouee.")
            raise UserMemoryServiceError(f"Lecture memoire utilisateur impossible : {exc}") from exc

        row = self._first_row(response)
        if row is None:
            raise UserMemoryNotFoundError(f"Memoire utilisateur introuvable : {memory_id}")
        return self._to_memory_dict(row)

    def update_memory(
        self,
        user_id: UUID | str,
        memory_id: UUID | str,
        data: UserMemoryUpdate | dict[str, Any],
    ) -> dict[str, Any]:
        """Met a jour une memoire dans le perimetre strict de l'utilisateur."""
        update_model = UserMemoryUpdate.model_validate(data)
        payload = update_model.model_dump(mode="json", exclude_none=True, exclude_unset=True)
        if not payload:
            raise ValueError("Aucune donnee de memoire utilisateur a mettre a jour.")

        try:
            response = (
                self._client.table(USER_MEMORIES_TABLE)
                .update(payload)
                .eq("id", str(memory_id))
                .eq("user_id", str(user_id))
                .execute()
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Mise a jour memoire utilisateur Supabase echouee.")
            raise UserMemoryServiceError(f"Mise a jour memoire utilisateur impossible : {exc}") from exc

        row = self._first_row(response)
        if row is None:
            raise UserMemoryNotFoundError(f"Memoire utilisateur introuvable : {memory_id}")
        return self._to_memory_dict(row)

    def delete_memory(self, user_id: UUID | str, memory_id: UUID | str) -> None:
        """Supprime une memoire dans le perimetre strict de l'utilisateur."""
        try:
            response = (
                self._client.table(USER_MEMORIES_TABLE)
                .delete()
                .eq("id", str(memory_id))
                .eq("user_id", str(user_id))
                .execute()
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Suppression memoire utilisateur Supabase echouee.")
            raise UserMemoryServiceError(f"Suppression memoire utilisateur impossible : {exc}") from exc

        if self._first_row(response) is None:
            raise UserMemoryNotFoundError(f"Memoire utilisateur introuvable : {memory_id}")

    @staticmethod
    def _rows(response: Any) -> list[dict[str, Any]]:
        """Extrait les lignes d'une reponse Supabase objet ou mapping."""
        data = response.get("data") if isinstance(response, dict) else getattr(response, "data", None)
        if data is None:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise UserMemoryServiceError("Reponse Supabase inattendue pour la table user_memories.")

    @classmethod
    def _first_row(cls, response: Any) -> dict[str, Any] | None:
        """Retourne la premiere ligne d'une reponse Supabase, si presente."""
        rows = cls._rows(response)
        return rows[0] if rows else None

    @staticmethod
    def _to_memory_dict(row: dict[str, Any]) -> dict[str, Any]:
        """Normalise une ligne Supabase via le DTO public ``UserMemory``."""
        return UserMemory.model_validate(row).model_dump(mode="json")


_default_service: UserMemoryService | None = None


def _get_default_service() -> UserMemoryService:
    """Retourne l'instance de service par defaut, initialisee au besoin."""
    global _default_service
    if _default_service is None:
        _default_service = UserMemoryService()
    return _default_service


def create_memory(
    user_id: UUID | str,
    content: str,
    source: UserMemorySource | str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cree une memoire utilisateur via le service par defaut."""
    return _get_default_service().create_memory(user_id, content, source, metadata)


def get_user_memories(user_id: UUID | str) -> list[dict[str, Any]]:
    """Retourne les memoires d'un utilisateur via le service par defaut."""
    return _get_default_service().get_user_memories(user_id)


def get_memory(user_id: UUID | str, memory_id: UUID | str) -> dict[str, Any]:
    """Retourne une memoire utilisateur via le service par defaut."""
    return _get_default_service().get_memory(user_id, memory_id)


def update_memory(user_id: UUID | str, memory_id: UUID | str, data: UserMemoryUpdate | dict[str, Any]) -> dict[str, Any]:
    """Met a jour une memoire utilisateur via le service par defaut."""
    return _get_default_service().update_memory(user_id, memory_id, data)


def delete_memory(user_id: UUID | str, memory_id: UUID | str) -> None:
    """Supprime une memoire utilisateur via le service par defaut."""
    _get_default_service().delete_memory(user_id, memory_id)

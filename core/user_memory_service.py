"""Service layer for user-scoped memories stored in Supabase."""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from core.supabase_client import get_supabase_admin_client
from models.user_memory import UserMemory, UserMemoryCreate, UserMemorySource, UserMemoryUpdate

logger = logging.getLogger("makenbrain.user_memories")

USER_MEMORIES_TABLE = "user_memories"


class UserMemoryServiceError(RuntimeError):
    """Raised when a user memory operation fails at the service or Supabase layer."""


class UserMemoryNotFoundError(UserMemoryServiceError):
    """Raised when a requested memory does not exist in the user's scope."""


class UserMemoryService:
    """Business service for memories owned by one Supabase user profile."""

    def __init__(self, client: Any | None = None) -> None:
        """Create the service with an optional Supabase client override."""
        self._client = client or get_supabase_admin_client()

    def create_memory(
        self,
        user_id: UUID | str,
        content: str,
        source: UserMemorySource | str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a memory for one user and return a normalized dict."""
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
        """Return all memories scoped to one user."""
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
        """Return one memory only if it belongs to the provided user."""
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
        """Update one memory only inside the provided user's scope."""
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
        """Delete one memory only inside the provided user's scope."""
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
        """Extract rows from a Supabase response object or mapping."""
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
        """Return the first row from a Supabase response, if any."""
        rows = cls._rows(response)
        return rows[0] if rows else None

    @staticmethod
    def _to_memory_dict(row: dict[str, Any]) -> dict[str, Any]:
        """Normalize a Supabase row into a JSON-ready memory dict."""
        return UserMemory.model_validate(row).model_dump(mode="json")


_default_service: UserMemoryService | None = None


def _get_default_service() -> UserMemoryService:
    """Return a lazily initialized default service instance."""
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
    """Create a memory for one user using the default service."""
    return _get_default_service().create_memory(user_id, content, source, metadata)


def get_user_memories(user_id: UUID | str) -> list[dict[str, Any]]:
    """Return all memories for one user using the default service."""
    return _get_default_service().get_user_memories(user_id)


def get_memory(user_id: UUID | str, memory_id: UUID | str) -> dict[str, Any]:
    """Return one user-scoped memory using the default service."""
    return _get_default_service().get_memory(user_id, memory_id)


def update_memory(user_id: UUID | str, memory_id: UUID | str, data: UserMemoryUpdate | dict[str, Any]) -> dict[str, Any]:
    """Update one user-scoped memory using the default service."""
    return _get_default_service().update_memory(user_id, memory_id, data)


def delete_memory(user_id: UUID | str, memory_id: UUID | str) -> None:
    """Delete one user-scoped memory using the default service."""
    _get_default_service().delete_memory(user_id, memory_id)

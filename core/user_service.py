"""Business service for users stored in Supabase."""
from __future__ import annotations

import logging
from typing import Any, Protocol

from models.user import User, UserCreate, UserUpdate
from core.supabase_client import get_supabase_admin_client

logger = logging.getLogger("makenbrain.users")

USERS_TABLE = "users"


class UserServiceError(RuntimeError):
    """Raised when a user operation fails at the service or Supabase layer."""


class UserNotFoundError(UserServiceError):
    """Raised when a requested user does not exist."""


class SupabaseLikeClient(Protocol):
    """Minimal Supabase client surface used by this service."""

    def table(self, table_name: str) -> Any:
        """Return a query builder for a Supabase table."""


class UserService:
    """Centralized user business layer backed by the Supabase `users` table."""

    def __init__(self, client: SupabaseLikeClient | None = None) -> None:
        """Create the service with an optional Supabase client override.

        When no client is provided, the service uses the server-side
        service-role client. Phase 2.2 has no login/session layer yet, so
        business operations must not depend on an authenticated end user.
        """
        self._client = client or get_supabase_admin_client()

    def create_user(self, user: UserCreate | dict[str, Any]) -> User:
        """Create a user in Supabase and return the inserted row."""
        payload = self._payload_for_write(UserCreate.model_validate(user))
        try:
            response = self._client.table(USERS_TABLE).insert(payload).execute()
        except Exception as exc:  # noqa: BLE001 - SDK exceptions are not stable across versions.
            logger.exception("Creation utilisateur Supabase echouee.")
            raise UserServiceError(f"Creation utilisateur impossible : {exc}") from exc

        row = self._first_row(response)
        if row is None:
            raise UserServiceError("Creation utilisateur impossible : Supabase n'a retourne aucune ligne.")
        return self._to_user(row)

    def get_user(self, user_id: str) -> User:
        """Return one user by id.

        Raises:
            UserNotFoundError: if no matching row exists.
        """
        try:
            response = (
                self._client.table(USERS_TABLE)
                .select("*")
                .eq("id", user_id)
                .limit(1)
                .execute()
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lecture utilisateur Supabase echouee.")
            raise UserServiceError(f"Lecture utilisateur impossible : {exc}") from exc

        row = self._first_row(response)
        if row is None:
            raise UserNotFoundError(f"Utilisateur introuvable : {user_id}")
        return self._to_user(row)

    def update_user(self, user_id: str, updates: UserUpdate | dict[str, Any]) -> User:
        """Update mutable user fields and return the updated row."""
        update_model = UserUpdate.model_validate(updates)
        payload = self._payload_for_write(update_model, exclude_unset=True)
        if not payload:
            raise ValueError("Aucune donnee utilisateur a mettre a jour.")

        try:
            response = (
                self._client.table(USERS_TABLE)
                .update(payload)
                .eq("id", user_id)
                .execute()
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Mise a jour utilisateur Supabase echouee.")
            raise UserServiceError(f"Mise a jour utilisateur impossible : {exc}") from exc

        row = self._first_row(response)
        if row is None:
            raise UserNotFoundError(f"Utilisateur introuvable : {user_id}")
        return self._to_user(row)

    def delete_user(self, user_id: str) -> None:
        """Delete one user by id.

        Raises:
            UserNotFoundError: if Supabase did not delete any row.
        """
        try:
            response = (
                self._client.table(USERS_TABLE)
                .delete()
                .eq("id", user_id)
                .execute()
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Suppression utilisateur Supabase echouee.")
            raise UserServiceError(f"Suppression utilisateur impossible : {exc}") from exc

        if self._first_row(response) is None:
            raise UserNotFoundError(f"Utilisateur introuvable : {user_id}")

    def list_users(self, limit: int = 100) -> list[User]:
        """Return visible users from Supabase, capped by `limit`."""
        if limit < 1:
            raise ValueError("limit doit etre superieur ou egal a 1.")

        try:
            response = self._client.table(USERS_TABLE).select("*").limit(limit).execute()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Liste utilisateurs Supabase echouee.")
            raise UserServiceError(f"Liste utilisateurs impossible : {exc}") from exc

        rows = self._rows(response)
        return [self._to_user(row) for row in rows]

    @staticmethod
    def _payload_for_write(model: UserCreate | UserUpdate, *, exclude_unset: bool = False) -> dict[str, Any]:
        """Convert a user model into a conservative Supabase payload."""
        payload = model.model_dump(mode="json", exclude_none=True, exclude_unset=exclude_unset)
        profile = payload.pop("profile", None)
        if profile and not payload.get("name") and profile.get("full_name"):
            payload["name"] = profile["full_name"]
        return payload

    @staticmethod
    def _rows(response: Any) -> list[dict[str, Any]]:
        """Extract list data from a Supabase response object or mapping."""
        data = response.get("data") if isinstance(response, dict) else getattr(response, "data", None)
        if data is None:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise UserServiceError("Reponse Supabase inattendue pour la table users.")

    @classmethod
    def _first_row(cls, response: Any) -> dict[str, Any] | None:
        """Return the first row from a Supabase response, if any."""
        rows = cls._rows(response)
        return rows[0] if rows else None

    @staticmethod
    def _to_user(row: dict[str, Any]) -> User:
        """Normalize common database field names into the public User model."""
        normalized = dict(row)
        if not normalized.get("name"):
            normalized["name"] = normalized.get("full_name") or normalized.get("display_name")
        return User.model_validate(normalized)


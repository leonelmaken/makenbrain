"""Authentication dependencies for MakenBrain endpoints.

Two authentication modes coexist:
- local admin API key via `X-API-Key` for sensitive internal endpoints;
- Supabase bearer token for future user-facing endpoints that must sync
  the authenticated user with the local `users` profile table.

The local admin gate keeps its fail-closed behavior: if `ADMIN_API_KEY` is
not configured, protected endpoints refuse access instead of staying open.
"""
from __future__ import annotations

import hmac
import logging
from typing import Any

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from core.audit import audit_event
from core.config import settings
from core.rate_limit import enforce_rate_limit
from core.supabase_client import get_supabase_admin_client
from core.user_service import UserService, UserServiceError
from models.user import AuthUserIdentity, User, UserProfile, UserRole

logger = logging.getLogger("makenbrain.auth")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_auth = HTTPBearer(auto_error=False)


def require_api_key(api_key: str | None = Security(_api_key_header)) -> str:
    """FastAPI dependency enforcing the local admin API key.

    Args:
        api_key: Value of the `X-API-Key` header, injected by FastAPI.

    Returns:
        The validated API key.

    Raises:
        HTTPException: 503 if no ADMIN_API_KEY is configured server-side
            (fail-closed), 401 if the header is missing or does not match.
    """
    if not settings.ADMIN_API_KEY:
        logger.error("ADMIN_API_KEY non configuree - acces refuse par defaut (fail-closed).")
        raise HTTPException(
            status_code=503,
            detail="Authentification non configuree cote serveur (ADMIN_API_KEY manquant dans .env).",
        )
    if not api_key or not hmac.compare_digest(api_key, settings.ADMIN_API_KEY):
        audit_event(
            action="auth.denied",
            tool="auth",
            result="invalid or missing X-API-Key",
            success=False,
        )
        raise HTTPException(status_code=401, detail="Cle API invalide ou manquante (en-tete X-API-Key).")
    return api_key


def require_supabase_user(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer_auth),
) -> User:
    """Validate a Supabase access token and sync its application profile.

    This dependency is separate from the local admin API-key gate. It is
    meant for future user-facing endpoints where Supabase Auth identifies
    the caller, while the local `users` table stores the application
    profile attached to that Auth identity.
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        audit_event(
            action="auth.denied",
            tool="auth",
            result="missing Supabase bearer token",
            success=False,
        )
        raise HTTPException(status_code=401, detail="Token Supabase manquant (Authorization: Bearer).")

    try:
        auth_response = get_supabase_admin_client().auth.get_user(credentials.credentials)
    except Exception as exc:  # noqa: BLE001 - Supabase auth exceptions vary by SDK version.
        audit_event(
            action="auth.denied",
            tool="auth",
            result="invalid Supabase bearer token",
            success=False,
        )
        logger.warning("Validation du token Supabase echouee : %s", exc)
        raise HTTPException(status_code=401, detail="Token Supabase invalide.") from exc

    auth_user = _extract_supabase_auth_user(auth_response)
    if auth_user is None:
        raise HTTPException(status_code=401, detail="Token Supabase invalide ou expire.")

    try:
        return UserService().sync_auth_user(_auth_identity_from_supabase_user(auth_user))
    except (UserServiceError, ValueError) as exc:
        logger.exception("Synchronisation utilisateur Auth/Profile echouee.")
        raise HTTPException(status_code=500, detail=f"Synchronisation utilisateur impossible : {exc}") from exc


def _extract_supabase_auth_user(auth_response: Any) -> Any | None:
    """Return the user object from Supabase Auth responses across SDK shapes."""
    if isinstance(auth_response, dict):
        return auth_response.get("user")
    return getattr(auth_response, "user", None)


def _auth_identity_from_supabase_user(auth_user: Any) -> AuthUserIdentity:
    """Normalize a Supabase Auth user into the local sync contract."""
    user_id = _get_auth_value(auth_user, "id")
    email = _get_auth_value(auth_user, "email")
    metadata = _get_auth_value(auth_user, "user_metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    full_name = metadata.get("full_name") or metadata.get("name") or metadata.get("display_name")
    profile = UserProfile(
        full_name=full_name,
        avatar_url=metadata.get("avatar_url") or metadata.get("picture"),
        metadata=metadata,
    )
    return AuthUserIdentity(
        id=user_id,
        email=email,
        name=full_name,
        profile=profile,
    )


def _get_auth_value(auth_user: Any, field_name: str) -> Any:
    """Read a field from Supabase Auth user objects or dicts."""
    if isinstance(auth_user, dict):
        return auth_user.get(field_name)
    return getattr(auth_user, field_name, None)


def require_chat_user(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer_auth),
    api_key: str | None = Security(_api_key_header),
) -> User:
    """Authentification pour POST /chat/ : accepte Bearer Supabase OU X-API-Key admin.

    Chemin 1 — Bearer présent : délègue à require_supabase_user (comportement inchangé).
    Chemin 2 — X-API-Key valide : retourne un User admin synthétique (mode SuperAdmin).
    Chemin 3 — Aucun credential valide : 401.
    """
    if credentials is not None and credentials.scheme.lower() == "bearer":
        return require_supabase_user(credentials)

    if not settings.ADMIN_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Authentification non configurée côté serveur (ADMIN_API_KEY manquant).",
        )
    if api_key and hmac.compare_digest(api_key, settings.ADMIN_API_KEY):
        return User(id="superadmin", name="Super Admin", role=UserRole.ADMIN)

    audit_event(
        action="auth.denied",
        tool="auth",
        result="missing Bearer token and invalid or absent X-API-Key",
        success=False,
    )
    raise HTTPException(
        status_code=401,
        detail="Authentification requise : Bearer Supabase ou X-API-Key admin (en-tête X-API-Key).",
    )


# Composite dependency reused on sensitive internal endpoints:
# rate limiting first, then local API-key authentication.
SECURE = [Depends(enforce_rate_limit), Depends(require_api_key)]

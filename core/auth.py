"""Authentication dependencies for MakenBrain endpoints.

Two authentication modes coexist:
- local admin API key via `X-API-Key` for sensitive internal endpoints;
- Supabase bearer token for future user-facing endpoints that must sync
  the authenticated user with the local `users` profile table.

The local admin gate keeps its fail-closed behavior: if `ADMIN_API_KEY` is
not configured, protected endpoints refuse access instead of staying open.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from typing import Any

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from core.audit import audit_event
from core.config import settings
from core.exceptions import SupabaseConfigError
from core.rate_limit import enforce_rate_limit
from core.supabase_client import get_supabase_admin_client
from core.user_service import UserService, UserServiceError
from models.user import AuthUserIdentity, User, UserProfile, UserRole

# Exceptions levées uniquement quand Supabase a réellement examiné le token
# et l'a rejeté. Toute autre exception (connexion HTTP/2 interrompue, timeout,
# erreur socket, indisponibilité temporaire) ne prouve PAS que le token est
# invalide et ne doit jamais produire un 401 — voir require_supabase_user.
try:
    from supabase_auth.errors import (
        AuthApiError,
        AuthInvalidJwtError,
        AuthSessionMissingError,
    )
except ImportError:  # anciennes versions du SDK : paquet "gotrue"
    from gotrue.errors import (  # type: ignore[no-redef]
        AuthApiError,
        AuthInvalidJwtError,
        AuthSessionMissingError,
    )

_TOKEN_REJECTED_ERRORS = (AuthApiError, AuthInvalidJwtError, AuthSessionMissingError)

logger = logging.getLogger("makenbrain.auth")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_auth = HTTPBearer(auto_error=False)

# ── Cache de validation de token ──────────────────────────────────────────────
# Chaque requête authentifiée revalidait le JWT auprès de Supabase (réseau).
# Sur réseau instable, cela transformait CHAQUE requête en pari — d'où les
# rafales de 503. Un token validé avec succès est réutilisé :
#   - < 5 min  : sans aucune revalidation réseau (frais) ;
#   - < 30 min : uniquement si Supabase est injoignable (stale-while-error).
# Compromis de sécurité assumé : une session révoquée côté serveur peut
# rester utilisable jusqu'à 5 min (30 min pendant une panne Supabase).
# Le logout local purge le token du navigateur — plus aucune requête ne le
# porte. Les tokens sont indexés par empreinte SHA-256, jamais en clair.
_TOKEN_CACHE_TTL_FRESH = 300.0
_TOKEN_CACHE_TTL_STALE = 1800.0
_TOKEN_CACHE_MAX = 500
_token_cache: dict[str, tuple[float, User]] = {}
_token_cache_lock = threading.Lock()


def _token_cache_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _token_cache_get(token: str, max_age: float) -> User | None:
    key = _token_cache_key(token)
    with _token_cache_lock:
        entry = _token_cache.get(key)
    if entry and (time.monotonic() - entry[0]) <= max_age:
        return entry[1].model_copy(deep=True)
    return None


def _token_cache_put(token: str, user: User) -> None:
    with _token_cache_lock:
        if len(_token_cache) >= _TOKEN_CACHE_MAX:
            oldest = min(_token_cache, key=lambda k: _token_cache[k][0])
            _token_cache.pop(oldest, None)
        _token_cache[_token_cache_key(token)] = (time.monotonic(), user.model_copy(deep=True))


def _token_cache_drop(token: str) -> None:
    with _token_cache_lock:
        _token_cache.pop(_token_cache_key(token), None)


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

    # ── Cache frais : token validé il y a < 5 min → aucun appel réseau ──────
    cached = _token_cache_get(credentials.credentials, _TOKEN_CACHE_TTL_FRESH)
    if cached is not None:
        return cached

    # Deux familles d'échec distinctes :
    # - le token est réellement rejeté par Supabase          → 401 (auth refusée)
    # - erreur réseau/transport vers Supabase (HTTP/2 fermé,
    #   timeout, socket, indisponibilité temporaire)          → 503 après un retry
    # Un 401 sur erreur réseau déclenchait la déconnexion automatique côté
    # frontend alors que la session était valide.
    auth_response = None
    last_transport_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            auth_response = get_supabase_admin_client().auth.get_user(credentials.credentials)
            break
        except _TOKEN_REJECTED_ERRORS as exc:
            _token_cache_drop(credentials.credentials)
            audit_event(
                action="auth.denied",
                tool="auth",
                result="invalid Supabase bearer token",
                success=False,
            )
            logger.warning("Token Supabase rejete par Supabase Auth : %s", exc)
            raise HTTPException(status_code=401, detail="Token Supabase invalide.") from exc
        except Exception as exc:  # noqa: BLE001 - erreurs transport/SDK variables selon versions.
            last_transport_exc = exc
            logger.warning(
                "Erreur reseau vers Supabase Auth (tentative %d/2) : %s", attempt, exc
            )

    if auth_response is None:
        # Supabase injoignable. Avant de répondre 503 : servir la session
        # depuis le cache stale (validée il y a < 30 min) — l'application
        # reste utilisable pendant les coupures réseau vers Supabase.
        stale = _token_cache_get(credentials.credentials, _TOKEN_CACHE_TTL_STALE)
        if stale is not None:
            logger.warning("Supabase Auth injoignable — session servie depuis le cache (stale).")
            return stale
        raise HTTPException(
            status_code=503,
            detail="Service d'authentification momentanement indisponible. Reessayez.",
        ) from last_transport_exc

    auth_user = _extract_supabase_auth_user(auth_response)
    if auth_user is None:
        raise HTTPException(status_code=401, detail="Token Supabase invalide ou expire.")

    identity = _auth_identity_from_supabase_user(auth_user)
    try:
        user = UserService().sync_auth_user(identity)
    except (UserServiceError, ValueError, SupabaseConfigError) as exc:
        # Le token est VALIDE (Supabase vient de le confirmer) : un échec de
        # synchronisation du profil (réseau vers la table users) ne doit pas
        # bloquer la requête. Profil minimal dérivé du JWT — le rôle vient
        # de app_metadata ci-dessous, l'autorisation n'est pas affaiblie.
        logger.warning("Sync profil indisponible — profil derive du JWT : %s", exc)
        user = User(
            id=identity.id,
            email=identity.email,
            full_name=identity.full_name,
            avatar_url=identity.avatar_url,
            role=UserRole.USER,
        )

    # Priorité au rôle encodé dans app_metadata (inviolable côté client).
    role = _extract_role_from_auth_user(auth_user)
    if role is not None:
        user.role = role

    _token_cache_put(credentials.credentials, user)
    return user


def _extract_supabase_auth_user(auth_response: Any) -> Any | None:
    """Return the user object from Supabase Auth responses across SDK shapes."""
    if isinstance(auth_response, dict):
        return auth_response.get("user")
    return getattr(auth_response, "user", None)


def _extract_role_from_auth_user(auth_user: Any) -> UserRole | None:
    """Extrait le rôle depuis app_metadata (positionné côté serveur, inviolable).

    Supabase encode le rôle dans `app_metadata.role`. Seul le service role
    peut le modifier — le client ne peut pas s'auto-promouvoir.
    """
    app_meta = _get_auth_value(auth_user, "app_metadata") or {}
    if not isinstance(app_meta, dict):
        return None
    role_str = app_meta.get("role", "")
    try:
        return UserRole(role_str)
    except ValueError:
        return None


def _auth_identity_from_supabase_user(auth_user: Any) -> AuthUserIdentity:
    """Normalize a Supabase Auth user into the local sync contract.

    Maps OAuth user_metadata fields to the exact SQL columns:
    full_name, avatar_url. username is derived from email in UserService.
    """
    user_id = _get_auth_value(auth_user, "id")
    email = _get_auth_value(auth_user, "email")
    metadata = _get_auth_value(auth_user, "user_metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    full_name = metadata.get("full_name") or metadata.get("name") or metadata.get("display_name")
    avatar_url = metadata.get("avatar_url") or metadata.get("picture")
    profile = UserProfile(
        full_name=full_name,
        avatar_url=avatar_url,
        metadata=metadata,
    )
    return AuthUserIdentity(
        id=user_id,
        email=email,
        full_name=full_name,
        avatar_url=avatar_url,
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
        return User(id="superadmin", full_name="Super Admin", role=UserRole.SUPERADMIN)

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


def require_admin_user(user: User = Depends(require_supabase_user)) -> User:
    """Guard FastAPI : exige le rôle ADMIN ou SUPERADMIN (JWT Supabase).

    Usage :
        @router.post("/sensitive")
        async def endpoint(user: User = Depends(require_admin_user)): ...
    """
    if user.role not in (UserRole.ADMIN, UserRole.SUPERADMIN):
        audit_event(action="auth.forbidden", tool="auth",
                    result=f"role {user.role} insuffisant (admin requis)", success=False)
        raise HTTPException(status_code=403, detail="Accès réservé aux administrateurs.")
    return user


def require_superadmin_user(user: User = Depends(require_supabase_user)) -> User:
    """Guard FastAPI : exige le rôle SUPERADMIN uniquement (JWT Supabase).

    Usage :
        @router.delete("/nuclear")
        async def endpoint(user: User = Depends(require_superadmin_user)): ...
    """
    if user.role != UserRole.SUPERADMIN:
        audit_event(action="auth.forbidden", tool="auth",
                    result=f"role {user.role} insuffisant (superadmin requis)", success=False)
        raise HTTPException(status_code=403, detail="Accès réservé au super-administrateur.")
    return user


# Composite dependency reused on sensitive internal endpoints:
# rate limiting first, then local API-key authentication.
SECURE = [Depends(enforce_rate_limit), Depends(require_api_key)]

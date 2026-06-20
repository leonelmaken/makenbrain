"""Local API key authentication for MakenBrain's sensitive endpoints.

A single admin key (`ADMIN_API_KEY`, set in `.env`) gates every endpoint
that can read sensitive logs, spend LLM credits autonomously, or mutate
files/memory: the file agent, the autonomous scheduler, the audit log and
memory writes. The key travels in the `X-API-Key` header.

Design choice -- fail closed: if `ADMIN_API_KEY` is not configured, every
protected endpoint refuses access (503) instead of silently staying open.
This forces an explicit security setup rather than an accidental one.
"""
from __future__ import annotations

import hmac
import logging

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from core.audit import audit_event
from core.config import settings
from core.rate_limit import enforce_rate_limit

logger = logging.getLogger("makenbrain.auth")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(_api_key_header)) -> str:
    """FastAPI dependency enforcing the local admin API key.

    Args:
        api_key: Value of the `X-API-Key` header, injected by FastAPI.

    Returns:
        The validated API key (rarely used by callers, but returned so the
        dependency can also be used with `Depends(require_api_key)` to
        retrieve the caller's key if ever needed).

    Raises:
        HTTPException: 503 if no ADMIN_API_KEY is configured server-side
            (fail-closed), 401 if the header is missing or does not match.
    """
    if not settings.ADMIN_API_KEY:
        logger.error("ADMIN_API_KEY non configurée — accès refusé par défaut (fail-closed).")
        raise HTTPException(
            status_code=503,
            detail="Authentification non configurée côté serveur (ADMIN_API_KEY manquant dans .env).",
        )
    if not api_key or not hmac.compare_digest(api_key, settings.ADMIN_API_KEY):
        audit_event(
            action="auth.denied",
            tool="auth",
            result="invalid or missing X-API-Key",
            success=False,
        )
        raise HTTPException(status_code=401, detail="Clé API invalide ou manquante (en-tête X-API-Key).")
    return api_key


# Dépendance composite réutilisée sur tous les endpoints sensibles :
# limitation de débit PUIS authentification, dans cet ordre — pour qu'une
# attaque par force brute sur la clé API soit elle aussi soumise au quota,
# avant même que la clé ne soit vérifiée.
SECURE = [Depends(enforce_rate_limit), Depends(require_api_key)]

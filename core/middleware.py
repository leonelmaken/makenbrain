"""Middleware HTTP pour MakenBrain — Phase 4.

RequestContextMiddleware :
    Injecte request_id, user_id, session_id et endpoint dans le ContextVar
    de core/observability/structured_logger.py au début de chaque requête.
    Enregistre également les métriques de latence et de succès HTTP à la fin.

    Le request_id est un UUID4 généré par requête. Si le client envoie
    l'en-tête X-Request-ID, la valeur fournie est utilisée à la place.

Ordre d'enregistrement dans main.py :
    Le middleware doit être ajouté APRÈS CORSMiddleware pour que les
    en-têtes Authorization soient disponibles lors de l'extraction du user_id.
"""
from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from core.observability import get_metrics, set_request_context


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware qui propage le contexte de requête dans les ContextVar asyncio.

    Pour chaque requête HTTP :
    1. Génère (ou lit) un request_id unique.
    2. Extrait user_id depuis le token JWT Supabase (si présent).
    3. Positionne le contexte via set_request_context().
    4. Après la réponse, enregistre endpoint + duration_ms + success dans MetricsCollector.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        # ── 1. request_id ─────────────────────────────────────────────────────
        request_id = (
            request.headers.get("X-Request-ID")
            or str(uuid.uuid4())
        )

        # ── 2. user_id (extrait du header Authorization si présent) ──────────
        user_id = _extract_user_id(request)

        # ── 3. session_id (en-tête optionnel fourni par le client) ───────────
        session_id = request.headers.get("X-Session-ID")

        # ── 4. Positionner le contexte ────────────────────────────────────────
        set_request_context(
            request_id = request_id,
            session_id = session_id,
            user_id    = user_id,
            endpoint   = request.url.path,
        )

        # ── 5. Exécuter la requête + mesurer la latence ───────────────────────
        t0 = time.monotonic()
        response = await call_next(request)
        duration_ms = round((time.monotonic() - t0) * 1000, 1)

        # ── 6. Injecter request_id dans les headers de réponse ────────────────
        response.headers["X-Request-ID"] = request_id

        # ── 7. Métriques ──────────────────────────────────────────────────────
        get_metrics().record_request(
            endpoint   = request.url.path,
            duration_ms= duration_ms,
            success    = response.status_code < 400,
        )

        return response


def _extract_user_id(request: Request) -> str | None:
    """Extrait le user_id du token JWT Supabase sans vérifier la signature.

    Cette extraction est "best-effort" et sert uniquement à enrichir les logs.
    La vérification d'authenticité réelle est faite par core/auth.py.

    Returns:
        Le champ 'sub' (user_id Supabase) ou None si absent/illisible.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None

    token = auth[len("Bearer "):]
    parts = token.split(".")
    if len(parts) != 3:
        return None

    try:
        import base64
        import json

        # Le payload JWT est la 2ème partie, encodé en base64url sans padding
        payload_b64 = parts[1]
        # Ajouter le padding manquant
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding

        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("sub")
    except Exception:
        return None

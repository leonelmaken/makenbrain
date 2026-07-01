"""Endpoint /health — Phase 4.

Fournit un check de santé complet de MakenBrain :
    GET /health         → statut global (rapide, sans probe réseau)
    GET /health/full    → statut détaillé avec probe des providers LLM

Design :
    - /health retourne 200 si l'application est opérationnelle (même si
      certains providers cloud sont down). Ce comportement est conforme
      aux standards Kubernetes readiness/liveness probes.
    - /health/full peut prendre 2-5 secondes (probe Ollama + Groq).
    - Aucune authentification requise (les ops doivent pouvoir checker).
"""
from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core.config import settings
from core.observability import get_metrics
from core.provider_layer import get_router
from core.version import APP_NAME, APP_VERSION, APP_RELEASE_NAME

router = APIRouter(tags=["Health"])

_startup_time = time.monotonic()


@router.get("/health", summary="Health check rapide")
async def health_check() -> JSONResponse:
    """Check de santé rapide — répond en < 50ms.

    Retourne toujours 200 si l'application a démarré correctement.
    Inclut les métriques agrégées depuis le démarrage.

    Returns:
        JSON avec :
            status        : "ok"
            version       : version de l'application
            uptime_seconds: durée depuis le démarrage du serveur
            metrics       : snapshot des métriques in-memory
    """
    metrics_snapshot = get_metrics().snapshot()

    return JSONResponse(
        content={
            "status"        : "ok",
            "app"           : APP_NAME,
            "version"       : APP_VERSION,
            "release"       : APP_RELEASE_NAME,
            "uptime_seconds": round(time.monotonic() - _startup_time, 1),
            "metrics"       : metrics_snapshot,
        },
        status_code=200,
    )


@router.get("/health/full", summary="Health check complet avec probe providers")
async def health_check_full() -> JSONResponse:
    """Check de santé complet — peut prendre 2-5 secondes.

    Probe chaque provider LLM (Ollama, Groq) pour vérifier
    leur disponibilité et mesurer leur latence actuelle.

    Returns:
        JSON avec :
            status    : "ok" | "degraded" (si au moins un provider down)
            providers : statut détaillé de chaque provider
            config    : paramètres de configuration non-sensibles
            metrics   : snapshot des métriques in-memory
    """
    t0 = time.monotonic()

    router_instance = get_router()
    provider_healths = await router_instance.health_all()

    all_available = all(h.available for h in provider_healths)
    at_least_one  = any(h.available for h in provider_healths)

    if at_least_one:
        status      = "ok" if all_available else "degraded"
        status_code = 200
    else:
        status      = "critical"
        status_code = 503

    probe_duration = round((time.monotonic() - t0) * 1000, 1)

    return JSONResponse(
        content={
            "status"          : status,
            "app"             : APP_NAME,
            "version"         : APP_VERSION,
            "release"         : APP_RELEASE_NAME,
            "uptime_seconds"  : round(time.monotonic() - _startup_time, 1),
            "probe_duration_ms": probe_duration,
            "providers"       : [h.to_dict() for h in provider_healths],
            "config"          : {
                "ollama_host"  : settings.OLLAMA_HOST,
                "ollama_model" : settings.OLLAMA_MODEL,
                "groq_configured": bool(settings.GROQ_API_KEY),
                "supabase_configured": bool(settings.SUPABASE_URL),
            },
            "metrics"         : get_metrics().snapshot(),
        },
        status_code=status_code,
    )

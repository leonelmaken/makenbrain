"""Centralized Supabase client for MakenBrain.

Single entry point for any Supabase access in the project. Nothing else
should call `create_client()` directly — import `get_supabase_client()`
(or `get_supabase_admin_client()` for trusted server-side access) from
this module instead.

Scope (Phase 2.1): client construction, configuration validation and a
health check. No business logic, no authentication flow, no FastAPI
routes live here — those are explicitly out of scope for this phase.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import httpx
from supabase import Client, create_client
from supabase.lib.client_options import SyncClientOptions

from core.config import settings
from core.exceptions import SupabaseConfigError


def _build_http_client() -> httpx.Client:
    """Client HTTP injecté dans Supabase — HTTP/1.1 forcé, connexions robustes.

    Le client httpx par défaut de supabase-py active HTTP/2, qui provoque des
    erreurs récurrentes sur réseau instable : ConnectionTerminated,
    "Received pseudo-header in trailer", handshakes SSL expirés, codes h2
    cryptiques (11, 13). HTTP/1.1 + timeouts explicites + keepalive court
    (les connexions inactives > 30 s sont jetées au lieu d'être réutilisées
    mortes) éliminent ces familles d'erreurs à la racine.
    """
    # connect=20s : les handshakes SSL peuvent être très lents sur ce réseau
    # (observé en conditions réelles) — un timeout court transformait une
    # connexion lente mais viable en erreur.
    return httpx.Client(
        http2=False,
        timeout=httpx.Timeout(25.0, connect=20.0),
        limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=30.0),
    )

logger = logging.getLogger("makenbrain.supabase")

# Table utilisée pour le ping de santé : la plus petite des quatre tables
# existantes (users, projects, audit_logs, user_settings). Le RLS étant
# actif, une réponse même vide (0 ligne visible) prouve déjà que la
# connexion et les identifiants sont valides.
_HEALTHCHECK_TABLE = "user_settings"


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Return a cached Supabase client authenticated with the anon key.

    Use this for any access that should respect Row Level Security
    policies, exactly as an anonymous or end-user caller would.

    Returns:
        A cached `supabase.Client` instance (built once per process).

    Raises:
        SupabaseConfigError: if SUPABASE_URL or SUPABASE_ANON_KEY is
            missing from the configuration.
    """
    if not settings.SUPABASE_URL or not settings.SUPABASE_ANON_KEY:
        raise SupabaseConfigError(
            "Configuration Supabase incomplète : SUPABASE_URL et "
            "SUPABASE_ANON_KEY sont requis dans .env."
        )
    logger.info("Initialisation du client Supabase (anon key, HTTP/1.1).")
    return create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_ANON_KEY,
        options=SyncClientOptions(httpx_client=_build_http_client()),
    )


@lru_cache(maxsize=1)
def get_supabase_admin_client() -> Client:
    """Return a cached Supabase client authenticated with the service-role key.

    This client bypasses Row Level Security entirely. Reserve it for
    trusted, server-side-only operations (background jobs, maintenance
    scripts). Never expose this client, or the key behind it, to
    anything client-facing.

    Returns:
        A cached `supabase.Client` instance with elevated privileges.

    Raises:
        SupabaseConfigError: if SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY
            is missing from the configuration.
    """
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise SupabaseConfigError(
            "Configuration Supabase admin incomplète : SUPABASE_URL et "
            "SUPABASE_SERVICE_ROLE_KEY sont requis dans .env."
        )
    logger.info("Initialisation du client Supabase (service role key, HTTP/1.1).")
    return create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_SERVICE_ROLE_KEY,
        options=SyncClientOptions(httpx_client=_build_http_client()),
    )


def test_connection() -> dict[str, bool | str]:
    """Verify that Supabase responds, without reading or writing real data.

    Performs a zero-row `count` request against `user_settings` using the
    anon client. Even a result filtered down to nothing by Row Level
    Security still proves the round trip and credentials are valid — only
    a network failure or an authentication error counts as "not connected".

    This function is designed to be safe to call from health checks or
    tests: it never raises. Every failure mode (missing config, network
    error, Supabase-side error) is caught and reported in the result.

    Returns:
        A dict with:
            - "connected" (bool): whether the request round-tripped.
            - "message" (str): human-readable detail, useful for logs
              or a future `/health` endpoint.
    """
    try:
        client = get_supabase_client()
    except SupabaseConfigError as exc:
        logger.error("Test de connexion Supabase impossible : %s", exc)
        return {"connected": False, "message": str(exc)}

    try:
        client.table(_HEALTHCHECK_TABLE).select("*", count="exact").limit(0).execute()
    except Exception as exc:  # noqa: BLE001 — health check : toute erreur réseau/SDK doit être capturée ici, pas propagée
        logger.error("Connexion Supabase échouée : %s", exc)
        return {"connected": False, "message": f"Connexion Supabase échouée : {exc}"}

    logger.info("Connexion Supabase OK.")
    return {"connected": True, "message": "Connexion Supabase opérationnelle."}

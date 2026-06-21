"""Phase 2.1 tests: Supabase client construction, config loading, health check.

These tests deliberately avoid asserting on live network reachability —
only that configuration is loaded, the client builds, and test_connection()
degrades gracefully instead of raising. That keeps the suite stable even
if Supabase or the network is temporarily unavailable.
"""
from __future__ import annotations

import pytest

from core.config import settings
from core.exceptions import SupabaseConfigError
from core.supabase_client import get_supabase_client
from core.supabase_client import test_connection as check_supabase_connection


def test_supabase_configuration_is_loaded() -> None:
    """SUPABASE_URL and SUPABASE_ANON_KEY must be exposed by core/config.py.

    This only checks that the values already set in .env are visible
    through `settings` — it does not validate the credentials themselves.
    """
    assert settings.SUPABASE_URL, "SUPABASE_URL manquant dans la configuration."
    assert settings.SUPABASE_ANON_KEY, "SUPABASE_ANON_KEY manquant dans la configuration."


def test_get_supabase_client_builds_successfully() -> None:
    """The cached client factory must return a usable client instance."""
    client = get_supabase_client()
    assert client is not None
    # Le SDK Supabase expose toujours .table() : signe que l'objet retourné
    # est bien un client Supabase fonctionnel, pas un mock ou None.
    assert hasattr(client, "table")


def test_get_supabase_client_is_cached() -> None:
    """Repeated calls must return the same cached instance (lru_cache)."""
    assert get_supabase_client() is get_supabase_client()


def test_get_supabase_client_fails_closed_without_url(monkeypatch) -> None:
    """Client construction raises a clear, specific error if URL is unset."""
    from core import supabase_client as sc

    sc.get_supabase_client.cache_clear()
    monkeypatch.setattr(settings, "SUPABASE_URL", "")
    try:
        with pytest.raises(SupabaseConfigError):
            sc.get_supabase_client()
    finally:
        sc.get_supabase_client.cache_clear()  # ne pas polluer les tests suivants


def test_connection_never_raises() -> None:
    """check_supabase_connection() must always return a dict, never raise.

    Whether Supabase is actually reachable from this machine at test time
    is not asserted here on purpose — that would make the test fragile
    and dependent on live network conditions.

    Imported under the alias `check_supabase_connection` rather than its
    real name `test_connection`: pytest auto-discovers any `test_*` name
    that ends up in a test module's namespace, so importing it under its
    real name would make pytest collect and execute it a second time as
    an accidental extra test case.
    """
    result = check_supabase_connection()
    assert isinstance(result, dict)
    assert isinstance(result.get("connected"), bool)
    assert isinstance(result.get("message"), str) and result["message"]

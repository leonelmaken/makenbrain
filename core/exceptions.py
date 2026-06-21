"""Lightweight custom exceptions for MakenBrain.

Kept intentionally minimal: one exception per genuinely distinct failure
mode that callers need to catch specifically. Generic, unstructured
errors should keep using Python's built-in exceptions instead of growing
this file unnecessarily.
"""
from __future__ import annotations


class SupabaseConfigError(RuntimeError):
    """Raised when required Supabase settings are missing or invalid.

    Distinct from a connection failure: this means the client could not
    even be constructed because configuration is incomplete (e.g. a
    missing SUPABASE_URL or SUPABASE_ANON_KEY in `.env`), not that the
    Supabase service itself is unreachable.
    """

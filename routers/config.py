"""Endpoint de configuration publique — Consolidation v1.0.

Expose uniquement les clés publiques nécessaires au frontend pour
initialiser le client Supabase JS (OAuth, auth state).
Aucun secret (service_role_key, ADMIN_API_KEY) n'est exposé ici.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from core.config import settings

router = APIRouter()


class PublicConfig(BaseModel):
    supabase_url      : str
    supabase_anon_key : str
    app_name          : str = "MakenBrain"


@router.get("/public", response_model=PublicConfig, tags=["⚙️ Config"])
def get_public_config() -> PublicConfig:
    """Retourne la configuration publique nécessaire au frontend.

    Ne contient que des clés ANON (conçues pour être exposées côté client).
    Accessible sans authentification.
    """
    return PublicConfig(
        supabase_url      = settings.SUPABASE_URL,
        supabase_anon_key = settings.SUPABASE_ANON_KEY,
    )

"""Profil métier de l'utilisateur connecté — Phase 10.

Voir, définir ou faire détecter automatiquement son domaine/profession.
Le profil pilote l'adaptation des réponses (prompt système) et, à terme,
les équipes d'agents par métier.

Isolation : user_id vient toujours du JWT — jamais du corps de requête.
Ne pas confondre avec /identity/* (profil local historique du SuperAdmin).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.auth import require_chat_user
from core.user_profiles import KNOWN_DOMAINS, detect_domain, get_profile, set_profile
from models.user import User

router = APIRouter()


class ProfileUpdateRequest(BaseModel):
    domain: str | None = Field(default=None, min_length=1, max_length=64)
    profession: str | None = Field(default=None, min_length=1, max_length=120)
    expertise_level: str | None = Field(default=None, min_length=1, max_length=32)


@router.get("")
async def read_my_profile(current_user: User = Depends(require_chat_user)):
    """Profil métier de l'utilisateur connecté + liste des domaines valides."""
    return {
        "profile": get_profile(str(current_user.id)),
        "known_domains": list(KNOWN_DOMAINS),
    }


@router.put("")
async def update_my_profile(
    req: ProfileUpdateRequest,
    current_user: User = Depends(require_chat_user),
):
    """Définit le profil manuellement — prioritaire sur la détection auto."""
    changes = req.model_dump(exclude_unset=True, exclude_none=True)
    if not changes:
        raise HTTPException(status_code=400, detail="Aucun champ fourni.")
    if "domain" in changes and changes["domain"] not in KNOWN_DOMAINS:
        raise HTTPException(
            status_code=400,
            detail=f"Domaine inconnu. Valides : {', '.join(KNOWN_DOMAINS)}",
        )
    profile = set_profile(str(current_user.id), source="manual", **changes)
    return {"profile": profile, "message": "Profil mis à jour."}


@router.post("/detect")
async def detect_my_profile(current_user: User = Depends(require_chat_user)):
    """Lance la détection automatique du domaine depuis mémoires + conversations."""
    profile = await detect_domain(str(current_user.id), force=True)
    if not profile.get("domain"):
        return {
            "profile": profile,
            "message": (
                "Détection non concluante — pas encore assez de conversations "
                "ou de souvenirs. Tu peux définir ton domaine manuellement."
            ),
        }
    return {"profile": profile, "message": "Domaine détecté."}

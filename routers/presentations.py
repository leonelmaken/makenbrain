"""Génération de présentations avec paramètres du wizard — Phase 10.

Le frontend interroge l'utilisateur (thème, police, nombre de diapositives,
texte fourni, langue, images IA) puis appelle ce endpoint. Chaque champ est
optionnel : MakenBrain choisit des valeurs intelligentes par défaut ("Passer").

Retourne les URLs de l'aperçu HTML animé et de l'export .pptx — le frontend
affiche l'aperçu AVANT tout téléchargement.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.agents.presenter_agent import FONTS, THEMES, generate_presentation
from core.auth import require_chat_user
from core.usage import enforce_quota
from models.user import User

router = APIRouter()


class PresentationRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=500)
    slides_count: int | None = Field(default=None, ge=3, le=15)
    theme: str = Field(default="makenbrain")
    font: str = Field(default="moderne")
    with_images: bool = True
    source_text: str | None = Field(default=None, max_length=20000)
    language: str | None = Field(default=None, max_length=32)


@router.get("/options")
async def presentation_options(current_user: User = Depends(require_chat_user)):
    """Thèmes et polices disponibles (pour construire le wizard)."""
    return {
        "themes": {name: {"label": cfg["label"], "accent": cfg["accent"], "bg": cfg["bg"]}
                   for name, cfg in THEMES.items()},
        "fonts": list(FONTS.keys()),
    }


@router.post("/generate")
async def create_presentation(
    req: PresentationRequest,
    current_user: User = Depends(require_chat_user),
):
    """Génère le deck HTML animé + l'export .pptx et retourne leurs URLs."""
    enforce_quota(current_user)
    if req.theme not in THEMES:
        raise HTTPException(status_code=400, detail=f"Thème inconnu. Valides : {', '.join(THEMES)}")
    if req.font not in FONTS:
        raise HTTPException(status_code=400, detail=f"Police inconnue. Valides : {', '.join(FONTS)}")

    try:
        result = await generate_presentation(
            topic       = req.topic,
            slides_count= req.slides_count,
            theme       = req.theme,
            font        = req.font,
            with_images = req.with_images,
            source_text = req.source_text,
            language    = req.language,
        )
    except Exception as exc:  # noqa: BLE001 — réponse propre plutôt que 500 brut.
        raise HTTPException(
            status_code=503,
            detail=f"Génération impossible pour le moment : {exc}",
        ) from exc
    return result

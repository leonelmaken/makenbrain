"""Mémoire personnelle visible et éditable — Phase 9 (confiance & différenciation).

Chaque utilisateur authentifié peut voir, ajouter, corriger et supprimer ce
que MakenBrain sait de lui (mémoires stockées dans public.user_memories,
alimentées automatiquement par le chat ou ajoutées manuellement ici).

Isolation stricte : `user_id` provient TOUJOURS de `current_user.id` (JWT),
jamais du corps de la requête — un utilisateur ne peut ni voir ni modifier
la mémoire d'un autre, quel que soit son rôle.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.auth import require_chat_user
from core.user_memory_service import (
    UserMemoryNotFoundError,
    UserMemoryServiceError,
    create_memory,
    delete_memory,
    get_user_memories,
    update_memory,
)
from models.user import User
from models.user_memory import UserMemorySource, UserMemoryUpdate

router = APIRouter()

_UNAVAILABLE_DETAIL = (
    "Mémoire personnelle momentanément indisponible. "
    "Réessaie dans un instant."
)


class MemoryCreateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    source: str = Field(default=UserMemorySource.MANUAL.value, min_length=1, max_length=64)


@router.get("")
async def list_my_memories(current_user: User = Depends(require_chat_user)):
    """Retourne toutes les mémoires de l'utilisateur connecté, plus récentes d'abord."""
    try:
        memories = get_user_memories(current_user.id)
    except UserMemoryServiceError as exc:
        raise HTTPException(status_code=503, detail=_UNAVAILABLE_DETAIL) from exc
    memories.sort(key=lambda m: m.get("created_at") or "", reverse=True)
    return {"memories": memories, "total": len(memories)}


@router.post("")
async def add_memory_manual(
    req: MemoryCreateRequest,
    current_user: User = Depends(require_chat_user),
):
    """Ajoute une mémoire manuelle — l'utilisateur enseigne directement MakenBrain."""
    try:
        memory = create_memory(
            user_id=current_user.id,
            content=req.content,
            source=req.source,
        )
    except ValueError as exc:
        # Inclut les erreurs de validation Pydantic (ex. identifiant non-UUID
        # du mode développeur X-API-Key, qui n'a pas de mémoire personnelle).
        raise HTTPException(
            status_code=400,
            detail="Mémoire personnelle indisponible pour ce type de compte.",
        ) from exc
    except UserMemoryServiceError as exc:
        raise HTTPException(status_code=503, detail=_UNAVAILABLE_DETAIL) from exc
    return memory


@router.patch("/{memory_id}")
async def edit_memory(
    memory_id: str,
    req: UserMemoryUpdate,
    current_user: User = Depends(require_chat_user),
):
    """Corrige une mémoire — uniquement si elle appartient à l'utilisateur connecté."""
    try:
        memory = update_memory(current_user.id, memory_id, req)
    except UserMemoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Mémoire introuvable.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UserMemoryServiceError as exc:
        raise HTTPException(status_code=503, detail=_UNAVAILABLE_DETAIL) from exc
    return memory


@router.delete("/{memory_id}")
async def remove_memory(
    memory_id: str,
    current_user: User = Depends(require_chat_user),
):
    """Supprime définitivement une mémoire de l'utilisateur connecté."""
    try:
        delete_memory(current_user.id, memory_id)
    except UserMemoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Mémoire introuvable.") from exc
    except UserMemoryServiceError as exc:
        raise HTTPException(status_code=503, detail=_UNAVAILABLE_DETAIL) from exc
    return {"message": "Mémoire supprimée."}

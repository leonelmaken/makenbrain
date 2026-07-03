"""
Endpoints historique des conversations — isolation par utilisateur.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel

from core.auth import require_chat_user
from core.chat_history import (
    create_session, get_session, get_session_for_user, list_sessions,
    delete_session, generate_smart_topic, update_session_topic
)
from models.user import User

router = APIRouter()


class NewSessionResponse(BaseModel):
    session_id: str


@router.post("/new", response_model=NewSessionResponse)
async def new_session(current_user: User = Depends(require_chat_user)):
    """Crée une nouvelle session de conversation pour l'utilisateur connecté."""
    session_id = create_session(user_id=str(current_user.id))
    return {"session_id": session_id}


@router.get("/list")
async def get_sessions(
    limit: int = 30,
    query: Optional[str] = None,
    current_user: User = Depends(require_chat_user),
):
    """Liste uniquement les conversations de l'utilisateur connecté."""
    sessions = list_sessions(user_id=str(current_user.id), limit=limit)
    if query:
        q = query.lower()
        sessions = [s for s in sessions if q in s["topic"].lower() or q in s.get("preview", "").lower()]
    return {"sessions": sessions}


@router.get("/{session_id}")
async def get_one_session(
    session_id: str,
    current_user: User = Depends(require_chat_user),
):
    """Récupère une conversation uniquement si elle appartient à l'utilisateur connecté."""
    session = get_session_for_user(session_id, str(current_user.id))
    if not session:
        raise HTTPException(404, f"Session {session_id} introuvable.")
    return session


@router.delete("/{session_id}")
async def remove_session(
    session_id: str,
    current_user: User = Depends(require_chat_user),
):
    """Supprime une conversation en vérifiant la propriété."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session introuvable.")
    owner = session.get("user_id")
    if owner and str(owner) != str(current_user.id):
        raise HTTPException(403, "Vous ne pouvez pas supprimer la session d'un autre utilisateur.")
    delete_session(session_id)
    return {"message": "Session supprimée."}


class RenameRequest(BaseModel):
    topic: str


@router.patch("/{session_id}")
async def rename_session(
    session_id: str,
    req: RenameRequest,
    current_user: User = Depends(require_chat_user),
):
    """Renomme une conversation en vérifiant la propriété."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session introuvable.")
    owner = session.get("user_id")
    if owner and str(owner) != str(current_user.id):
        raise HTTPException(403, "Vous ne pouvez pas renommer la session d'un autre utilisateur.")
    update_session_topic(session_id, req.topic)
    return {"message": "Titre mis à jour.", "topic": req.topic}


@router.post("/{session_id}/retitle")
async def retitle_session(
    session_id: str,
    current_user: User = Depends(require_chat_user),
):
    """Force la régénération du titre via LLM (après vérification de propriété)."""
    session = get_session_for_user(session_id, str(current_user.id))
    if not session:
        raise HTTPException(404, "Session introuvable.")
    title = await generate_smart_topic(session_id)
    if not title:
        raise HTTPException(400, "Pas assez de messages pour générer un titre.")
    return {"session_id": session_id, "topic": title}

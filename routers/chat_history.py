"""
Endpoints historique des conversations.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from core.chat_history import (
    create_session, get_session, list_sessions,
    delete_session, generate_smart_topic, update_session_topic
)

router = APIRouter()


class NewSessionResponse(BaseModel):
    session_id: str


@router.post("/new", response_model=NewSessionResponse)
async def new_session():
    """Crée une nouvelle session de conversation."""
    session_id = create_session()
    return {"session_id": session_id}


@router.get("/list")
async def get_sessions(limit: int = 30, query: Optional[str] = None):
    """Liste toutes les conversations passées, avec option de recherche."""
    sessions = list_sessions(limit)
    if query:
        q = query.lower()
        sessions = [s for s in sessions if q in s['topic'].lower() or q in s.get('preview', '').lower()]
    return {"sessions": sessions}


@router.get("/{session_id}")
async def get_one_session(session_id: str):
    """Récupère une conversation complète avec tous ses messages."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} introuvable.")
    return session


@router.delete("/{session_id}")
async def remove_session(session_id: str):
    """Supprime définitivement une conversation."""
    if delete_session(session_id):
        return {"message": "Session supprimée."}
    raise HTTPException(404, "Session introuvable.")

class RenameRequest(BaseModel):
    topic: str

@router.patch("/{session_id}")
async def rename_session(session_id: str, req: RenameRequest):
    """Renomme manuellement une conversation."""
    if update_session_topic(session_id, req.topic):
        return {"message": "Titre mis à jour.", "topic": req.topic}
    raise HTTPException(404, "Session introuvable.")


@router.post("/{session_id}/retitle")
async def retitle_session(session_id: str):
    """Force la régénération du titre via LLM."""
    title = await generate_smart_topic(session_id)
    if not title:
        raise HTTPException(400, "Pas assez de messages pour générer un titre.")
    return {"session_id": session_id, "topic": title}

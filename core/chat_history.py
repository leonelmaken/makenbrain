"""
Historique des conversations — Phase 6+
Chaque session de chat est sauvegardée avec un sujet auto-détecté.
MAKEN peut retrouver et reprendre n'importe quelle conversation passée.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

SESSIONS_FILE = Path("brain_data/chat_sessions.json")


def _load() -> dict:
    if SESSIONS_FILE.exists():
        try:
            return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save(data: dict) -> None:
    SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSIONS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _quick_topic(first_message: str) -> str:
    """Génère un titre de sujet rapide sans LLM (heuristique)."""
    words = first_message.strip().split()
    topic = " ".join(words[:6])
    return topic[:60] + ("..." if len(words) > 6 else "")


def create_session() -> str:
    """Crée une nouvelle session de chat. Retourne son ID."""
    sessions = _load()
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = {
        "id":          session_id,
        "topic":       "Nouvelle conversation",
        "created_at":  datetime.now().isoformat(),
        "updated_at":  datetime.now().isoformat(),
        "messages":    [],
    }
    _save(sessions)
    return session_id


def add_message(session_id: str, role: str, content: str) -> None:
    """Ajoute un message à une session. Crée la session si inexistante."""
    sessions = _load()
    if session_id not in sessions:
        sessions[session_id] = {
            "id":         session_id,
            "topic":      "Nouvelle conversation",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages":   [],
        }

    session = sessions[session_id]
    session["messages"].append({
        "role":      role,
        "content":   content,
        "timestamp": datetime.now().isoformat(),
    })
    session["updated_at"] = datetime.now().isoformat()

    # Auto-générer le sujet sur le premier message utilisateur
    if role == "user" and session["topic"] == "Nouvelle conversation":
        session["topic"] = _quick_topic(content)

    _save(sessions)


def get_session(session_id: str) -> Optional[dict]:
    sessions = _load()
    return sessions.get(session_id)


def list_sessions(limit: int = 30) -> list[dict]:
    """Liste les sessions, plus récentes en premier."""
    sessions = _load()
    items = sorted(sessions.values(), key=lambda s: s["updated_at"], reverse=True)
    return [
        {
            "id":            s["id"],
            "topic":         s["topic"],
            "created_at":    s["created_at"],
            "updated_at":    s["updated_at"],
            "message_count": len(s["messages"]),
            "preview":       s["messages"][0]["content"][:80] if s["messages"] else "",
        }
        for s in items[:limit]
    ]


def delete_session(session_id: str) -> bool:
    sessions = _load()
    if session_id in sessions:
        del sessions[session_id]
        _save(sessions)
        return True
    return False


def update_session_topic(session_id: str, new_topic: str) -> bool:
    """Met à jour manuellement le titre d'une session."""
    sessions = _load()
    if session_id in sessions:
        sessions[session_id]["topic"] = new_topic
        sessions[session_id]["updated_at"] = datetime.now().isoformat()
        _save(sessions)
        return True
    return False


async def generate_smart_topic(session_id: str) -> Optional[str]:
    """Génère un titre plus intelligent via LLM (appelé après quelques messages)."""
    session = get_session(session_id)
    if not session or len(session["messages"]) < 2:
        return None
    try:
        from core.providers import groq_generate
        first_msgs = " ".join([m["content"] for m in session["messages"][:3]])[:500]
        prompt = (
            f"Résume cette conversation en un titre court (5 mots maximum), "
            f"sans ponctuation finale :\n\n{first_msgs}"
        )
        title = await groq_generate(prompt, "")
        title = title.strip().strip('"').strip("'")[:60]

        sessions = _load()
        if session_id in sessions:
            sessions[session_id]["topic"] = title
            _save(sessions)
        return title
    except Exception:
        return None

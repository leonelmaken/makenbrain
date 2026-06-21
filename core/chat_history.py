"""Persistent chat history with best-effort user memory extraction."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core.memory_router import format_memory, should_store_memory
from core.user_memory_service import create_memory

SESSIONS_FILE = Path("brain_data/chat_sessions.json")


def _load() -> dict[str, Any]:
    """Load persisted chat sessions from disk."""
    if SESSIONS_FILE.exists():
        try:
            return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save(data: dict[str, Any]) -> None:
    """Persist chat sessions to disk."""
    SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSIONS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _quick_topic(first_message: str) -> str:
    """Generate a short heuristic topic without calling an LLM."""
    words = first_message.strip().split()
    topic = " ".join(words[:6])
    return topic[:60] + ("..." if len(words) > 6 else "")


def create_session() -> str:
    """Create a new chat session and return its id."""
    sessions = _load()
    session_id = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()
    sessions[session_id] = {
        "id": session_id,
        "topic": "Nouvelle conversation",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    _save(sessions)
    return session_id


def add_message(
    session_id: str,
    role: str,
    content: str,
    user_id: str | None = None,
) -> None:
    """Add a message to a session, creating the session when needed.

    When an assistant response is saved, the latest user message is routed
    to the user memory layer if a user id is attached to the session. This
    is best effort and never blocks chat history persistence.
    """
    sessions = _load()
    if session_id not in sessions:
        now = datetime.now().isoformat()
        sessions[session_id] = {
            "id": session_id,
            "topic": "Nouvelle conversation",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }

    session = sessions[session_id]
    if user_id:
        session["user_id"] = user_id

    session["messages"].append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
    )
    session["updated_at"] = datetime.now().isoformat()

    if role == "user" and session["topic"] == "Nouvelle conversation":
        session["topic"] = _quick_topic(content)

    _save(sessions)

    if role == "assistant":
        _store_recent_user_memory_best_effort(session)


def get_session(session_id: str) -> Optional[dict[str, Any]]:
    """Return a persisted chat session by id."""
    sessions = _load()
    return sessions.get(session_id)


def list_sessions(limit: int = 30) -> list[dict[str, Any]]:
    """List sessions, newest first."""
    sessions = _load()
    items = sorted(
        sessions.values(),
        key=lambda session: session["updated_at"],
        reverse=True,
    )
    return [
        {
            "id": session["id"],
            "topic": session["topic"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "message_count": len(session["messages"]),
            "preview": session["messages"][0]["content"][:80] if session["messages"] else "",
        }
        for session in items[:limit]
    ]


def delete_session(session_id: str) -> bool:
    """Delete a chat session by id."""
    sessions = _load()
    if session_id in sessions:
        del sessions[session_id]
        _save(sessions)
        return True
    return False


def update_session_topic(session_id: str, new_topic: str) -> bool:
    """Manually update a chat session topic."""
    sessions = _load()
    if session_id in sessions:
        sessions[session_id]["topic"] = new_topic
        sessions[session_id]["updated_at"] = datetime.now().isoformat()
        _save(sessions)
        return True
    return False


async def generate_smart_topic(session_id: str) -> Optional[str]:
    """Generate a smarter short topic via LLM after a few messages."""
    session = get_session(session_id)
    if not session or len(session["messages"]) < 2:
        return None
    try:
        from core.providers import groq_generate

        first_messages = " ".join(
            [message["content"] for message in session["messages"][:3]]
        )[:500]
        prompt = (
            "Resume cette conversation en un titre court (5 mots maximum), "
            f"sans ponctuation finale :\n\n{first_messages}"
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


def _store_recent_user_memory_best_effort(session: dict[str, Any]) -> None:
    """Store the latest user message when it contains stable personal facts."""
    try:
        user_id = session.get("user_id")
        if not user_id:
            return

        user_message = _latest_user_message(session)
        if not user_message or not should_store_memory(user_message):
            return

        memory = format_memory(user_message, str(user_id))
        create_memory(
            user_id=memory["user_id"],
            content=memory["content"],
            source=memory["source"],
            metadata=memory["metadata"],
        )
    except Exception:
        return


def _latest_user_message(session: dict[str, Any]) -> str | None:
    """Return the most recent user message from a chat session."""
    for message in reversed(session.get("messages", [])):
        if message.get("role") == "user":
            return message.get("content")
    return None

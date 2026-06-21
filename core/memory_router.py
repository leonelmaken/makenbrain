"""Heuristics for routing user text into the personalized memory layer."""
from __future__ import annotations

from datetime import datetime
from typing import Any

MEMORY_SOURCE_CHAT = "chat"

_MEMORY_PATTERNS = (
    "je suis ",
    "j'aime ",
    "je n'aime pas ",
    "je prefere ",
    "mon ",
    "ma ",
    "mes ",
    "j'utilise ",
    "je travaille ",
    "je veux ",
    "appelle-moi ",
    "souviens-toi ",
    "retient ",
    "retiens ",
)

_QUESTION_PREFIXES = (
    "qui ",
    "quoi ",
    "quand ",
    "comment ",
    "pourquoi ",
    "combien ",
    "est-ce ",
    "peux-tu ",
    "tu peux ",
)


def extract_user_intent(text: str) -> dict[str, Any]:
    """Analyze a user message with lightweight, deterministic heuristics.

    Args:
        text: Raw user message.

    Returns:
        A normalized intent dictionary usable by the memory service.
    """
    normalized = _normalize_text(text)
    is_question = normalized.endswith("?") or normalized.startswith(_QUESTION_PREFIXES)
    is_memory_candidate = should_store_memory(text)

    if "souviens-toi" in normalized or "retiens" in normalized or "retient" in normalized:
        intent = "explicit_memory"
    elif is_memory_candidate:
        intent = "profile_signal"
    elif is_question:
        intent = "question"
    else:
        intent = "conversation"

    return {
        "intent": intent,
        "is_question": is_question,
        "should_store": is_memory_candidate,
        "confidence": 0.8 if intent == "explicit_memory" else 0.55,
    }


def should_store_memory(text: str) -> bool:
    """Return whether text likely contains stable user-specific information."""
    normalized = _normalize_text(text)
    if len(normalized) < 8:
        return False
    if normalized.startswith(_QUESTION_PREFIXES) and "souviens-toi" not in normalized:
        return False
    return any(pattern in normalized for pattern in _MEMORY_PATTERNS)


def format_memory(text: str, user_id: str) -> dict[str, Any]:
    """Normalize a user message into a memory creation payload.

    Args:
        text: Raw user message to store.
        user_id: Supabase user id owning the memory.

    Returns:
        A JSON-ready dictionary for `user_memory_service.create_memory`.
    """
    intent = extract_user_intent(text)
    return {
        "user_id": user_id,
        "content": text.strip(),
        "source": MEMORY_SOURCE_CHAT,
        "metadata": {
            "intent": intent["intent"],
            "confidence": intent["confidence"],
            "stored_at": datetime.utcnow().isoformat(),
        },
    }


def _normalize_text(text: str) -> str:
    """Return lowercase text with collapsed whitespace."""
    return " ".join(text.strip().lower().split())

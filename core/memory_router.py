"""Routage heuristique des textes utilisateur vers la memoire personnalisee."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

MEMORY_SOURCE_CHAT = "chat"
_MIN_MEMORY_LENGTH = 12
_MAX_RECENCY_DAYS = 90

_MEMORY_PATTERNS = (
    "je suis ",
    "j'aime ",
    "je n'aime pas ",
    "je prefere ",
    "je préfère ",
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

_NOISY_PATTERNS = (
    "bonjour",
    "salut",
    "merci",
    "ok",
    "d'accord",
    "peux-tu",
    "tu peux",
    "explique",
    "resume",
    "résume",
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
    """Analyse une intention utilisateur avec des heuristiques deterministes.

    Cette fonction permet de classifier rapidement un message sans appel LLM.
    Elle est utilisee pour enrichir les metadonnees de memoire.
    Parametres:
        text: Message brut de l'utilisateur.
    Retour:
        Dictionnaire normalise decrivant l'intention detectee.
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
    """Determine si un texte contient une information personnelle durable.

    Cette fonction privilegie la precision pour eviter les faux positifs.
    Les questions, salutations et consignes generiques ne sont pas stockees,
    sauf demande explicite de memorisation.
    Parametres:
        text: Message utilisateur a analyser.
    Retour:
        True si le message peut etre stocke comme memoire utilisateur.
    """
    normalized = _normalize_text(text)
    if len(normalized) < _MIN_MEMORY_LENGTH:
        return False
    explicit_memory = _has_explicit_memory_signal(normalized)
    if normalized.endswith("?") and not explicit_memory:
        return False
    if normalized.startswith(_QUESTION_PREFIXES) and not explicit_memory:
        return False
    if normalized in _NOISY_PATTERNS:
        return False
    if any(normalized.startswith(pattern) for pattern in _NOISY_PATTERNS) and not explicit_memory:
        return False
    if explicit_memory:
        return True
    return any(pattern in normalized for pattern in _MEMORY_PATTERNS)


def format_memory(text: str, user_id: str) -> dict[str, Any]:
    """Normalise un message utilisateur en payload de creation memoire.

    Cette fonction centralise le format attendu par `user_memory_service`.
    Parametres:
        text: Message brut a stocker.
        user_id: Identifiant Supabase du proprietaire de la memoire.
    Retour:
        Dictionnaire compatible avec la creation de memoire utilisateur.
    """
    intent = extract_user_intent(text)
    return {
        "user_id": user_id,
        "content": text.strip(),
        "source": MEMORY_SOURCE_CHAT,
        "metadata": {
            "intent": intent["intent"],
            "confidence": intent["confidence"],
            "stored_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def score_memory_relevance(text: str, memory: dict[str, Any]) -> float:
    """Calcule un score simple de pertinence entre un texte et une memoire.

    Le score combine trois signaux peu couteux: similarite lexicale,
    fraicheur temporelle et pertinence contextuelle issue des metadonnees.
    Parametres:
        text: Texte courant ou contexte de requete.
        memory: Memoire utilisateur normalisee.
    Retour:
        Score compris entre 0.0 et 1.0.
    """
    content = str(memory.get("content", ""))
    lexical_score = _token_similarity(text, content)
    recency_score = _recency_score(memory.get("updated_at") or memory.get("created_at"))
    metadata_score = _metadata_score(memory.get("metadata"))
    score = (lexical_score * 0.6) + (recency_score * 0.25) + (metadata_score * 0.15)
    return max(0.0, min(1.0, score))


def _normalize_text(text: str) -> str:
    """Retourne un texte minuscule avec les espaces normalises."""
    return " ".join(text.strip().lower().split())


def _has_explicit_memory_signal(normalized_text: str) -> bool:
    """Indique si l'utilisateur demande explicitement une memorisation."""
    return (
        "souviens-toi" in normalized_text
        or "retiens" in normalized_text
        or "retient" in normalized_text
        or "memorise" in normalized_text
        or "mémorise" in normalized_text
    )


def _token_similarity(left: str, right: str) -> float:
    """Calcule une similarite Jaccard simple entre deux textes."""
    left_tokens = _meaningful_tokens(left)
    right_tokens = _meaningful_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return intersection / union if union else 0.0


def _meaningful_tokens(text: str) -> set[str]:
    """Extrait les mots utiles pour une comparaison rapide."""
    return {
        token
        for token in _normalize_text(text).replace("'", " ").split()
        if len(token) >= 3
    }


def _recency_score(value: Any) -> float:
    """Transforme une date de memoire en score de fraicheur."""
    if not value:
        return 0.0
    try:
        raw_value = str(value).replace("Z", "+00:00")
        memory_date = datetime.fromisoformat(raw_value)
        if memory_date.tzinfo is None:
            memory_date = memory_date.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - memory_date).days
    except Exception:
        return 0.0
    return max(0.0, 1.0 - (age_days / _MAX_RECENCY_DAYS))


def _metadata_score(metadata: Any) -> float:
    """Valorise les memoires explicitement personnelles ou confirmees."""
    if not isinstance(metadata, dict):
        return 0.0
    intent = str(metadata.get("intent", ""))
    confidence = metadata.get("confidence", 0.0)
    if intent == "explicit_memory":
        return 1.0
    if intent == "profile_signal":
        try:
            return min(0.8, float(confidence))
        except (TypeError, ValueError):
            return 0.5
    return 0.2

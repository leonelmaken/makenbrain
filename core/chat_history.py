"""Historique de chat persistant avec extraction de memoire utilisateur."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core.memory_router import format_memory, score_memory_relevance, should_store_memory
from core.user_memory_service import create_memory, get_user_memories

SESSIONS_FILE = Path("brain_data/chat_sessions.json")


def _load() -> dict[str, Any]:
    """Charge les sessions de conversation depuis le disque.

    Cette fonction permet de centraliser la lecture de l'historique local.
    Parametres:
        Aucun.
    Retour:
        Dictionnaire des sessions connues, ou dictionnaire vide en cas
        d'erreur de lecture.
    """
    if SESSIONS_FILE.exists():
        try:
            return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save(data: dict[str, Any]) -> None:
    """Persiste les sessions de conversation sur le disque.

    Parametres:
        data: Sessions a serialiser dans le fichier local.
    Retour:
        Aucun.
    """
    SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSIONS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _quick_topic(first_message: str) -> str:
    """Genere un titre court sans appel LLM.

    Parametres:
        first_message: Premier message utilisateur de la session.
    Retour:
        Titre court exploitable dans la liste des conversations.
    """
    words = first_message.strip().split()
    topic = " ".join(words[:6])
    return topic[:60] + ("..." if len(words) > 6 else "")


def create_session(user_id: str = "") -> str:
    """Cree une nouvelle session de chat.

    Parametres:
        user_id: Propriétaire de la session. Optionnel pour la compatibilité
                 avec les appels existants, mais requis pour l'isolation.
    Retour:
        Identifiant court de la session creee.
    """
    sessions = _load()
    session_id = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()
    entry: dict = {
        "id": session_id,
        "topic": "Nouvelle conversation",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    if user_id:
        entry["user_id"] = user_id
    sessions[session_id] = entry
    _save(sessions)
    return session_id


def add_message(
    session_id: str,
    role: str,
    content: str,
    user_id: str,
    extras: dict[str, Any] | None = None,
) -> None:
    """Ajoute un message utilisateur-scope a une session.

    Cette fonction garantit que chaque message persiste contient le
    `user_id`. Elle maintient aussi la coherence session-utilisateur:
    une session deja associee a un utilisateur ne peut pas etre enrichie
    avec les messages d'un autre utilisateur.
    Parametres:
        session_id: Identifiant de la session de conversation.
        role: Role du message, par exemple `user` ou `assistant`.
        content: Contenu textuel a persister.
        user_id: Identifiant Supabase du proprietaire de la session.
    Retour:
        Aucun.
    """
    sessions = _load()
    if session_id not in sessions:
        now = datetime.now().isoformat()
        sessions[session_id] = {
            "id": session_id,
            "topic": "Nouvelle conversation",
            "created_at": now,
            "updated_at": now,
            "user_id": user_id,
            "messages": [],
        }

    session = sessions[session_id]
    if not _ensure_session_user(session, user_id):
        return

    message: dict[str, Any] = {
        "role": role,
        "content": content,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
    }
    # extras : données jointes au message (ex. web_sources/web_images du
    # grounding) — persistées pour être ré-affichées au rechargement.
    if extras:
        message.update(extras)
    session["messages"].append(message)
    session["updated_at"] = datetime.now().isoformat()

    if role == "user" and session["topic"] == "Nouvelle conversation":
        session["topic"] = _quick_topic(content)

    _save(sessions)

    if role == "assistant":
        _store_recent_user_memory_best_effort(session)


def get_session(session_id: str) -> Optional[dict[str, Any]]:
    """Retourne une session de conversation par identifiant.

    Parametres:
        session_id: Identifiant de la session recherchee.
    Retour:
        Session trouvee ou None.
    """
    sessions = _load()
    return sessions.get(session_id)


def list_sessions(user_id: str | None = None, limit: int = 30) -> list[dict[str, Any]]:
    """Liste les sessions les plus recentes.

    Parametres:
        user_id: Si fourni, retourne uniquement les sessions de cet utilisateur.
                 Si None, retourne toutes les sessions (usage interne/admin).
        limit: Nombre maximal de sessions retournees.
    Retour:
        Liste compacte des sessions pour affichage ou navigation.
    """
    sessions = _load()
    candidates = sessions.values()
    if user_id is not None:
        candidates = [s for s in candidates if s.get("user_id") == user_id]
    items = sorted(
        candidates,
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


def get_session_for_user(session_id: str, user_id: str) -> Optional[dict[str, Any]]:
    """Retourne une session uniquement si elle appartient à user_id.

    Parametres:
        session_id: Identifiant de la session.
        user_id: Propriétaire attendu.
    Retour:
        Session si la propriété est vérifiée, None sinon.
    """
    session = get_session(session_id)
    if session is None:
        return None
    owner = session.get("user_id")
    if owner and str(owner) != str(user_id):
        return None  # appartient à un autre utilisateur
    return session


def delete_session(session_id: str) -> bool:
    """Supprime une session de conversation.

    Parametres:
        session_id: Identifiant de la session a supprimer.
    Retour:
        True si une session a ete supprimee, False sinon.
    """
    sessions = _load()
    if session_id in sessions:
        del sessions[session_id]
        _save(sessions)
        return True
    return False


def update_session_topic(session_id: str, new_topic: str) -> bool:
    """Met a jour manuellement le titre d'une session.

    Parametres:
        session_id: Identifiant de la session.
        new_topic: Nouveau titre a enregistrer.
    Retour:
        True si la session existe et a ete modifiee.
    """
    sessions = _load()
    if session_id in sessions:
        sessions[session_id]["topic"] = new_topic
        sessions[session_id]["updated_at"] = datetime.now().isoformat()
        _save(sessions)
        return True
    return False


async def generate_smart_topic(session_id: str) -> Optional[str]:
    """Genere un titre court via LLM apres plusieurs messages.

    Parametres:
        session_id: Identifiant de la session a analyser.
    Retour:
        Titre genere, ou None si la generation echoue.
    """
    session = get_session(session_id)
    if not session or len(session["messages"]) < 2:
        return None
    # Un titre intelligent n'est genere qu'une seule fois par session :
    # le titre reste stable et on evite un appel LLM a chaque message.
    if session.get("smart_topic_done"):
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
        if not title:
            return None

        sessions = _load()
        if session_id in sessions:
            sessions[session_id]["topic"] = title
            sessions[session_id]["smart_topic_done"] = True
            _save(sessions)
        return title
    except Exception:
        return None


def _store_recent_user_memory_best_effort(session: dict[str, Any]) -> None:
    """Stocke le dernier message utilisateur si une memoire utile est detectee.

    Cette operation est volontairement best effort: aucune erreur Supabase,
    validation ou reseau ne doit interrompre le flux de chat.
    Parametres:
        session: Session locale contenant les messages et le proprietaire.
    Retour:
        Aucun.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return

        user_message = _latest_user_message(session)
        if not user_message or not should_store_memory(user_message):
            return
        if _has_similar_memory(str(user_id), user_message):
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
    """Retourne le dernier message utilisateur d'une session.

    Parametres:
        session: Session locale a parcourir.
    Retour:
        Contenu du dernier message utilisateur, ou None.
    """
    for message in reversed(session.get("messages", [])):
        if message.get("role") == "user":
            return message.get("content")
    return None


def _ensure_session_user(session: dict[str, Any], user_id: str) -> bool:
    """Valide et fixe le proprietaire d'une session.

    Cette fonction evite qu'une session existante soit reutilisee par un
    autre utilisateur. En cas de conflit, l'ecriture est ignoree pour ne
    jamais faire echouer le chat principal.
    Parametres:
        session: Session locale cible.
        user_id: Utilisateur authentifie courant.
    Retour:
        True si l'ecriture peut continuer, False en cas de conflit.
    """
    existing_user_id = session.get("user_id")
    if existing_user_id and str(existing_user_id) != str(user_id):
        return False
    session["user_id"] = user_id
    return True


def _has_similar_memory(user_id: str, content: str) -> bool:
    """Detecte les doublons evidents avant insertion memoire.

    Parametres:
        user_id: Proprietaire des memoires a comparer.
        content: Nouveau contenu candidat.
    Retour:
        True si une memoire similaire existe deja.
    """
    try:
        normalized_content = _normalize_for_duplicate(content)
        for memory in get_user_memories(user_id):
            existing = _normalize_for_duplicate(str(memory.get("content", "")))
            if not existing:
                continue
            if normalized_content in existing or existing in normalized_content:
                return True
            if score_memory_relevance(content, memory) >= 0.82:
                return True
    except Exception:
        return False
    return False


def _normalize_for_duplicate(text: str) -> str:
    """Normalise un texte pour une comparaison de doublons rapide."""
    return " ".join(text.strip().lower().split())

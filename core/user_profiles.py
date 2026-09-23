"""Profils par utilisateur : domaine, profession, adaptation — Phase 10.

MakenBrain ne répond pas pareil à un médecin, un enseignant ou un ingénieur.
Ce module stocke le profil métier de CHAQUE utilisateur et sait le déduire
automatiquement de ses mémoires et conversations.

Stockage local-first (brain_data/user_profiles.json) : cohérent avec les
plans/quotas — le profil fonctionne même quand Supabase est injoignable.
Ne pas confondre avec core/user_profile.py (singulier) : le profil local
historique du SuperAdmin (bio, objectifs, valeurs).

Structure du fichier :
    { "<user_id>": {"domain": "software_engineering",
                    "profession": "Ingénieur full-stack",
                    "expertise_level": "expert",
                    "source": "manual" | "auto",
                    "updated_at": iso8601,
                    "last_detect_attempt": iso8601} }
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("makenbrain.user_profiles")

USER_PROFILES_FILE = Path("brain_data/user_profiles.json")

# Domaines reconnus — utilisés par la détection ET les futures équipes
# d'agents par métier ("crée-moi mon équipe"). Extensible sans migration.
KNOWN_DOMAINS = (
    "software_engineering", "data_science", "medicine", "law", "education",
    "business", "finance", "marketing", "design", "science", "agriculture",
    "arts", "student", "other",
)

_lock = threading.Lock()


def _load() -> dict[str, Any]:
    if USER_PROFILES_FILE.exists():
        try:
            return json.loads(USER_PROFILES_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save(data: dict[str, Any]) -> None:
    USER_PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
    USER_PROFILES_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_profile(user_id: str) -> dict[str, Any]:
    """Retourne le profil de l'utilisateur (dict vide si inconnu)."""
    return dict(_load().get(str(user_id)) or {})


def set_profile(user_id: str, **fields: Any) -> dict[str, Any]:
    """Crée ou met à jour le profil. Seuls les champs connus sont acceptés."""
    allowed = {
        k: v for k, v in fields.items()
        if k in ("domain", "profession", "expertise_level", "source", "last_detect_attempt")
        and v is not None
    }
    with _lock:
        data = _load()
        profile = data.setdefault(str(user_id), {})
        profile.update(allowed)
        profile["updated_at"] = datetime.now().isoformat()
        _save(data)
        return dict(profile)


def _detection_recently_attempted(profile: dict[str, Any], hours: int = 24) -> bool:
    """Anti-spam : au plus une tentative de détection auto par période."""
    last = profile.get("last_detect_attempt")
    if not last:
        return False
    try:
        return datetime.fromisoformat(last) > datetime.now() - timedelta(hours=hours)
    except ValueError:
        return False


async def detect_domain(user_id: str, force: bool = False) -> dict[str, Any]:
    """Déduit domaine + profession depuis les mémoires et conversations.

    Best-effort : toute erreur retourne le profil actuel inchangé.
    Un profil défini MANUELLEMENT n'est jamais écrasé par la détection auto.

    Args:
        user_id : Utilisateur concerné.
        force   : Ignore l'anti-spam 24 h (utilisé par POST /profile/detect).

    Returns:
        Le profil (mis à jour ou non).
    """
    profile = get_profile(user_id)
    if profile.get("source") == "manual" and not force:
        return profile
    if not force and _detection_recently_attempted(profile):
        return profile

    set_profile(user_id, last_detect_attempt=datetime.now().isoformat())

    # ── Matière première : mémoires + derniers sujets de conversation ────────
    evidence: list[str] = []
    try:
        from core.user_memory_service import get_user_memories
        for m in get_user_memories(user_id)[:20]:
            content = str(m.get("content", "")).strip()
            if content:
                evidence.append(f"- {content[:200]}")
    except Exception:
        pass
    try:
        from core.chat_history import list_sessions
        for s in list_sessions(user_id=str(user_id), limit=10):
            topic = str(s.get("topic", "")).strip()
            if topic and topic != "Nouvelle conversation":
                evidence.append(f"- Sujet de conversation : {topic[:120]}")
    except Exception:
        pass

    if len(evidence) < 3:
        logger.info("Détection de domaine : pas assez de données pour %s.", user_id)
        return get_profile(user_id)

    prompt = (
        "Voici ce que l'on sait d'un utilisateur (mémoires et sujets de conversation) :\n\n"
        + "\n".join(evidence[:25])
        + "\n\nDéduis son domaine professionnel et sa profession probable.\n"
        f"Domaines valides : {', '.join(KNOWN_DOMAINS)}.\n"
        'Réponds UNIQUEMENT en JSON : {"domain": "...", "profession": "...", '
        '"expertise_level": "beginner|intermediate|expert", "confidence": 0.0}\n'
        "Si les données sont insuffisantes ou ambiguës, mets confidence sous 0.5."
    )

    try:
        from core.providers import groq_generate, GROQ_FAST

        raw = await asyncio.wait_for(
            groq_generate(
                prompt, "",
                model=GROQ_FAST,
                system_prompt="Tu es un classifieur de profils professionnels. Réponds uniquement le JSON demandé.",
            ),
            timeout=10,
        )
        import re
        match = re.search(r"\{.*\}", str(raw), re.DOTALL)
        if not match:
            return get_profile(user_id)
        parsed = json.loads(match.group())
        domain = str(parsed.get("domain", "")).strip()
        confidence = float(parsed.get("confidence", 0))
        if domain not in KNOWN_DOMAINS or confidence < 0.5:
            logger.info("Détection de domaine non concluante pour %s (conf=%.2f).", user_id, confidence)
            return get_profile(user_id)
        return set_profile(
            user_id,
            domain=domain,
            profession=str(parsed.get("profession", "")).strip()[:120] or None,
            expertise_level=str(parsed.get("expertise_level", "")).strip() or None,
            source="auto",
        )
    except Exception as exc:
        logger.warning("Détection de domaine échouée pour %s : %s: %s",
                       user_id, type(exc).__name__, exc)
        return get_profile(user_id)


async def maybe_autodetect(user_id: str) -> None:
    """Hook d'arrière-plan (appelé après un chat) : détecte si profil vide.

    Ne fait rien si un domaine existe déjà ou si une tentative a eu lieu
    dans les dernières 24 h. Jamais bloquant, jamais d'exception.
    """
    try:
        profile = get_profile(user_id)
        if profile.get("domain"):
            return
        await detect_domain(user_id, force=False)
    except Exception:
        pass

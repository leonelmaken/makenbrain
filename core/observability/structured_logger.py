"""Structured logging pour MakenBrain — Phase 4.

Architecture :
    BrainLogger          : wrapper autour de logging.Logger qui injecte
                           automatiquement les champs de contexte dans chaque
                           entrée de log.
    RequestContext       : ContextVar propagé par RequestContextMiddleware
                           (défini dans core/middleware.py) pour transporter
                           request_id, user_id, session_id sans les passer
                           manuellement à chaque appel.
    BrainJSONFormatter   : Formatter stdlib qui sérialise chaque LogRecord en
                           JSON sur une seule ligne — compatible avec les
                           agrégateurs de logs (Loki, DataDog, etc.) et avec
                           le futur Dashboard Super Admin.

Utilisation :
    from core.observability import get_logger

    logger = get_logger("makenbrain.reasoning.synthesizer")

    # Log simple
    logger.info("Synthèse terminée", duration_ms=42.5, agent="synthesizer")

    # Log avec contexte de requête (automatiquement injecté par le middleware)
    logger.error("LLM timeout", provider="groq", success=False,
                 error="Connection refused")

Compatibilité :
    Les modules existants qui utilisent logging.getLogger() directement
    continuent de fonctionner sans modification. BrainLogger améliore
    l'expérience pour les nouveaux modules et les modules Phase 4+.
"""
from __future__ import annotations

import json
import logging
import threading
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from core.observability.log_entry import LogEntry

# ── Contexte de requête (propagé par le middleware) ───────────────────────────

_request_context: ContextVar[dict[str, str | None]] = ContextVar(
    "_request_context",
    default={
        "request_id": None,
        "session_id": None,
        "user_id"   : None,
        "endpoint"  : None,
    },
)


def set_request_context(
    *,
    request_id : str | None = None,
    session_id : str | None = None,
    user_id    : str | None = None,
    endpoint   : str | None = None,
) -> None:
    """Positionne le contexte de la requête courante dans cette tâche asyncio.

    Appelé par RequestContextMiddleware au début de chaque requête HTTP.
    Le contexte est automatiquement isolé par tâche asyncio — pas de risque
    de contamination entre requêtes concurrentes.

    Args:
        request_id : ID unique de la requête (UUID généré par le middleware).
        session_id : ID de la session de conversation.
        user_id    : ID Supabase de l'utilisateur authentifié.
        endpoint   : Chemin HTTP de la requête (ex. "/reasoning/analyze").
    """
    _request_context.set({
        "request_id": request_id,
        "session_id": session_id,
        "user_id"   : user_id,
        "endpoint"  : endpoint,
    })


def get_request_context() -> dict[str, str | None]:
    """Retourne le contexte de la requête courante.

    Returns:
        Dict avec les clés request_id, session_id, user_id, endpoint.
        Les valeurs sont None si le contexte n'a pas été positionné.
    """
    return _request_context.get()


# ── JSON Formatter ─────────────────────────────────────────────────────────────

class BrainJSONFormatter(logging.Formatter):
    """Formatter qui sérialise chaque LogRecord en une ligne JSON.

    Injecte automatiquement les champs du contexte de requête courant.
    Les champs None sont omis pour garder les payloads compacts.

    Format de sortie :
        {"timestamp": "...", "level": "INFO", "logger": "...", "message": "...",
         "request_id": "...", "agent": "...", ...}
    """

    def format(self, record: logging.LogRecord) -> str:
        ctx = get_request_context()
        entry = LogEntry(
            timestamp   = datetime.fromtimestamp(record.created).isoformat(),
            level       = record.levelname,
            logger_name = record.name,
            message     = record.getMessage(),
            request_id  = ctx.get("request_id"),
            session_id  = getattr(record, "session_id", None) or ctx.get("session_id"),
            user_id     = getattr(record, "user_id", None)    or ctx.get("user_id"),
            endpoint    = getattr(record, "endpoint", None)   or ctx.get("endpoint"),
            agent       = getattr(record, "agent", None),
            provider    = getattr(record, "provider", None),
            model       = getattr(record, "model", None),
            duration_ms = getattr(record, "duration_ms", None),
            success     = getattr(record, "success", None),
            error       = getattr(record, "error", None),
            extra       = getattr(record, "brain_extra", {}),
        )
        if record.exc_info:
            entry.error = self.formatException(record.exc_info)
        return json.dumps(entry.to_dict(), ensure_ascii=False)


# ── BrainLogger ───────────────────────────────────────────────────────────────

class BrainLogger:
    """Wrapper autour de logging.Logger qui injecte les champs de contexte.

    Fournit la même interface que logging.Logger (debug/info/warning/error/
    critical) mais chaque méthode accepte des kwargs additionnels qui
    deviennent des champs structurés dans le LogEntry.

    Usage :
        logger = BrainLogger("makenbrain.reasoning.synthesizer")
        logger.info("Synthèse", agent="synthesizer", duration_ms=42.5)
        logger.error("Échec LLM", provider="groq", success=False,
                     error="timeout")
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    @property
    def name(self) -> str:
        return self._logger.name

    def _log(
        self,
        level : int,
        msg   : str,
        *args : Any,
        **kwargs: Any,
    ) -> None:
        """Appel de log générique avec injection des champs structurés.

        Les kwargs connus sont mappés vers des attributs du LogRecord.
        Tous les autres kwargs sont regroupés dans brain_extra.
        """
        known_fields = {
            "agent", "provider", "model", "duration_ms",
            "success", "error", "session_id", "user_id", "endpoint",
        }
        extra: dict[str, Any] = {}
        brain_extra: dict[str, Any] = {}

        for k, v in kwargs.items():
            if k in known_fields:
                extra[k] = v
            else:
                brain_extra[k] = v

        if brain_extra:
            extra["brain_extra"] = brain_extra

        self._logger.log(level, msg, *args, extra=extra, stacklevel=2)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("exc_info", True)
        self._log(logging.ERROR, msg, *args, **kwargs)

    def is_enabled_for(self, level: int) -> bool:
        return self._logger.isEnabledFor(level)


# ── Factory ───────────────────────────────────────────────────────────────────

def get_logger(name: str) -> BrainLogger:
    """Retourne un BrainLogger pour le module donné.

    Les noms suivent la convention Python existante du projet :
        "makenbrain.reasoning.synthesizer"
        "makenbrain.providers.groq"
        "makenbrain.health"

    Args:
        name : Nom du logger (convention pointée recommandée).

    Returns:
        BrainLogger wrappant le logging.Logger standard du même nom.
    """
    return BrainLogger(name)


def configure_json_logging(level: int = logging.INFO) -> None:
    """Configure le root logger pour émettre du JSON structuré.

    À appeler une seule fois au démarrage de l'application (dans main.py
    ou lifespan). Après cet appel, tous les loggers du projet — qu'ils
    utilisent BrainLogger ou logging.getLogger() directement — produisent
    du JSON sur stdout.

    Args:
        level : Niveau de log minimum (défaut : INFO).

    Note :
        En développement, on peut préférer le format texte par défaut.
        Appeler cette fonction uniquement si JSON_LOGS=true dans .env.
    """
    formatter = BrainJSONFormatter()
    handler   = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

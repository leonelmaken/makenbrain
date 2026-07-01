"""DTO structuré pour chaque entrée de log MakenBrain — Phase 4.

Centralise les champs d'un événement de log afin que le futur Dashboard
Super Admin puisse les consommer sans dépendre d'un format ad-hoc.

Règles :
- Zéro logique métier dans ce module.
- Tous les champs sont optionnels sauf timestamp, level et message.
- Le champ extra accepte tout dict arbitraire pour les données spécifiques à
  un agent ou un endpoint qui ne méritent pas un champ de premier niveau.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LogEntry:
    """Entrée de log structurée.

    Produite par BrainLogger et stockée / transmise pour le Dashboard.

    Champs de contexte communs :
        request_id : ID unique de la requête HTTP (propagé par RequestContextMiddleware).
        session_id : ID de la session de conversation.
        user_id    : ID Supabase de l'utilisateur.
        endpoint   : Chemin HTTP (ex. "/reasoning/analyze").
        agent      : Nom de l'agent concerné (ex. "synthesizer", "decision_engine").
        provider   : Provider LLM utilisé (ex. "ollama", "groq", "claude").
        model      : Modèle LLM précis (ex. "llama3.2:3b", "llama-3.3-70b-versatile").
        duration_ms: Durée de l'opération en millisecondes.
        success    : Résultat de l'opération (None si non applicable).
        error      : Message d'erreur textuel en cas d'échec.
        extra      : Données additionnelles libres (tokens, scores, etc.)
    """

    timestamp  : str
    level      : str   # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    logger_name: str
    message    : str

    # ── Contexte de requête (propagé par le middleware) ───────────────────────
    request_id : str | None = None
    session_id : str | None = None
    user_id    : str | None = None
    endpoint   : str | None = None

    # ── Contexte d'exécution ─────────────────────────────────────────────────
    agent      : str | None = None
    provider   : str | None = None
    model      : str | None = None
    duration_ms: float | None = None
    success    : bool | None = None
    error      : str | None = None

    extra      : dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Sérialise l'entrée en dict JSON-compatible.

        Les champs None sont exclus pour garder les payloads compacts.
        """
        base = {
            "timestamp"  : self.timestamp,
            "level"      : self.level,
            "logger"     : self.logger_name,
            "message"    : self.message,
        }
        optional = {
            "request_id" : self.request_id,
            "session_id" : self.session_id,
            "user_id"    : self.user_id,
            "endpoint"   : self.endpoint,
            "agent"      : self.agent,
            "provider"   : self.provider,
            "model"      : self.model,
            "duration_ms": self.duration_ms,
            "success"    : self.success,
            "error"      : self.error,
        }
        result = {**base, **{k: v for k, v in optional.items() if v is not None}}
        if self.extra:
            result["extra"] = self.extra
        return result

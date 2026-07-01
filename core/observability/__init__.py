"""Package observabilité MakenBrain — Phase 4.

Exports publics :
    get_logger             : Factory BrainLogger (à utiliser dans tous les modules).
    set_request_context    : Positionne le contexte de requête (appelé par le middleware).
    get_request_context    : Lit le contexte courant.
    configure_json_logging : Active le format JSON sur stdout (à appeler au démarrage).
    MetricsCollector       : Collecteur in-memory de métriques.
    get_metrics            : Retourne le singleton global MetricsCollector.
    LogEntry               : DTO d'une entrée de log structurée.
"""
from core.observability.log_entry import LogEntry
from core.observability.structured_logger import (
    BrainLogger,
    configure_json_logging,
    get_logger,
    get_request_context,
    set_request_context,
)
from core.observability.metrics import MetricsCollector, get_metrics

__all__ = [
    "LogEntry",
    "BrainLogger",
    "get_logger",
    "set_request_context",
    "get_request_context",
    "configure_json_logging",
    "MetricsCollector",
    "get_metrics",
]

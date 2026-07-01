"""Package mémoire épisodique MakenBrain — Phase 6.

Enregistre l'historique chronologique des actions, décisions, apprentissages
et événements projets de MakenBrain. Complément à la mémoire vectorielle
(sémantique) et à la mémoire graphe (relationnelle).

Exports :
    EpisodicEntry      : DTO d'une entrée épisodique.
    EpisodicType       : Enum des types d'événements.
    EpisodicMemoryStore: Stockage JSONL append-only.
    get_episodic_store : Singleton du store global.
"""
from core.episodic.models import EpisodicEntry, EpisodicType
from core.episodic.store import EpisodicMemoryStore, get_episodic_store

__all__ = [
    "EpisodicEntry",
    "EpisodicType",
    "EpisodicMemoryStore",
    "get_episodic_store",
]

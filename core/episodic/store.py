"""EpisodicMemoryStore — stockage de la mémoire épisodique — Phase 6.

Implémentation : fichier JSONL append-only.
    - Chaque ligne est un JSON valide (une EpisodicEntry sérialisée).
    - Append-only : aucune ligne n'est jamais modifiée ni supprimée.
    - Thread-safe via threading.Lock.

Chemin par défaut : brain_data/episodic/entries.jsonl

Phase 7+ : migration vers Supabase (table `episodic_memories`) pour
permettre les requêtes SQL, la consolidation et le Dashboard Super Admin.
L'interface publique (record/get_recent/get_by_type/get_by_agent) restera
identique — seul le backend changera.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterator

from core.episodic.models import EpisodicEntry, EpisodicType

_DEFAULT_PATH = Path("brain_data/episodic/entries.jsonl")


class EpisodicMemoryStore:
    """Stockage chronologique des événements épisodiques.

    Usage :
        store = get_episodic_store()
        store.record(EpisodicEntry(type=EpisodicType.ACTION, agent_name="memory_agent",
                                   content="Écriture d'un souvenir"))
        recent = store.get_recent(10)
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DEFAULT_PATH
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: EpisodicEntry) -> None:
        """Enregistre une entrée épisodique (append-only, thread-safe).

        Args:
            entry : L'événement à enregistrer.
        """
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def get_recent(self, n: int = 20) -> list[EpisodicEntry]:
        """Retourne les n entrées les plus récentes (ordre chronologique inverse).

        Args:
            n : Nombre maximum d'entrées à retourner.
        """
        entries = list(self._iter_all())
        return list(reversed(entries))[:n]

    def get_by_type(self, type: EpisodicType, limit: int = 50) -> list[EpisodicEntry]:
        """Retourne les entrées d'un type donné (les plus récentes en premier).

        Args:
            type  : Type d'événement à filtrer.
            limit : Nombre maximum de résultats.
        """
        matching = [e for e in self._iter_all() if e.type == type]
        return list(reversed(matching))[:limit]

    def get_by_agent(self, agent_name: str, limit: int = 50) -> list[EpisodicEntry]:
        """Retourne les entrées produites par un agent donné.

        Args:
            agent_name : Nom de l'agent.
            limit      : Nombre maximum de résultats.
        """
        matching = [e for e in self._iter_all() if e.agent_name == agent_name]
        return list(reversed(matching))[:limit]

    def get_by_user(self, user_id: str, limit: int = 50) -> list[EpisodicEntry]:
        """Retourne les entrées d'un utilisateur donné."""
        matching = [e for e in self._iter_all() if e.user_id == user_id]
        return list(reversed(matching))[:limit]

    def count(self) -> int:
        """Retourne le nombre total d'entrées enregistrées."""
        return sum(1 for _ in self._iter_all())

    def _iter_all(self) -> Iterator[EpisodicEntry]:
        """Itère sur toutes les entrées du fichier JSONL (ordre chronologique)."""
        if not self._path.exists():
            return
        with self._lock:
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield EpisodicEntry.from_dict(json.loads(line))
                    except Exception:
                        continue  # ligne corrompue ignorée silencieusement


# ── Singleton global ───────────────────────────────────────────────────────────

_global_store: EpisodicMemoryStore | None = None
_store_lock = threading.Lock()


def get_episodic_store(path: Path | None = None) -> EpisodicMemoryStore:
    """Retourne le singleton global EpisodicMemoryStore.

    Thread-safe. Crée l'instance au premier appel.

    Args:
        path : Chemin du fichier JSONL. Utilisé uniquement lors du
               premier appel (ignoré ensuite). Pour les tests, créer
               une instance directement : EpisodicMemoryStore(path=...).
    """
    global _global_store
    if _global_store is None:
        with _store_lock:
            if _global_store is None:
                _global_store = EpisodicMemoryStore(path)
    return _global_store

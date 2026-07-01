"""MemoryAgent — point d'accès unique à la mémoire vectorielle — Phase 6.

Tous les agents qui ont besoin de lire ou d'écrire en mémoire doivent
passer par cet agent via l'Orchestrator. Aucun agent ne doit importer
directement core.memory.

Actions supportées (via task.context["action"]) :
    "search"  : recherche sémantique (défaut si action absente).
    "write"   : ajoute un souvenir.
    "delete"  : supprime un souvenir par ID.
    "stats"   : retourne les statistiques de la mémoire.

Propagation du contexte :
    Après un "search" réussi, les résultats sont stockés dans
    ctx.shared["memory_results"] pour que les agents suivants
    (ReasoningAgent, etc.) puissent les consommer sans re-requêter.

Autonomie : SANDBOXED_EXECUTE — l'agent écrit en mémoire de façon
contrôlée (toutes les écritures sont tracées dans la mémoire épisodique).
"""
from __future__ import annotations

import json
import time
from typing import Any

from core.agents.base import AgentAutonomy, BaseAgent
from core.agents.models import AgentResult, AgentTask, ExecutionContext
from core.memory import add_memory, delete_memory, get_memory_stats, search_memory


class MemoryAgent(BaseAgent):
    """Agent de gestion de la mémoire vectorielle ChromaDB.

    Point d'accès unique à la mémoire pour tous les autres agents.

    task.context attendu selon l'action :
        action="search"  : task.input = query, context.get("n_results", 5)
        action="write"   : task.input = contenu, context.get("metadata", {})
        action="delete"  : context["memory_id"] = ID à supprimer
        action="stats"   : aucun paramètre supplémentaire requis
    """

    name                : str           = "memory_agent"
    description         : str           = (
        "Agent de gestion de la mémoire vectorielle. Point d'accès unique "
        "à ChromaDB pour la lecture, l'écriture et la suppression de souvenirs."
    )
    capabilities        : list[str]     = [
        "memory", "read_memory", "write_memory", "search_memory", "delete_memory",
    ]
    autonomy            : AgentAutonomy = AgentAutonomy.SANDBOXED_EXECUTE
    version             : str           = "1.0.0"

    cost_per_call       : float         = 0.1
    confidence_threshold: float         = 0.90

    async def run(self, task: AgentTask, ctx: ExecutionContext) -> AgentResult:
        """Exécute l'opération mémoire demandée.

        L'action est lue depuis task.context["action"].
        Par défaut : "search" avec task.input comme requête.
        """
        t0     = time.monotonic()
        action = task.context.get("action", "search")

        try:
            if action == "search":
                return await self._search(task, ctx, t0)
            elif action == "write":
                return await self._write(task, ctx, t0)
            elif action == "delete":
                return await self._delete(task, ctx, t0)
            elif action == "stats":
                return await self._stats(task, t0)
            else:
                return self._timed_result(
                    task, t0,
                    success = False,
                    error   = f"Action inconnue : '{action}'. Valeurs valides : search, write, delete, stats.",
                )
        except Exception as exc:
            return self._timed_result(
                task, t0,
                success = False,
                error   = str(exc),
            )

    # ── Actions ───────────────────────────────────────────────────────────────

    async def _search(
        self,
        task: AgentTask,
        ctx : ExecutionContext,
        t0  : float,
    ) -> AgentResult:
        n_results = int(task.context.get("n_results", 5))
        results   = await search_memory(task.input, n_results)

        ctx.shared["memory_results"] = results

        return self._timed_result(
            task, t0,
            success  = True,
            output   = json.dumps(results, ensure_ascii=False),
            metadata = {"n_results": len(results), "query": task.input},
        )

    async def _write(
        self,
        task: AgentTask,
        ctx : ExecutionContext,
        t0  : float,
    ) -> AgentResult:
        metadata   = task.context.get("metadata", {})
        memory_id  = await add_memory(task.input, metadata)

        self._record_episodic(task, "write", {"memory_id": memory_id})

        return self._timed_result(
            task, t0,
            success  = True,
            output   = memory_id,
            metadata = {"memory_id": memory_id},
        )

    async def _delete(
        self,
        task: AgentTask,
        ctx : ExecutionContext,
        t0  : float,
    ) -> AgentResult:
        memory_id = task.context.get("memory_id")
        if not memory_id:
            return self._timed_result(
                task, t0,
                success = False,
                error   = "memory_id manquant dans task.context pour l'action 'delete'.",
            )

        ok = await delete_memory(memory_id)
        self._record_episodic(task, "delete", {"memory_id": memory_id, "success": ok})

        return self._timed_result(
            task, t0,
            success  = ok,
            output   = memory_id if ok else None,
            error    = None if ok else f"Impossible de supprimer le souvenir '{memory_id}'.",
            metadata = {"memory_id": memory_id},
        )

    async def _stats(self, task: AgentTask, t0: float) -> AgentResult:
        stats = await get_memory_stats()
        return self._timed_result(
            task, t0,
            success = True,
            output  = json.dumps(stats, ensure_ascii=False),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _record_episodic(
        self,
        task   : AgentTask,
        action : str,
        details: dict[str, Any],
    ) -> None:
        """Enregistre l'action dans la mémoire épisodique (best-effort)."""
        try:
            from core.episodic.store import get_episodic_store
            from core.episodic.models import EpisodicEntry, EpisodicType

            get_episodic_store().record(EpisodicEntry(
                agent_name = self.name,
                task_id    = task.task_id,
                user_id    = task.user_id,
                type       = EpisodicType.ACTION,
                content    = f"Memory {action}: {task.input[:100]}",
                metadata   = details,
            ))
        except Exception:
            pass

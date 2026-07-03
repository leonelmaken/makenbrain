"""BrainOrchestrator — cœur du système multi-agents — Phase 5.

Responsabilités :
    1. Recevoir une AgentTask depuis le router HTTP.
    2. Déléguer la sélection des agents au TaskRouter.
    3. Orchestrer l'exécution séquentielle des agents sélectionnés.
    4. Accumuler les résultats dans un ExecutionContext partagé.
    5. Produire un ExecutionReport final.
    6. Enregistrer les métriques et les logs structurés.

Philosophie :
    Aucun agent ne doit jamais connaître l'existence d'un autre.
    Tout partage de données inter-agents passe par ExecutionContext.shared.
    L'Orchestrator est le seul coordinateur — il ne délègue jamais
    cette responsabilité à un agent.

Phase 5 vs Phase 6+ :
    Phase 5 : exécution séquentielle uniquement (RoutingPlan.sequential).
    Phase 6+ : l'Orchestrator utilisera RoutingPlan.parallel_groups
               avec asyncio.gather() pour l'exécution parallèle.
               L'interface publique (execute()) ne changera pas.
"""
from __future__ import annotations

import asyncio
import time
import uuid

from core.agents.models import (
    AgentResult,
    AgentTask,
    ExecutionContext,
    ExecutionReport,
)
from core.agents.registry import AgentRegistry, get_registry
from core.agents.task_router import TaskRouter
from core.observability import get_logger, get_metrics

_logger = get_logger("makenbrain.agents.orchestrator")


class BrainOrchestrator:
    """Orchestrateur central du système multi-agents MakenBrain.

    Usage :
        orchestrator = BrainOrchestrator()
        report = await orchestrator.execute(task)

    Args:
        registry : AgentRegistry à utiliser. Si None, utilise le singleton global.
    """

    def __init__(self, registry: AgentRegistry | None = None) -> None:
        self._registry = registry if registry is not None else get_registry()
        self._router   = TaskRouter(self._registry)

    async def execute(self, task: AgentTask, request_id: str | None = None) -> ExecutionReport:
        """Exécute une tâche en orchestrant les agents appropriés.

        C'est le point d'entrée principal de toute orchestration.
        Les routers HTTP appellent cette méthode.

        Args:
            task       : La tâche à accomplir.
            request_id : ID de la requête HTTP parente (pour les logs).

        Returns:
            ExecutionReport avec le résultat agrégé et les métriques.
        """
        request_id = request_id or str(uuid.uuid4())
        t_start    = time.monotonic()

        ctx = ExecutionContext(
            request_id = request_id,
            task       = task,
            user_id    = task.user_id,
            session_id = task.session_id,
        )

        _logger.info(
            "Orchestration démarrée",
            agent    = "orchestrator",
            endpoint = f"task:{task.type}",
        )

        # ── Sélection des agents ──────────────────────────────────────────────
        plan = await self._router.route(task)

        if plan.is_empty:
            return self._no_agent_report(task, request_id, t_start)

        # ── Exécution parallèle (Phase 8) ────────────────────────────────────
        for group in plan.parallel_groups:
            group_results = await asyncio.gather(
                *[self._run_agent(agent, task, ctx) for agent in group],
                return_exceptions=False,
            )
            for result in group_results:
                ctx.add_result(result)

        # ── Exécution séquentielle (chaîne de dépendances) ───────────────────
        for agent in plan.sequential:
            result = await self._run_agent(agent, task, ctx)
            ctx.add_result(result)

        # ── Agrégation ────────────────────────────────────────────────────────
        total_ms = round((time.monotonic() - t_start) * 1000, 1)
        report   = self._build_report(task, request_id, ctx, total_ms)

        _logger.info(
            "Orchestration terminée",
            agent      = "orchestrator",
            duration_ms= total_ms,
            success    = report.status != "failed",
        )

        return report

    async def _run_agent(
        self,
        agent: object,
        task : AgentTask,
        ctx  : ExecutionContext,
    ) -> AgentResult:
        """Exécute un agent et enregistre les métriques.

        Capture toute exception non gérée par l'agent pour garantir
        que l'Orchestrator ne crash jamais à cause d'un agent défaillant.
        """
        metrics    = get_metrics()
        agent_name = getattr(agent, "name", "unknown")
        t0         = time.monotonic()

        try:
            result = await agent.run(task, ctx)
        except Exception as exc:
            duration_ms = round((time.monotonic() - t0) * 1000, 1)
            _logger.error(
                f"Agent {agent_name} a levé une exception non gérée",
                agent      = agent_name,
                duration_ms= duration_ms,
                success    = False,
                error      = str(exc),
            )
            result = AgentResult(
                task_id    = task.task_id,
                agent_name = agent_name,
                success    = False,
                error      = f"Exception non gérée : {exc}",
                duration_ms= duration_ms,
            )

        metrics.record_pipeline_step(
            step       = agent_name,
            duration_ms= result.duration_ms,
            success    = result.success,
        )

        _logger.info(
            f"Agent {agent_name} {'réussi' if result.success else 'échoué'}",
            agent      = agent_name,
            duration_ms= result.duration_ms,
            success    = result.success,
        )

        return result

    def _build_report(
        self,
        task      : AgentTask,
        request_id: str,
        ctx       : ExecutionContext,
        total_ms  : float,
    ) -> ExecutionReport:
        """Construit le rapport final enrichi (Phase 8)."""
        successful = [r for r in ctx.results if r.success]
        failed     = [r for r in ctx.results if not r.success]

        if successful:
            status = "success"
        else:
            status = "failed"

        final_output = self._merge_outputs(ctx.results)

        confidences    = [r.confidence for r in ctx.results if r.confidence is not None]
        global_conf    = round(sum(confidences) / len(confidences), 3) if confidences else None

        metadata: dict = {
            "task_type"        : task.type,
            "task_priority"    : task.priority,
            "successful_agents": [r.agent_name for r in successful],
            "failed_agents"    : [r.agent_name for r in failed],
            "per_agent_ms"     : {r.agent_name: r.duration_ms for r in ctx.results},
        }
        if global_conf is not None:
            metadata["global_confidence"] = global_conf

        return ExecutionReport(
            request_id       = request_id,
            task_id          = task.task_id,
            status           = status,
            final_output     = final_output,
            agents_used      = [r.agent_name for r in ctx.results],
            total_duration_ms= total_ms,
            results          = list(ctx.results),
            metadata         = metadata,
        )

    def _merge_outputs(self, results: list[AgentResult]) -> str | None:
        """Fusionne les sorties de plusieurs agents réussis.

        Un seul agent → retourne son output directement.
        Plusieurs agents → trie par confiance décroissante et concatène
        avec attribution pour que l'appelant puisse distinguer les sources.
        """
        successful = [r for r in results if r.success and r.output]
        if not successful:
            return None
        if len(successful) == 1:
            return successful[0].output
        ranked = sorted(successful, key=lambda r: r.confidence or 0.0, reverse=True)
        parts  = [f"[{r.agent_name}]\n{r.output}" for r in ranked]
        return "\n\n---\n\n".join(parts)

    def _no_agent_report(
        self,
        task      : AgentTask,
        request_id: str,
        t_start   : float,
    ) -> ExecutionReport:
        """Rapport retourné quand aucun agent n'est disponible pour la tâche."""
        _logger.warning(
            f"Aucun agent disponible pour la tâche de type '{task.type}'",
            agent= "orchestrator",
        )
        return ExecutionReport(
            request_id       = request_id,
            task_id          = task.task_id,
            status           = "failed",
            final_output     = (
                "Aucun agent disponible pour traiter cette tâche. "
                "Vérifiez que les agents sont correctement enregistrés."
            ),
            agents_used      = [],
            total_duration_ms= round((time.monotonic() - t_start) * 1000, 1),
        )


# ── Singleton global ───────────────────────────────────────────────────────────

_global_orchestrator: BrainOrchestrator | None = None
import threading
_orchestrator_lock = threading.Lock()


def get_orchestrator() -> BrainOrchestrator:
    """Retourne le singleton global BrainOrchestrator.

    Thread-safe. Crée l'instance au premier appel en utilisant
    le singleton global AgentRegistry.
    """
    global _global_orchestrator
    if _global_orchestrator is None:
        with _orchestrator_lock:
            if _global_orchestrator is None:
                _global_orchestrator = BrainOrchestrator()
    return _global_orchestrator

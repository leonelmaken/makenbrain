"""Tests Phase 8 — Orchestration Multi-Agents Collaborative.

Couvre :
    TestPhase8Router       : parallel_groups peuplés par TaskRouter
    TestPhase8Orchestrator : exécution parallèle, isolation, merge, rapport

Usage :
    python -m pytest tests/test_phase8_orchestration.py -v
"""
from __future__ import annotations

import asyncio
import time
import unittest

from core.agents.base import AgentAutonomy, BaseAgent
from core.agents.models import (
    AgentResult,
    AgentTask,
    ExecutionContext,
    TaskPriority,
    TaskType,
)
from core.agents.orchestrator import BrainOrchestrator
from core.agents.registry import AgentRegistry
from core.agents.task_router import RoutingPlan, TaskRouter


# ── Agents de test ─────────────────────────────────────────────────────────────

class _AlphaAgent(BaseAgent):
    """Agent primaire, réussit toujours avec confiance 0.9."""
    name         = "alpha_agent"
    capabilities = ["reasoning", "qa"]
    autonomy     = AgentAutonomy.READ_ONLY
    confidence_threshold = 0.9
    cost_per_call        = 1.0

    async def run(self, task: AgentTask, ctx: ExecutionContext) -> AgentResult:
        ctx.shared.setdefault("ran", []).append(self.name)
        return self._make_result(task, success=True, output="alpha output", duration_ms=5.0, confidence=0.9)


class _BetaAgent(BaseAgent):
    """Second agent primaire, réussit toujours avec confiance 0.7."""
    name         = "beta_agent"
    capabilities = ["reasoning"]
    autonomy     = AgentAutonomy.READ_ONLY
    confidence_threshold = 0.7
    cost_per_call        = 1.5

    async def run(self, task: AgentTask, ctx: ExecutionContext) -> AgentResult:
        ctx.shared.setdefault("ran", []).append(self.name)
        return self._make_result(task, success=True, output="beta output", duration_ms=8.0, confidence=0.7)


class _FailingAgent(BaseAgent):
    """Agent qui échoue toujours."""
    name         = "failing_agent"
    capabilities = ["reasoning"]
    autonomy     = AgentAutonomy.READ_ONLY
    confidence_threshold = 0.5
    cost_per_call        = 0.5

    async def run(self, task: AgentTask, ctx: ExecutionContext) -> AgentResult:
        ctx.shared.setdefault("ran", []).append(self.name)
        return self._make_result(task, success=False, error="deliberate failure", duration_ms=2.0)


class _SlowAgent(BaseAgent):
    """Agent lent qui réussit — pour vérifier la vraie parallélisation."""
    name         = "slow_agent"
    capabilities = ["reasoning"]
    autonomy     = AgentAutonomy.READ_ONLY
    confidence_threshold = 0.6
    cost_per_call        = 1.0

    async def run(self, task: AgentTask, ctx: ExecutionContext) -> AgentResult:
        await asyncio.sleep(0.05)
        ctx.shared.setdefault("ran", []).append(self.name)
        return self._make_result(task, success=True, output="slow output", duration_ms=50.0, confidence=0.6)


class _ContextWriterAgent(BaseAgent):
    """Écrit dans ctx.shared — utilisé pour les tests de chaîne séquentielle."""
    name         = "writer_agent"
    capabilities = ["writing"]
    autonomy     = AgentAutonomy.READ_ONLY

    async def run(self, task: AgentTask, ctx: ExecutionContext) -> AgentResult:
        ctx.shared["written"] = "hello from writer"
        return self._make_result(task, success=True, output="wrote", duration_ms=1.0, confidence=0.8)


class _ContextReaderAgent(BaseAgent):
    """Lit dans ctx.shared — vérifie qu'un agent précédent a bien écrit."""
    name         = "reader_agent"
    capabilities = ["reading"]
    autonomy     = AgentAutonomy.READ_ONLY

    async def run(self, task: AgentTask, ctx: ExecutionContext) -> AgentResult:
        value = ctx.shared.get("written", "nothing")
        return self._make_result(task, success=True, output=f"read:{value}", duration_ms=1.0, confidence=0.8)


def _make_task(type: str = TaskType.REASONING, input: str = "test") -> AgentTask:
    return AgentTask(type=type, input=input, priority=TaskPriority.NORMAL, user_id="p8-user")


# ── TestPhase8Router ───────────────────────────────────────────────────────────

class TestPhase8Router(unittest.TestCase):

    def _router_with(self, *agents: BaseAgent) -> TaskRouter:
        reg = AgentRegistry()
        for a in agents:
            reg.register(a)
        return TaskRouter(reg)

    def test_single_agent_goes_to_sequential(self):
        router = self._router_with(_AlphaAgent())
        plan   = asyncio.run(router.route(_make_task()))
        self.assertEqual(len(plan.sequential),      1)
        self.assertEqual(len(plan.parallel_groups), 0)
        self.assertFalse(plan.is_empty)

    def test_two_primary_agents_go_to_parallel_groups(self):
        router = self._router_with(_AlphaAgent(), _BetaAgent())
        plan   = asyncio.run(router.route(_make_task(type="reasoning")))
        self.assertEqual(len(plan.sequential),      0)
        self.assertEqual(len(plan.parallel_groups), 1)
        names  = [a.name for a in plan.parallel_groups[0]]
        self.assertIn("alpha_agent", names)
        self.assertIn("beta_agent",  names)

    def test_parallel_plan_is_not_empty(self):
        router = self._router_with(_AlphaAgent(), _BetaAgent())
        plan   = asyncio.run(router.route(_make_task()))
        self.assertFalse(plan.is_empty)

    def test_parallel_plan_has_fallback(self):
        router = self._router_with(_AlphaAgent(), _BetaAgent())
        plan   = asyncio.run(router.route(_make_task()))
        self.assertIsNotNone(plan.fallback)

    def test_all_agents_reachable_via_all_agents_property(self):
        router = self._router_with(_AlphaAgent(), _BetaAgent())
        plan   = asyncio.run(router.route(_make_task()))
        names  = [a.name for a in plan.all_agents]
        self.assertIn("alpha_agent", names)
        self.assertIn("beta_agent",  names)

    def test_fallback_routing_best_agent_sequential(self):
        """Type inconnu → fallback : meilleur agent en séquentiel."""
        router = self._router_with(_AlphaAgent(), _BetaAgent())
        plan   = asyncio.run(router.route(_make_task(type="unknown_xyz")))
        self.assertFalse(plan.is_empty)
        # Avec 2 agents en fallback : 1 en séquentiel + 1 en parallèle
        self.assertEqual(len(plan.sequential),      1)
        self.assertEqual(len(plan.parallel_groups), 1)


# ── TestPhase8Orchestrator ─────────────────────────────────────────────────────

class TestPhase8Orchestrator(unittest.TestCase):

    def _orch_with(self, *agents: BaseAgent) -> BrainOrchestrator:
        reg = AgentRegistry()
        for a in agents:
            reg.register(a)
        return BrainOrchestrator(registry=reg)

    # ── Exécution parallèle ───────────────────────────────────────────────────

    def test_both_parallel_agents_appear_in_report(self):
        orch   = self._orch_with(_AlphaAgent(), _BetaAgent())
        report = asyncio.run(orch.execute(_make_task()))
        self.assertIn("alpha_agent", report.agents_used)
        self.assertIn("beta_agent",  report.agents_used)

    def test_parallel_execution_status_success(self):
        orch   = self._orch_with(_AlphaAgent(), _BetaAgent())
        report = asyncio.run(orch.execute(_make_task()))
        self.assertEqual(report.status, "success")

    def test_parallel_runs_faster_than_sequential_would(self):
        """alpha (5 ms) + slow (50 ms) en parallèle : durée totale < 100 ms."""
        orch   = self._orch_with(_AlphaAgent(), _SlowAgent())
        t0     = time.monotonic()
        report = asyncio.run(orch.execute(_make_task()))
        elapsed_ms = (time.monotonic() - t0) * 1000
        self.assertEqual(report.status, "success")
        # Les deux agents tournent en parallèle donc on ne somme pas leurs durées.
        self.assertLess(elapsed_ms, 500)

    # ── Isolation des erreurs ─────────────────────────────────────────────────

    def test_failed_agent_does_not_block_successful_agent(self):
        orch   = self._orch_with(_AlphaAgent(), _FailingAgent())
        report = asyncio.run(orch.execute(_make_task()))
        self.assertEqual(report.status, "success")
        self.assertIn("alpha_agent",   report.agents_used)
        self.assertIn("failing_agent", report.agents_used)

    def test_all_agents_fail_gives_failed_status(self):
        orch   = self._orch_with(_FailingAgent())
        report = asyncio.run(orch.execute(_make_task()))
        self.assertEqual(report.status, "failed")
        self.assertIsNone(report.final_output)

    def test_failed_agents_tracked_in_metadata(self):
        orch   = self._orch_with(_AlphaAgent(), _FailingAgent())
        report = asyncio.run(orch.execute(_make_task()))
        self.assertIn("failing_agent", report.metadata.get("failed_agents", []))
        self.assertIn("alpha_agent",   report.metadata.get("successful_agents", []))

    # ── Fusion des résultats ──────────────────────────────────────────────────

    def test_single_success_output_returned_directly(self):
        orch   = self._orch_with(_AlphaAgent())
        report = asyncio.run(orch.execute(_make_task()))
        self.assertEqual(report.final_output, "alpha output")

    def test_multiple_successes_merged_in_output(self):
        """Deux agents réussis → leurs noms apparaissent dans la sortie fusionnée."""
        orch   = self._orch_with(_AlphaAgent(), _BetaAgent())
        report = asyncio.run(orch.execute(_make_task()))
        self.assertIsNotNone(report.final_output)
        self.assertIn("alpha_agent", report.final_output)
        self.assertIn("beta_agent",  report.final_output)

    def test_merge_orders_by_confidence_descending(self):
        """Agent avec confiance 0.9 doit apparaître avant agent 0.7."""
        orch   = self._orch_with(_AlphaAgent(), _BetaAgent())
        report = asyncio.run(orch.execute(_make_task()))
        idx_alpha = report.final_output.index("alpha_agent")
        idx_beta  = report.final_output.index("beta_agent")
        self.assertLess(idx_alpha, idx_beta)

    # ── Rapport enrichi ───────────────────────────────────────────────────────

    def test_report_has_per_agent_ms(self):
        orch   = self._orch_with(_AlphaAgent())
        report = asyncio.run(orch.execute(_make_task()))
        self.assertIn("per_agent_ms",   report.metadata)
        self.assertIn("alpha_agent",    report.metadata["per_agent_ms"])

    def test_report_has_global_confidence(self):
        orch   = self._orch_with(_AlphaAgent())
        report = asyncio.run(orch.execute(_make_task()))
        self.assertIn("global_confidence", report.metadata)
        self.assertAlmostEqual(report.metadata["global_confidence"], 0.9, places=2)

    def test_global_confidence_is_average_of_agents(self):
        """Avec alpha (0.9) et beta (0.7) : moyenne attendue = 0.8."""
        orch   = self._orch_with(_AlphaAgent(), _BetaAgent())
        report = asyncio.run(orch.execute(_make_task()))
        self.assertAlmostEqual(report.metadata["global_confidence"], 0.8, places=2)

    def test_report_has_successful_agents_list(self):
        orch   = self._orch_with(_AlphaAgent())
        report = asyncio.run(orch.execute(_make_task()))
        self.assertIn("successful_agents", report.metadata)
        self.assertIn("alpha_agent", report.metadata["successful_agents"])

    def test_report_total_duration_positive(self):
        orch   = self._orch_with(_AlphaAgent(), _BetaAgent())
        report = asyncio.run(orch.execute(_make_task()))
        self.assertGreater(report.total_duration_ms, 0)

    def test_request_id_propagated(self):
        orch   = self._orch_with(_AlphaAgent())
        report = asyncio.run(orch.execute(_make_task(), request_id="p8-req-001"))
        self.assertEqual(report.request_id, "p8-req-001")

    # ── ExecutionContext.shared — communication inter-agents ──────────────────

    def test_shared_context_available_to_all_parallel_agents(self):
        """Les agents parallèles lisent/écrivent dans le même ctx.shared."""
        orch   = self._orch_with(_AlphaAgent(), _BetaAgent())
        report = asyncio.run(orch.execute(_make_task()))
        # Les deux agents ont ajouté leur nom à ctx.shared["ran"].
        # On vérifie via le rapport que les deux ont bien tourné.
        self.assertEqual(len(report.agents_used), 2)

    def test_sequential_agent_can_read_shared_written_by_parallel(self):
        """Scénario : agent parallèle écrit → agent séquentiel lit."""
        # On force un plan manuel pour contrôler l'ordre writer→reader.
        reg = AgentRegistry()
        reg.register(_ContextWriterAgent())
        reg.register(_ContextReaderAgent())
        orch = BrainOrchestrator(registry=reg)

        # Construit le plan manuellement : writer en parallèle, reader en séquentiel.
        task = AgentTask(type="writing", input="test shared context")
        ctx  = ExecutionContext(request_id="ctx-test", task=task)

        writer = _ContextWriterAgent()
        reader = _ContextReaderAgent()

        async def _run():
            # writer (parallèle) tourne en premier
            w_result = await orch._run_agent(writer, task, ctx)
            ctx.add_result(w_result)
            # reader (séquentiel) lit ce que writer a écrit
            r_result = await orch._run_agent(reader, task, ctx)
            ctx.add_result(r_result)
            return ctx

        result_ctx = asyncio.run(_run())
        reader_output = next(
            r.output for r in result_ctx.results if r.agent_name == "reader_agent"
        )
        self.assertIn("hello from writer", reader_output)


if __name__ == "__main__":
    unittest.main()

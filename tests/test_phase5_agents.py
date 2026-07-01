"""Tests Phase 5 — Architecture Multi-Agents.

Strategy d'isolation :
    Aucun test ne contacte Ollama, Groq ou ChromaDB.
    Le pipeline de raisonnement est mocké dans les tests de ReasoningAgent.

Structure :
    TestAgentModels         : AgentTask, AgentResult, ExecutionContext, ExecutionReport
    TestBaseAgent           : ABC, héritage, helpers _make_result/_timed_result
    TestAgentRegistry       : register/get/by_capability/by_task_type/snapshot
    TestTaskRouter          : routing par capacité, ranking, RoutingPlan
    TestBrainOrchestrator   : exécution séquentielle, fallback, no-agent
    TestReasoningAgent      : interface, is_available, health, run (mocké)

Usage :
    python -m pytest tests/test_phase5_agents.py -v
"""
from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from core.agents.base import AgentAutonomy, BaseAgent
from core.agents.models import (
    AgentResult,
    AgentTask,
    ExecutionContext,
    ExecutionReport,
    TaskPriority,
    TaskType,
)
from core.agents.orchestrator import BrainOrchestrator
from core.agents.reasoning_agent import ReasoningAgent
from core.agents.registry import AgentRegistry
from core.agents.task_router import RoutingPlan, TaskRouter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_task(
    input: str      = "test question",
    type: str       = TaskType.REASONING,
    priority: int   = TaskPriority.NORMAL,
    user_id: str    = "user-test",
) -> AgentTask:
    return AgentTask(type=type, input=input, priority=priority, user_id=user_id)


class _OkAgent(BaseAgent):
    """Agent de test qui réussit toujours."""
    name         = "ok_agent"
    description  = "Agent de test OK"
    capabilities = ["reasoning", "qa"]
    autonomy     = AgentAutonomy.READ_ONLY

    async def run(self, task, ctx) -> AgentResult:
        return self._make_result(task, success=True, output="ok response", duration_ms=10.0)


class _FailAgent(BaseAgent):
    """Agent de test qui échoue toujours."""
    name         = "fail_agent"
    description  = "Agent de test FAIL"
    capabilities = ["reasoning"]
    autonomy     = AgentAutonomy.READ_ONLY

    async def run(self, task, ctx) -> AgentResult:
        return self._make_result(task, success=False, error="always fails", duration_ms=5.0)


class _UnavailableAgent(BaseAgent):
    """Agent de test toujours indisponible."""
    name         = "unavailable_agent"
    description  = "Agent indisponible"
    capabilities = ["reasoning"]
    autonomy     = AgentAutonomy.READ_ONLY

    async def is_available(self) -> bool:
        return False

    async def run(self, task, ctx) -> AgentResult:
        return self._make_result(task, success=False, error="unavailable")


# ── TestAgentModels ───────────────────────────────────────────────────────────

class TestAgentModels(unittest.TestCase):

    def test_agent_task_defaults(self):
        task = AgentTask(type=TaskType.REASONING, input="test")
        self.assertIsNotNone(task.task_id)
        self.assertEqual(task.priority, TaskPriority.NORMAL)
        self.assertIsNone(task.user_id)

    def test_agent_task_uuid_unique(self):
        t1 = AgentTask(type="reasoning", input="a")
        t2 = AgentTask(type="reasoning", input="b")
        self.assertNotEqual(t1.task_id, t2.task_id)

    def test_agent_result_to_dict_success(self):
        r = AgentResult(task_id="t1", agent_name="ok_agent", success=True,
                        output="hello", duration_ms=42.0, confidence=0.9)
        d = r.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["output"],     "hello")
        self.assertEqual(d["confidence"], 0.9)
        self.assertNotIn("error", d)

    def test_agent_result_to_dict_failure(self):
        r = AgentResult(task_id="t1", agent_name="fail", success=False,
                        error="timeout", duration_ms=5.0)
        d = r.to_dict()
        self.assertFalse(d["success"])
        self.assertEqual(d["error"], "timeout")
        self.assertNotIn("output", d)

    def test_execution_context_add_result(self):
        task = _make_task()
        ctx  = ExecutionContext(request_id="req-1", task=task)
        r    = AgentResult(task_id=task.task_id, agent_name="ok", success=True, output="x")
        ctx.add_result(r)
        self.assertEqual(len(ctx.results), 1)

    def test_execution_context_last_successful_output(self):
        task = _make_task()
        ctx  = ExecutionContext(request_id="req-1", task=task)
        ctx.add_result(AgentResult(task_id=task.task_id, agent_name="a",
                                   success=False, error="e"))
        ctx.add_result(AgentResult(task_id=task.task_id, agent_name="b",
                                   success=True, output="best output"))
        self.assertEqual(ctx.last_successful_output, "best output")

    def test_execution_context_last_successful_none_if_all_failed(self):
        task = _make_task()
        ctx  = ExecutionContext(request_id="req-1", task=task)
        ctx.add_result(AgentResult(task_id=task.task_id, agent_name="a",
                                   success=False, error="e"))
        self.assertIsNone(ctx.last_successful_output)

    def test_execution_report_to_dict(self):
        report = ExecutionReport(
            request_id       = "req-1",
            task_id          = "task-1",
            status           = "success",
            final_output     = "réponse",
            agents_used      = ["ok_agent"],
            total_duration_ms= 55.0,
        )
        d = report.to_dict()
        self.assertEqual(d["status"],      "success")
        self.assertEqual(d["final_output"], "réponse")
        self.assertIn("agents_used", d)

    def test_task_priority_constants(self):
        self.assertLess(TaskPriority.LOW, TaskPriority.NORMAL)
        self.assertLess(TaskPriority.NORMAL, TaskPriority.HIGH)
        self.assertLess(TaskPriority.HIGH, TaskPriority.CRITICAL)


# ── TestBaseAgent ─────────────────────────────────────────────────────────────

class TestBaseAgent(unittest.TestCase):

    def test_cannot_instantiate_abstract(self):
        with self.assertRaises(TypeError):
            BaseAgent()  # type: ignore

    def test_ok_agent_instantiates(self):
        agent = _OkAgent()
        self.assertEqual(agent.name, "ok_agent")

    def test_is_available_default_true(self):
        agent = _OkAgent()
        result = asyncio.run(agent.is_available())
        self.assertTrue(result)

    def test_health_returns_dict(self):
        agent = _OkAgent()
        h = asyncio.run(agent.health())
        self.assertIn("name", h)
        self.assertIn("available", h)
        self.assertEqual(h["name"], "ok_agent")

    def test_make_result_helper(self):
        agent = _OkAgent()
        task  = _make_task()
        r = agent._make_result(task, success=True, output="ok", duration_ms=10.0)
        self.assertEqual(r.task_id,    task.task_id)
        self.assertEqual(r.agent_name, "ok_agent")
        self.assertTrue(r.success)

    def test_timed_result_helper_duration(self):
        agent  = _OkAgent()
        task   = _make_task()
        t0     = time.monotonic()
        result = agent._timed_result(task, t0, success=True, output="x")
        self.assertGreaterEqual(result.duration_ms, 0.0)


# ── TestAgentRegistry ─────────────────────────────────────────────────────────

class TestAgentRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = AgentRegistry()

    def test_register_and_get(self):
        agent = _OkAgent()
        self.registry.register(agent)
        self.assertIs(self.registry.get("ok_agent"), agent)

    def test_get_unknown_returns_none(self):
        self.assertIsNone(self.registry.get("nonexistent"))

    def test_unregister(self):
        agent = _OkAgent()
        self.registry.register(agent)
        self.registry.unregister("ok_agent")
        self.assertIsNone(self.registry.get("ok_agent"))

    def test_list_all(self):
        self.registry.register(_OkAgent())
        self.registry.register(_FailAgent())
        all_agents = self.registry.list_all()
        self.assertEqual(len(all_agents), 2)

    def test_by_capability(self):
        self.registry.register(_OkAgent())
        self.registry.register(_FailAgent())
        qa_agents = self.registry.by_capability("qa")
        names = [a.name for a in qa_agents]
        self.assertIn("ok_agent", names)
        self.assertNotIn("fail_agent", names)

    def test_by_task_type_reasoning(self):
        self.registry.register(_OkAgent())
        agents = self.registry.by_task_type("reasoning")
        self.assertTrue(len(agents) > 0)
        self.assertEqual(agents[0].name, "ok_agent")

    def test_by_task_type_unknown_returns_empty(self):
        self.registry.register(_OkAgent())
        agents = self.registry.by_task_type("cooking")
        self.assertEqual(agents, [])

    def test_snapshot_returns_list_of_dicts(self):
        self.registry.register(_OkAgent())
        snap = self.registry.snapshot()
        self.assertIsInstance(snap, list)
        self.assertEqual(snap[0]["name"], "ok_agent")
        self.assertIn("capabilities", snap[0])

    def test_len(self):
        self.assertEqual(len(self.registry), 0)
        self.registry.register(_OkAgent())
        self.assertEqual(len(self.registry), 1)

    def test_register_replaces_existing(self):
        self.registry.register(_OkAgent())
        self.registry.register(_OkAgent())
        self.assertEqual(len(self.registry), 1)


# ── TestTaskRouter ────────────────────────────────────────────────────────────

class TestTaskRouter(unittest.TestCase):

    def _router_with(self, *agents: BaseAgent) -> TaskRouter:
        registry = AgentRegistry()
        for a in agents:
            registry.register(a)
        return TaskRouter(registry)

    def test_routes_to_capable_agent(self):
        router = self._router_with(_OkAgent())
        task   = _make_task(type="reasoning")
        plan   = asyncio.run(router.route(task))
        self.assertFalse(plan.is_empty)
        self.assertIn(_OkAgent.name, [a.name for a in plan.sequential])

    def test_excludes_unavailable_agents(self):
        router = self._router_with(_UnavailableAgent())
        task   = _make_task(type="reasoning")
        plan   = asyncio.run(router.route(task))
        self.assertTrue(plan.is_empty)

    def test_empty_plan_when_no_agent_registered(self):
        router = self._router_with()
        plan   = asyncio.run(router.route(_make_task()))
        self.assertTrue(plan.is_empty)

    def test_fallback_agent_set(self):
        router = self._router_with(_OkAgent(), _FailAgent())
        task   = _make_task(type="reasoning")
        plan   = asyncio.run(router.route(task))
        self.assertIsNotNone(plan.fallback)

    def test_routing_plan_all_agents(self):
        router = self._router_with(_OkAgent())
        task   = _make_task(type="reasoning")
        plan   = asyncio.run(router.route(task))
        self.assertEqual(len(plan.all_agents), len(plan.sequential))

    def test_fallback_to_any_agent_when_no_match_by_type(self):
        router = self._router_with(_OkAgent())
        task   = _make_task(type="planning")
        plan   = asyncio.run(router.route(task))
        # Doit fallback vers les agents disponibles même si type ne correspond pas
        self.assertFalse(plan.is_empty)


# ── TestBrainOrchestrator ─────────────────────────────────────────────────────

class TestBrainOrchestrator(unittest.TestCase):

    def _orchestrator_with(self, *agents: BaseAgent) -> BrainOrchestrator:
        registry = AgentRegistry()
        for a in agents:
            registry.register(a)
        return BrainOrchestrator(registry=registry)

    def test_success_with_ok_agent(self):
        orch   = self._orchestrator_with(_OkAgent())
        task   = _make_task()
        report = asyncio.run(orch.execute(task))
        self.assertEqual(report.status, "success")
        self.assertEqual(report.final_output, "ok response")
        self.assertIn("ok_agent", report.agents_used)

    def test_failed_when_agent_fails(self):
        orch   = self._orchestrator_with(_FailAgent())
        task   = _make_task()
        report = asyncio.run(orch.execute(task))
        self.assertEqual(report.status, "failed")
        self.assertIsNone(report.final_output)

    def test_fallback_to_ok_agent_after_fail(self):
        orch   = self._orchestrator_with(_FailAgent(), _OkAgent())
        task   = _make_task()
        report = asyncio.run(orch.execute(task))
        self.assertEqual(report.status, "success")

    def test_no_agent_returns_failed_report(self):
        orch   = BrainOrchestrator(registry=AgentRegistry())
        task   = _make_task()
        report = asyncio.run(orch.execute(task))
        self.assertEqual(report.status, "failed")
        self.assertIsNotNone(report.final_output)

    def test_report_contains_task_id(self):
        orch   = self._orchestrator_with(_OkAgent())
        task   = _make_task()
        report = asyncio.run(orch.execute(task))
        self.assertEqual(report.task_id, task.task_id)

    def test_report_total_duration_positive(self):
        orch   = self._orchestrator_with(_OkAgent())
        task   = _make_task()
        report = asyncio.run(orch.execute(task))
        self.assertGreater(report.total_duration_ms, 0)

    def test_agent_exception_does_not_crash_orchestrator(self):
        """Si un agent lève une exception non gérée, l'Orchestrator continue."""
        class CrashAgent(BaseAgent):
            name         = "crash_agent"
            capabilities = ["reasoning"]
            autonomy     = AgentAutonomy.READ_ONLY
            async def run(self, task, ctx):
                raise RuntimeError("Agent crash intentionnel (test)")

        orch   = BrainOrchestrator(registry=AgentRegistry())
        orch._registry.register(CrashAgent())
        task   = _make_task()
        # Ne doit pas lever d'exception
        report = asyncio.run(orch.execute(task))
        self.assertEqual(report.status, "failed")

    def test_request_id_propagated(self):
        orch   = self._orchestrator_with(_OkAgent())
        task   = _make_task()
        report = asyncio.run(orch.execute(task, request_id="custom-req-id"))
        self.assertEqual(report.request_id, "custom-req-id")


# ── TestReasoningAgent ────────────────────────────────────────────────────────

class TestReasoningAgent(unittest.TestCase):

    def test_name_and_capabilities(self):
        agent = ReasoningAgent()
        self.assertEqual(agent.name, "reasoning_agent")
        self.assertIn("reasoning", agent.capabilities)
        self.assertIn("qa", agent.capabilities)

    def test_autonomy_is_read_only(self):
        self.assertEqual(ReasoningAgent.autonomy, AgentAutonomy.READ_ONLY)

    def test_is_available_true(self):
        agent = ReasoningAgent()
        result = asyncio.run(agent.is_available())
        self.assertTrue(result)

    def test_health_returns_pipeline_info(self):
        agent = ReasoningAgent()
        h = asyncio.run(agent.health())
        self.assertIn("pipeline", h)
        self.assertIn("3.1", h["pipeline"])

    def test_run_success_with_mocked_pipeline(self):
        """run() retourne un résultat réussi si le pipeline fonctionne."""
        agent = ReasoningAgent()
        task  = _make_task(input="Quelle est ma stratégie ?")
        ctx   = ExecutionContext(request_id="req-1", task=task)

        mock_ctx = MagicMock()
        mock_ctx.final_answer    = "Voici ma réponse."
        mock_ctx.confidence      = 0.85
        mock_ctx.synthesis_result= MagicMock()
        mock_ctx.synthesis_result.strategy.value = "llm"
        mock_ctx.hypotheses      = [1, 2]
        mock_ctx.evidence        = [1]
        mock_ctx.reasoning_trace = [1, 2, 3, 4, 5]

        with patch("core.agents.reasoning_agent.run_reasoning",
                   new=AsyncMock(return_value=mock_ctx)):
            result = asyncio.run(agent.run(task, ctx))

        self.assertTrue(result.success)
        self.assertEqual(result.output, "Voici ma réponse.")
        self.assertAlmostEqual(result.confidence, 0.85)
        self.assertEqual(result.agent_name, "reasoning_agent")

    def test_run_failure_when_no_final_answer(self):
        """run() retourne success=False si le pipeline ne produit aucune réponse."""
        agent = ReasoningAgent()
        task  = _make_task()
        ctx   = ExecutionContext(request_id="req-1", task=task)

        mock_ctx = MagicMock()
        mock_ctx.final_answer    = None
        mock_ctx.confidence      = 0.0
        mock_ctx.synthesis_result= None
        mock_ctx.hypotheses      = []
        mock_ctx.evidence        = []
        mock_ctx.reasoning_trace = []

        with patch("core.agents.reasoning_agent.run_reasoning",
                   new=AsyncMock(return_value=mock_ctx)):
            result = asyncio.run(agent.run(task, ctx))

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_run_failure_on_exception(self):
        """run() capture les exceptions et retourne success=False."""
        agent = ReasoningAgent()
        task  = _make_task()
        ctx   = ExecutionContext(request_id="req-1", task=task)

        with patch("core.agents.reasoning_agent.run_reasoning",
                   new=AsyncMock(side_effect=RuntimeError("pipeline crash"))):
            result = asyncio.run(agent.run(task, ctx))

        self.assertFalse(result.success)
        self.assertIn("pipeline crash", result.error)

    def test_run_with_shared_context_succeeds(self):
        """run() fonctionne correctement quand ctx.shared['memory'] est renseigné."""
        agent = ReasoningAgent()
        task  = _make_task()
        ctx   = ExecutionContext(request_id="req-1", task=task)
        ctx.shared["memory"] = "Contexte mémoire important"

        mock_ctx = MagicMock()
        mock_ctx.final_answer = "ok avec mémoire"
        mock_ctx.confidence   = 0.8
        mock_ctx.synthesis_result = MagicMock()
        mock_ctx.synthesis_result.strategy.value = "template"
        mock_ctx.hypotheses = []
        mock_ctx.evidence   = []
        mock_ctx.reasoning_trace = []

        with patch("core.agents.reasoning_agent.run_reasoning",
                   new=AsyncMock(return_value=mock_ctx)):
            result = asyncio.run(agent.run(task, ctx))

        self.assertTrue(result.success)
        self.assertEqual(result.output, "ok avec mémoire")


if __name__ == "__main__":
    unittest.main()

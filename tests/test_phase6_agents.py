"""Tests Phase 6 — MemoryAgent, PlanningAgent, EpisodicMemory, ResearchAgent.

Strategy d'isolation :
    - ChromaDB est mocké dans tous les tests MemoryAgent.
    - Le LLM Router est mocké dans les tests PlanningAgent.
    - EpisodicMemoryStore utilise un fichier temporaire (tmp_path).
    - ResearchAgent : tests d'interface uniquement, pas d'appels réseau.

Structure :
    TestEpisodicModels        : EpisodicEntry sérialisation/désérialisation
    TestEpisodicStore         : record, get_recent, get_by_type, get_by_agent
    TestMemoryAgentSearch     : action search avec mock ChromaDB
    TestMemoryAgentWrite      : action write avec mock ChromaDB
    TestMemoryAgentDelete     : action delete avec mock ChromaDB
    TestMemoryAgentStats      : action stats avec mock ChromaDB
    TestMemoryAgentErrors     : actions inconnues, memory_id manquant
    TestPlanningAgentFallback : plan déterministe sans LLM
    TestPlanningAgentLLM      : plan avec LLM mocké
    TestPlanningModels        : SubTask, ExecutionPlan sérialisation
    TestResearchDTOs          : ResearchQuery, ResearchSource, ResearchResult
    TestResearchAgent         : interface stub, plan_research, research

Usage :
    python -m pytest tests/test_phase6_agents.py -v
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from core.agents.models import AgentTask, ExecutionContext, TaskPriority, TaskType
from core.agents.planning_agent import ExecutionPlan, PlanningAgent, SubTask
from core.agents.research_agent import (
    ResearchAgent,
    ResearchDepth,
    ResearchPlan,
    ResearchQuery,
    ResearchResult,
    ResearchSource,
    SourceType,
)
from core.episodic.models import EpisodicEntry, EpisodicType
from core.episodic.store import EpisodicMemoryStore


# ── Helpers ───────────────────────────────────────────────────────────────────

def _task(
    input: str      = "test",
    type: str       = TaskType.MEMORY,
    context: dict   = None,
    user_id: str    = "u-test",
) -> AgentTask:
    return AgentTask(type=type, input=input, user_id=user_id, context=context or {})


def _ctx(task: AgentTask) -> ExecutionContext:
    return ExecutionContext(request_id="req-test", task=task)


# ── TestEpisodicModels ────────────────────────────────────────────────────────

class TestEpisodicModels(unittest.TestCase):

    def test_entry_defaults(self):
        e = EpisodicEntry(type=EpisodicType.ACTION, agent_name="test", content="did something")
        self.assertIsNotNone(e.entry_id)
        self.assertIsNotNone(e.timestamp)
        self.assertIsNone(e.task_id)
        self.assertIsNone(e.user_id)

    def test_to_dict_required_fields(self):
        e = EpisodicEntry(type=EpisodicType.DECISION, agent_name="planning_agent",
                          content="Chose option A")
        d = e.to_dict()
        self.assertEqual(d["type"],       "decision")
        self.assertEqual(d["agent_name"], "planning_agent")
        self.assertEqual(d["content"],    "Chose option A")
        self.assertIn("entry_id",  d)
        self.assertIn("timestamp", d)

    def test_to_dict_omits_none_fields(self):
        e = EpisodicEntry(type=EpisodicType.ACTION, agent_name="x", content="y")
        d = e.to_dict()
        self.assertNotIn("task_id", d)
        self.assertNotIn("user_id", d)

    def test_to_dict_includes_optional_fields(self):
        e = EpisodicEntry(
            type=EpisodicType.PROJECT, agent_name="a", content="b",
            task_id="task-1", user_id="user-1", metadata={"key": "val"},
        )
        d = e.to_dict()
        self.assertEqual(d["task_id"], "task-1")
        self.assertEqual(d["user_id"], "user-1")
        self.assertEqual(d["metadata"]["key"], "val")

    def test_from_dict_roundtrip(self):
        e = EpisodicEntry(
            type=EpisodicType.LEARNING, agent_name="reasoning_agent",
            content="Learned X", task_id="t1", user_id="u1",
        )
        reconstructed = EpisodicEntry.from_dict(e.to_dict())
        self.assertEqual(reconstructed.entry_id,   e.entry_id)
        self.assertEqual(reconstructed.type,       e.type)
        self.assertEqual(reconstructed.agent_name, e.agent_name)
        self.assertEqual(reconstructed.content,    e.content)

    def test_episodic_type_values(self):
        self.assertEqual(EpisodicType.ACTION,   "action")
        self.assertEqual(EpisodicType.DECISION, "decision")
        self.assertEqual(EpisodicType.LEARNING, "learning")
        self.assertEqual(EpisodicType.PROJECT,  "project")
        self.assertEqual(EpisodicType.ERROR,    "error")


# ── TestEpisodicStore ─────────────────────────────────────────────────────────

class TestEpisodicStore(unittest.TestCase):

    def _store(self) -> EpisodicMemoryStore:
        tmp = tempfile.mktemp(suffix=".jsonl")
        return EpisodicMemoryStore(path=Path(tmp))

    def _entry(self, type=EpisodicType.ACTION, agent="agent_a", content="event") -> EpisodicEntry:
        return EpisodicEntry(type=type, agent_name=agent, content=content)

    def test_record_and_count(self):
        store = self._store()
        store.record(self._entry())
        store.record(self._entry())
        self.assertEqual(store.count(), 2)

    def test_get_recent_returns_most_recent_first(self):
        store = self._store()
        store.record(self._entry(content="first"))
        store.record(self._entry(content="second"))
        store.record(self._entry(content="third"))
        recent = store.get_recent(2)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0].content, "third")
        self.assertEqual(recent[1].content, "second")

    def test_get_recent_respects_limit(self):
        store = self._store()
        for i in range(10):
            store.record(self._entry(content=f"event_{i}"))
        recent = store.get_recent(3)
        self.assertEqual(len(recent), 3)

    def test_get_by_type_filters(self):
        store = self._store()
        store.record(self._entry(type=EpisodicType.ACTION,   content="action"))
        store.record(self._entry(type=EpisodicType.DECISION, content="decision"))
        store.record(self._entry(type=EpisodicType.ACTION,   content="action2"))
        actions = store.get_by_type(EpisodicType.ACTION)
        self.assertEqual(len(actions), 2)
        decisions = store.get_by_type(EpisodicType.DECISION)
        self.assertEqual(len(decisions), 1)

    def test_get_by_agent_filters(self):
        store = self._store()
        store.record(self._entry(agent="agent_a"))
        store.record(self._entry(agent="agent_b"))
        store.record(self._entry(agent="agent_a"))
        results = store.get_by_agent("agent_a")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(e.agent_name == "agent_a" for e in results))

    def test_empty_store_returns_empty_list(self):
        store = self._store()
        self.assertEqual(store.get_recent(), [])
        self.assertEqual(store.get_by_type(EpisodicType.ACTION), [])

    def test_count_zero_on_empty(self):
        store = self._store()
        self.assertEqual(store.count(), 0)

    def test_file_persists_between_instances(self):
        tmp = Path(tempfile.mktemp(suffix=".jsonl"))
        store1 = EpisodicMemoryStore(path=tmp)
        store1.record(self._entry(content="persistent"))
        store2 = EpisodicMemoryStore(path=tmp)
        self.assertEqual(store2.count(), 1)
        self.assertEqual(store2.get_recent(1)[0].content, "persistent")


# ── TestMemoryAgentSearch ─────────────────────────────────────────────────────

class TestMemoryAgentSearch(unittest.TestCase):

    def _run(self, task, mock_search_return):
        from core.agents.memory_agent import MemoryAgent
        agent = MemoryAgent()
        ctx   = _ctx(task)
        with patch("core.agents.memory_agent.search_memory",
                   new=AsyncMock(return_value=mock_search_return)):
            return asyncio.run(agent.run(task, ctx)), ctx

    def test_search_success(self):
        task = _task(input="Ma stratégie")
        memories = [{"id": "m1", "content": "contenu", "distance": 0.1, "metadata": {}}]
        result, ctx = self._run(task, memories)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.output)
        data = json.loads(result.output)
        self.assertEqual(len(data), 1)

    def test_search_stores_in_shared(self):
        task = _task(input="query")
        memories = [{"id": "m1", "content": "x", "distance": 0.2, "metadata": {}}]
        result, ctx = self._run(task, memories)
        self.assertIn("memory_results", ctx.shared)
        self.assertEqual(len(ctx.shared["memory_results"]), 1)

    def test_search_empty_results(self):
        task = _task(input="inconnu")
        result, ctx = self._run(task, [])
        self.assertTrue(result.success)
        data = json.loads(result.output)
        self.assertEqual(data, [])

    def test_search_default_action(self):
        """Sans action explicite, search est l'action par défaut."""
        task = _task(input="default search", context={})
        memories = []
        result, _ = self._run(task, memories)
        self.assertTrue(result.success)


# ── TestMemoryAgentWrite ──────────────────────────────────────────────────────

class TestMemoryAgentWrite(unittest.TestCase):

    def _run_write(self, content, metadata=None):
        from core.agents.memory_agent import MemoryAgent
        task = _task(
            input   = content,
            context = {"action": "write", "metadata": metadata or {}},
        )
        ctx  = _ctx(task)
        with patch("core.agents.memory_agent.add_memory",
                   new=AsyncMock(return_value="mem-uuid-123")):
            with patch("core.episodic.store.get_episodic_store", return_value=MagicMock()):
                result = asyncio.run(MemoryAgent().run(task, ctx))
        return result

    def test_write_success(self):
        result = self._run_write("Nouveau souvenir")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "mem-uuid-123")

    def test_write_returns_memory_id(self):
        result = self._run_write("test content")
        self.assertIn("memory_id", result.metadata)

    def test_write_with_metadata(self):
        result = self._run_write("content", {"source": "user", "tags": "important"})
        self.assertTrue(result.success)


# ── TestMemoryAgentDelete ─────────────────────────────────────────────────────

class TestMemoryAgentDelete(unittest.TestCase):

    def _run_delete(self, memory_id=None, delete_return=True):
        from core.agents.memory_agent import MemoryAgent
        task = _task(
            input   = "",
            context = {"action": "delete", **({"memory_id": memory_id} if memory_id else {})},
        )
        ctx = _ctx(task)
        with patch("core.agents.memory_agent.delete_memory",
                   new=AsyncMock(return_value=delete_return)):
            with patch("core.episodic.store.get_episodic_store", return_value=MagicMock()):
                return asyncio.run(MemoryAgent().run(task, ctx))

    def test_delete_success(self):
        result = self._run_delete(memory_id="mem-123")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "mem-123")

    def test_delete_failure(self):
        result = self._run_delete(memory_id="mem-404", delete_return=False)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_delete_missing_memory_id(self):
        result = self._run_delete(memory_id=None)
        self.assertFalse(result.success)
        self.assertIn("memory_id", result.error)


# ── TestMemoryAgentStats ──────────────────────────────────────────────────────

class TestMemoryAgentStats(unittest.TestCase):

    def test_stats_success(self):
        from core.agents.memory_agent import MemoryAgent
        task = _task(context={"action": "stats"})
        ctx  = _ctx(task)
        mock_stats = {"total_memories": 42, "collection": "makenbrain"}
        with patch("core.agents.memory_agent.get_memory_stats",
                   new=AsyncMock(return_value=mock_stats)):
            result = asyncio.run(MemoryAgent().run(task, ctx))
        self.assertTrue(result.success)
        data = json.loads(result.output)
        self.assertEqual(data["total_memories"], 42)


# ── TestMemoryAgentErrors ─────────────────────────────────────────────────────

class TestMemoryAgentErrors(unittest.TestCase):

    def test_unknown_action(self):
        from core.agents.memory_agent import MemoryAgent
        task = _task(context={"action": "fly_to_mars"})
        result = asyncio.run(MemoryAgent().run(task, _ctx(task)))
        self.assertFalse(result.success)
        self.assertIn("fly_to_mars", result.error)

    def test_search_exception_caught(self):
        from core.agents.memory_agent import MemoryAgent
        task = _task(input="query")
        with patch("core.agents.memory_agent.search_memory",
                   new=AsyncMock(side_effect=RuntimeError("ChromaDB offline"))):
            result = asyncio.run(MemoryAgent().run(task, _ctx(task)))
        self.assertFalse(result.success)
        self.assertIn("ChromaDB offline", result.error)

    def test_name_and_autonomy(self):
        from core.agents.memory_agent import MemoryAgent
        from core.agents.base import AgentAutonomy
        agent = MemoryAgent()
        self.assertEqual(agent.name, "memory_agent")
        self.assertEqual(agent.autonomy, AgentAutonomy.SANDBOXED_EXECUTE)
        self.assertIn("memory", agent.capabilities)


# ── TestPlanningModels ────────────────────────────────────────────────────────

class TestPlanningModels(unittest.TestCase):

    def test_subtask_to_dict(self):
        s = SubTask(subtask_id="step_1", title="Analyser")
        d = s.to_dict()
        self.assertEqual(d["subtask_id"], "step_1")
        self.assertEqual(d["title"],      "Analyser")
        self.assertIn("dependencies", d)

    def test_subtask_dependencies(self):
        s = SubTask(subtask_id="s3", title="X", dependencies=["s1", "s2"])
        d = s.to_dict()
        self.assertEqual(d["dependencies"], ["s1", "s2"])

    def test_execution_plan_to_dict(self):
        plan = ExecutionPlan(
            goal    = "Mon objectif",
            subtasks= [SubTask(subtask_id="s1", title="Étape 1")],
        )
        d = plan.to_dict()
        self.assertEqual(d["goal"],         "Mon objectif")
        self.assertEqual(len(d["subtasks"]), 1)
        self.assertIn("plan_id", d)

    def test_execution_plan_unique_ids(self):
        p1 = ExecutionPlan(goal="a")
        p2 = ExecutionPlan(goal="b")
        self.assertNotEqual(p1.plan_id, p2.plan_id)


# ── TestPlanningAgentFallback ─────────────────────────────────────────────────

class TestPlanningAgentFallback(unittest.TestCase):

    def _run_with_llm_fail(self, input_text="Créer une app mobile"):
        task  = _task(input=input_text, type=TaskType.PLANNING)
        ctx   = _ctx(task)
        agent = PlanningAgent()
        with patch("core.agents.planning_agent.get_router") as mock_router_fn:
            mock_router = MagicMock()
            mock_router.generate = AsyncMock(side_effect=RuntimeError("LLM offline"))
            mock_router_fn.return_value = mock_router
            result = asyncio.run(agent.run(task, ctx))
        return result, ctx

    def test_fallback_success(self):
        result, _ = self._run_with_llm_fail()
        self.assertTrue(result.success)

    def test_fallback_has_subtasks(self):
        result, _ = self._run_with_llm_fail()
        plan = json.loads(result.output)
        self.assertGreater(len(plan["subtasks"]), 0)

    def test_fallback_strategy_in_metadata(self):
        result, _ = self._run_with_llm_fail()
        self.assertEqual(result.metadata.get("strategy"), "fallback")

    def test_fallback_stores_plan_in_shared(self):
        _, ctx = self._run_with_llm_fail()
        self.assertIn("execution_plan", ctx.shared)


# ── TestPlanningAgentLLM ──────────────────────────────────────────────────────

class TestPlanningAgentLLM(unittest.TestCase):

    _VALID_JSON = json.dumps({
        "goal": "Construire une application mobile",
        "subtasks": [
            {
                "subtask_id": "step_1",
                "title"     : "Analyser les besoins",
                "type"      : "reasoning",
                "description": "Identifier les fonctionnalités clés.",
                "priority"  : 8,
                "dependencies": [],
            },
            {
                "subtask_id" : "step_2",
                "title"      : "Concevoir l'architecture",
                "type"       : "planning",
                "description": "Définir la structure technique.",
                "priority"   : 7,
                "dependencies": ["step_1"],
            },
        ],
    })

    def _run_with_llm_ok(self, llm_response):
        task  = _task(input="Construire une app", type=TaskType.PLANNING)
        ctx   = _ctx(task)
        agent = PlanningAgent()
        with patch("core.agents.planning_agent.get_router") as mock_router_fn:
            mock_router = MagicMock()
            mock_router.generate = AsyncMock(return_value=llm_response)
            mock_router_fn.return_value = mock_router
            result = asyncio.run(agent.run(task, ctx))
        return result, ctx

    def test_llm_success(self):
        result, _ = self._run_with_llm_ok(self._VALID_JSON)
        self.assertTrue(result.success)

    def test_llm_plan_has_two_subtasks(self):
        result, _ = self._run_with_llm_ok(self._VALID_JSON)
        plan = json.loads(result.output)
        self.assertEqual(len(plan["subtasks"]), 2)

    def test_llm_strategy_in_metadata(self):
        result, _ = self._run_with_llm_ok(self._VALID_JSON)
        self.assertEqual(result.metadata.get("strategy"), "llm")

    def test_llm_dependencies_preserved(self):
        result, _ = self._run_with_llm_ok(self._VALID_JSON)
        plan = json.loads(result.output)
        step_2 = next(s for s in plan["subtasks"] if s["subtask_id"] == "step_2")
        self.assertEqual(step_2["dependencies"], ["step_1"])

    def test_invalid_json_falls_back(self):
        """Si le LLM retourne du texte sans JSON valide, fallback s'active."""
        result, _ = self._run_with_llm_ok("Je pense que voici ce que tu devrais faire...")
        self.assertTrue(result.success)
        self.assertEqual(result.metadata.get("strategy"), "fallback")

    def test_name_and_autonomy(self):
        from core.agents.base import AgentAutonomy
        agent = PlanningAgent()
        self.assertEqual(agent.name, "planning_agent")
        self.assertEqual(agent.autonomy, AgentAutonomy.SUGGEST)
        self.assertIn("planning", agent.capabilities)


# ── TestResearchDTOs ──────────────────────────────────────────────────────────

class TestResearchDTOs(unittest.TestCase):

    def test_research_query_defaults(self):
        q = ResearchQuery(query="Qu'est-ce que l'IA ?")
        self.assertEqual(q.depth, ResearchDepth.STANDARD)
        self.assertEqual(q.max_sources, 5)
        self.assertEqual(q.language, "fr")

    def test_research_query_to_dict(self):
        q = ResearchQuery(query="test", depth=ResearchDepth.DEEP, max_sources=10)
        d = q.to_dict()
        self.assertEqual(d["query"],       "test")
        self.assertEqual(d["depth"],       "deep")
        self.assertEqual(d["max_sources"], 10)

    def test_research_source_to_dict(self):
        s = ResearchSource(
            title          = "Wikipedia IA",
            snippet        = "L'IA est...",
            url            = "https://fr.wikipedia.org/wiki/Intelligence_artificielle",
            relevance_score= 0.95,
            source_type    = SourceType.WIKIPEDIA,
        )
        d = s.to_dict()
        self.assertEqual(d["title"],           "Wikipedia IA")
        self.assertEqual(d["relevance_score"], 0.95)
        self.assertEqual(d["source_type"],     "wikipedia")

    def test_research_source_omits_empty_url(self):
        s = ResearchSource(title="x", snippet="y")
        d = s.to_dict()
        self.assertNotIn("url", d)

    def test_research_result_to_dict(self):
        r = ResearchResult(
            query   = "test",
            summary = "Voici ce que j'ai trouvé.",
            confidence= 0.8,
        )
        d = r.to_dict()
        self.assertEqual(d["query"],      "test")
        self.assertEqual(d["confidence"], 0.8)
        self.assertEqual(d["sources"],    [])

    def test_research_plan_to_dict(self):
        plan = ResearchPlan(steps=["Étape 1", "Étape 2"], estimated_sources=5)
        d = plan.to_dict()
        self.assertEqual(len(d["steps"]), 2)
        self.assertEqual(d["estimated_sources"], 5)

    def test_research_depth_values(self):
        self.assertEqual(ResearchDepth.QUICK,    "quick")
        self.assertEqual(ResearchDepth.STANDARD, "standard")
        self.assertEqual(ResearchDepth.DEEP,     "deep")

    def test_source_type_values(self):
        self.assertEqual(SourceType.WEB,       "web")
        self.assertEqual(SourceType.WIKIPEDIA, "wikipedia")


# ── TestResearchAgent ─────────────────────────────────────────────────────────

class TestResearchAgent(unittest.TestCase):

    def test_name_and_capabilities(self):
        agent = ResearchAgent()
        self.assertEqual(agent.name, "research_agent")
        self.assertIn("research", agent.capabilities)
        self.assertIn("wikipedia", agent.capabilities)

    def test_autonomy_sandboxed(self):
        from core.agents.base import AgentAutonomy
        self.assertEqual(ResearchAgent.autonomy, AgentAutonomy.SANDBOXED_EXECUTE)

    def test_run_returns_failed_stub(self):
        agent = ResearchAgent()
        task  = _task(type="research", input="Qu'est-ce que Python ?")
        result = asyncio.run(agent.run(task, _ctx(task)))
        self.assertFalse(result.success)
        self.assertIn("Phase 7", result.error)

    def test_plan_research_returns_plan(self):
        agent  = ResearchAgent()
        query  = ResearchQuery(query="Machine learning", max_sources=3)
        plan   = asyncio.run(agent.plan_research(query))
        self.assertIsInstance(plan, ResearchPlan)
        self.assertGreater(len(plan.steps), 0)
        self.assertEqual(plan.estimated_sources, 3)

    def test_research_returns_stub_result(self):
        agent  = ResearchAgent()
        query  = ResearchQuery(query="test")
        result = asyncio.run(agent.research(query))
        self.assertIsInstance(result, ResearchResult)
        self.assertEqual(result.confidence, 0.0)

    def test_is_available_true(self):
        agent = ResearchAgent()
        self.assertTrue(asyncio.run(agent.is_available()))


if __name__ == "__main__":
    unittest.main()

"""Tests Phase 4 — Observabilité (structured logging + metrics).

Structure :
    TestLogEntry           : sérialisation et omission des champs None
    TestRequestContext     : ContextVar set/get
    TestBrainJSONFormatter : format JSON d'un LogRecord
    TestBrainLogger        : injection des champs structurés
    TestMetricsCollector   : enregistrement et snapshot des métriques
    TestGetMetricsSingleton: singleton thread-safe

Usage :
    python -m pytest tests/test_phase4_observability.py -v
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import unittest

from core.observability.log_entry import LogEntry
from core.observability.metrics import MetricsCollector, get_metrics
from core.observability.structured_logger import (
    BrainJSONFormatter,
    BrainLogger,
    get_logger,
    get_request_context,
    set_request_context,
)


# ── TestLogEntry ───────────────────────────────────────────────────────────────

class TestLogEntry(unittest.TestCase):

    def test_required_fields_present(self):
        entry = LogEntry(
            timestamp   = "2026-07-01T12:00:00",
            level       = "INFO",
            logger_name = "test.logger",
            message     = "Hello",
        )
        d = entry.to_dict()
        self.assertEqual(d["timestamp"], "2026-07-01T12:00:00")
        self.assertEqual(d["level"], "INFO")
        self.assertEqual(d["logger"], "test.logger")
        self.assertEqual(d["message"], "Hello")

    def test_none_fields_omitted(self):
        entry = LogEntry(
            timestamp   = "2026-07-01T12:00:00",
            level       = "DEBUG",
            logger_name = "x",
            message     = "test",
        )
        d = entry.to_dict()
        self.assertNotIn("request_id", d)
        self.assertNotIn("user_id", d)
        self.assertNotIn("agent", d)
        self.assertNotIn("error", d)

    def test_optional_fields_included_when_set(self):
        entry = LogEntry(
            timestamp   = "2026-07-01T12:00:00",
            level       = "ERROR",
            logger_name = "brain",
            message     = "fail",
            agent       = "synthesizer",
            provider    = "groq",
            duration_ms = 42.5,
            success     = False,
            error       = "timeout",
        )
        d = entry.to_dict()
        self.assertEqual(d["agent"],       "synthesizer")
        self.assertEqual(d["provider"],    "groq")
        self.assertEqual(d["duration_ms"], 42.5)
        self.assertFalse(d["success"])
        self.assertEqual(d["error"],       "timeout")

    def test_extra_dict_included_when_nonempty(self):
        entry = LogEntry(
            timestamp   = "t",
            level       = "INFO",
            logger_name = "x",
            message     = "m",
            extra       = {"tokens": 512},
        )
        d = entry.to_dict()
        self.assertIn("extra", d)
        self.assertEqual(d["extra"]["tokens"], 512)

    def test_extra_dict_omitted_when_empty(self):
        entry = LogEntry(timestamp="t", level="INFO", logger_name="x", message="m")
        d = entry.to_dict()
        self.assertNotIn("extra", d)

    def test_success_false_included(self):
        entry = LogEntry(
            timestamp="t", level="INFO", logger_name="x", message="m",
            success=False,
        )
        d = entry.to_dict()
        self.assertIn("success", d)
        self.assertFalse(d["success"])

    def test_success_true_included(self):
        entry = LogEntry(
            timestamp="t", level="INFO", logger_name="x", message="m",
            success=True,
        )
        d = entry.to_dict()
        self.assertIn("success", d)
        self.assertTrue(d["success"])


# ── TestRequestContext ────────────────────────────────────────────────────────

class TestRequestContext(unittest.TestCase):

    def test_default_context_all_none(self):
        ctx = get_request_context()
        self.assertIsNone(ctx["request_id"])
        self.assertIsNone(ctx["session_id"])
        self.assertIsNone(ctx["user_id"])
        self.assertIsNone(ctx["endpoint"])

    def test_set_and_get_context(self):
        set_request_context(
            request_id = "req-123",
            session_id = "ses-456",
            user_id    = "usr-789",
            endpoint   = "/test",
        )
        ctx = get_request_context()
        self.assertEqual(ctx["request_id"], "req-123")
        self.assertEqual(ctx["session_id"], "ses-456")
        self.assertEqual(ctx["user_id"],    "usr-789")
        self.assertEqual(ctx["endpoint"],   "/test")

    def test_context_is_isolated_between_async_tasks(self):
        results: list[dict] = []

        async def task_a():
            set_request_context(request_id="task-a")
            await asyncio.sleep(0)
            results.append(get_request_context().copy())

        async def task_b():
            set_request_context(request_id="task-b")
            await asyncio.sleep(0)
            results.append(get_request_context().copy())

        async def run():
            await asyncio.gather(task_a(), task_b())

        asyncio.run(run())
        request_ids = {r["request_id"] for r in results}
        self.assertIn("task-a", request_ids)
        self.assertIn("task-b", request_ids)


# ── TestBrainJSONFormatter ────────────────────────────────────────────────────

class TestBrainJSONFormatter(unittest.TestCase):

    def setUp(self):
        self.formatter = BrainJSONFormatter()
        set_request_context(request_id="fmt-test-req", user_id=None,
                            session_id=None, endpoint=None)

    def _make_record(self, msg: str, **extra) -> logging.LogRecord:
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg=msg, args=(), exc_info=None,
        )
        for k, v in extra.items():
            setattr(record, k, v)
        return record

    def test_output_is_valid_json(self):
        record = self._make_record("test message")
        output = self.formatter.format(record)
        parsed = json.loads(output)
        self.assertIsInstance(parsed, dict)

    def test_required_fields_present(self):
        record = self._make_record("hello")
        parsed = json.loads(self.formatter.format(record))
        self.assertIn("timestamp", parsed)
        self.assertIn("level", parsed)
        self.assertIn("logger", parsed)
        self.assertIn("message", parsed)

    def test_request_id_injected_from_context(self):
        record = self._make_record("ctx test")
        parsed = json.loads(self.formatter.format(record))
        self.assertEqual(parsed.get("request_id"), "fmt-test-req")

    def test_agent_field_from_record(self):
        record = self._make_record("agent test")
        record.agent = "synthesizer"
        parsed = json.loads(self.formatter.format(record))
        self.assertEqual(parsed.get("agent"), "synthesizer")

    def test_brain_extra_included(self):
        record = self._make_record("extra test")
        record.brain_extra = {"tokens": 128}
        parsed = json.loads(self.formatter.format(record))
        self.assertEqual(parsed.get("extra", {}).get("tokens"), 128)


# ── TestBrainLogger ───────────────────────────────────────────────────────────

class TestBrainLogger(unittest.TestCase):

    def test_get_logger_returns_brain_logger(self):
        logger = get_logger("makenbrain.test")
        self.assertIsInstance(logger, BrainLogger)

    def test_name_property(self):
        logger = get_logger("makenbrain.test.name")
        self.assertEqual(logger.name, "makenbrain.test.name")

    def test_log_with_agent_kwarg(self):
        """Vérifie que agent=... est accepté sans erreur."""
        logger = get_logger("makenbrain.test.brain_logger")
        # Ne doit pas lever d'exception
        logger.info("test info", agent="synthesizer", duration_ms=12.3)

    def test_log_with_unknown_kwarg_goes_to_extra(self):
        """Vérifie que les kwargs inconnus sont regroupés dans brain_extra."""
        captured: list[logging.LogRecord] = []

        class CapturingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record)

        handler = CapturingHandler()
        internal_logger = logging.getLogger("makenbrain.test.extra_capture")
        internal_logger.addHandler(handler)
        internal_logger.setLevel(logging.DEBUG)

        logger = BrainLogger("makenbrain.test.extra_capture")
        logger.info("extra test", custom_field="custom_value")

        self.assertTrue(len(captured) > 0)
        record = captured[-1]
        brain_extra = getattr(record, "brain_extra", {})
        self.assertIn("custom_field", brain_extra)
        self.assertEqual(brain_extra["custom_field"], "custom_value")

    def test_all_log_levels_callable(self):
        logger = get_logger("makenbrain.test.levels")
        logger.debug("debug")
        logger.info("info")
        logger.warning("warning")
        logger.error("error")
        logger.critical("critical")

    def test_is_enabled_for(self):
        logger = get_logger("makenbrain.test.enabled")
        # Par défaut le root logger est WARNING, donc DEBUG est désactivé
        result = logger.is_enabled_for(logging.CRITICAL)
        self.assertIsInstance(result, bool)


# ── TestMetricsCollector ──────────────────────────────────────────────────────

class TestMetricsCollector(unittest.TestCase):

    def setUp(self):
        self.metrics = MetricsCollector()

    def test_initial_snapshot_empty(self):
        snap = self.metrics.snapshot()
        self.assertEqual(snap["endpoints"], {})
        self.assertEqual(snap["providers"], {})
        self.assertEqual(snap["pipeline_steps"], {})

    def test_uptime_seconds_positive(self):
        snap = self.metrics.snapshot()
        self.assertGreaterEqual(snap["uptime_seconds"], 0.0)

    def test_record_request_success(self):
        self.metrics.record_request(endpoint="/test", duration_ms=100.0, success=True)
        snap = self.metrics.snapshot()
        ep = snap["endpoints"]["/test"]
        self.assertEqual(ep["total_requests"], 1)
        self.assertEqual(ep["total_errors"],   0)
        self.assertEqual(ep["avg_latency_ms"], 100.0)

    def test_record_request_error(self):
        self.metrics.record_request(endpoint="/fail", duration_ms=50.0, success=False)
        snap = self.metrics.snapshot()
        ep = snap["endpoints"]["/fail"]
        self.assertEqual(ep["total_errors"], 1)
        self.assertGreater(ep["error_rate"], 0)

    def test_error_rate_calculation(self):
        self.metrics.record_request(endpoint="/mixed", duration_ms=100.0, success=True)
        self.metrics.record_request(endpoint="/mixed", duration_ms=100.0, success=False)
        snap = self.metrics.snapshot()
        ep = snap["endpoints"]["/mixed"]
        self.assertAlmostEqual(ep["error_rate"], 0.5)

    def test_avg_latency_multiple_requests(self):
        self.metrics.record_request(endpoint="/latency", duration_ms=100.0, success=True)
        self.metrics.record_request(endpoint="/latency", duration_ms=200.0, success=True)
        snap = self.metrics.snapshot()
        ep = snap["endpoints"]["/latency"]
        self.assertAlmostEqual(ep["avg_latency_ms"], 150.0)

    def test_record_llm_call_success(self):
        self.metrics.record_llm_call(provider="groq", model="llama-3.3-70b", success=True, tokens=512)
        snap = self.metrics.snapshot()
        prov = snap["providers"]["groq"]
        self.assertEqual(prov["total_calls"],   1)
        self.assertEqual(prov["success_calls"], 1)
        self.assertEqual(prov["failed_calls"],  0)
        self.assertEqual(prov["total_tokens"],  512)

    def test_record_llm_call_failure(self):
        self.metrics.record_llm_call(provider="ollama", model="llama3.2:3b", success=False)
        snap = self.metrics.snapshot()
        prov = snap["providers"]["ollama"]
        self.assertEqual(prov["failed_calls"], 1)
        self.assertEqual(prov["success_rate"], 0.0)

    def test_record_llm_call_fallback_counted(self):
        self.metrics.record_llm_call(
            provider="ollama", model="llama3.2:3b", success=True, fallback=True
        )
        snap = self.metrics.snapshot()
        self.assertEqual(snap["providers"]["ollama"]["fallbacks"], 1)

    def test_model_stats_tracked_separately(self):
        self.metrics.record_llm_call(provider="groq", model="llama-3.3-70b", success=True)
        snap = self.metrics.snapshot()
        self.assertIn("llama-3.3-70b", snap["models"])

    def test_record_pipeline_step(self):
        self.metrics.record_pipeline_step(step="synthesizer", duration_ms=38.0, success=True)
        snap = self.metrics.snapshot()
        step = snap["pipeline_steps"]["synthesizer"]
        self.assertEqual(step["total_runs"],    1)
        self.assertEqual(step["total_errors"],  0)
        self.assertEqual(step["avg_latency_ms"], 38.0)

    def test_record_pipeline_step_error(self):
        self.metrics.record_pipeline_step(step="decision_engine", duration_ms=10.0, success=False)
        snap = self.metrics.snapshot()
        self.assertEqual(snap["pipeline_steps"]["decision_engine"]["total_errors"], 1)

    def test_reset_clears_all_metrics(self):
        self.metrics.record_request(endpoint="/x", duration_ms=10.0, success=True)
        self.metrics.record_llm_call(provider="groq", model="m", success=True)
        self.metrics.reset()
        snap = self.metrics.snapshot()
        self.assertEqual(snap["endpoints"], {})
        self.assertEqual(snap["providers"], {})

    def test_thread_safety(self):
        """Enregistrement concurrent depuis 50 threads — aucune exception."""
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(20):
                    self.metrics.record_request(
                        endpoint="/concurrent", duration_ms=10.0, success=True
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        snap = self.metrics.snapshot()
        self.assertEqual(snap["endpoints"]["/concurrent"]["total_requests"], 50 * 20)


# ── TestGetMetricsSingleton ───────────────────────────────────────────────────

class TestGetMetricsSingleton(unittest.TestCase):

    def test_same_instance_returned(self):
        m1 = get_metrics()
        m2 = get_metrics()
        self.assertIs(m1, m2)

    def test_singleton_thread_safe(self):
        instances: list[MetricsCollector] = []
        lock = threading.Lock()

        def grab():
            m = get_metrics()
            with lock:
                instances.append(m)

        threads = [threading.Thread(target=grab) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = instances[0]
        self.assertTrue(all(m is first for m in instances))


if __name__ == "__main__":
    unittest.main()

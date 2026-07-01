"""Tests Phase 4 — Provider Layer (base, providers, router).

Strategy d'isolation :
    Aucun test ne contacte Ollama ou l'API Groq réelle.
    Les providers sont instanciés avec des mocks qui remplacent les appels réseau.

Structure :
    TestProviderHealth    : DTO ProviderHealth
    TestLLMProviderBase   : Héritage et interface abstraite
    TestOllamaProvider    : generate() + health() avec mock du client Ollama
    TestGroqProvider      : generate() + health() avec mock du client Groq
    TestLLMRouter         : fallback automatique, ordre des providers, métriques
    TestGetRouterSingleton: singleton thread-safe

Usage :
    python -m pytest tests/test_phase4_provider_layer.py -v
"""
from __future__ import annotations

import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from core.provider_layer.base import LLMProvider, ProviderHealth
from core.provider_layer.groq_provider import GroqProvider
from core.provider_layer.ollama_provider import OllamaProvider
from core.provider_layer.router import LLMRouter, get_router


# ── Helpers ───────────────────────────────────────────────────────────────────

class AlwaysSucceedProvider(LLMProvider):
    """Faux provider qui répond toujours avec succès."""

    @property
    def name(self) -> str:
        return "always_ok"

    @property
    def model(self) -> str:
        return "test-model"

    async def generate(self, prompt, context="", system_prompt=None) -> str:
        return f"Réponse de always_ok : {prompt[:30]}"

    async def health(self) -> ProviderHealth:
        return ProviderHealth(name=self.name, available=True, model=self.model)


class AlwaysFailProvider(LLMProvider):
    """Faux provider qui échoue toujours (retourne None)."""

    @property
    def name(self) -> str:
        return "always_fail"

    @property
    def model(self) -> str:
        return "fail-model"

    async def generate(self, prompt, context="", system_prompt=None) -> None:
        return None

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            name=self.name, available=False, model=self.model,
            error="Toujours hors ligne.",
        )


# ── TestProviderHealth ────────────────────────────────────────────────────────

class TestProviderHealth(unittest.TestCase):

    def test_to_dict_required_fields(self):
        h = ProviderHealth(name="ollama", available=True, model="llama3.2:3b")
        d = h.to_dict()
        self.assertEqual(d["name"],      "ollama")
        self.assertTrue(d["available"])
        self.assertEqual(d["model"],     "llama3.2:3b")

    def test_to_dict_none_latency_omitted(self):
        h = ProviderHealth(name="x", available=False)
        d = h.to_dict()
        self.assertNotIn("latency_ms", d)

    def test_to_dict_latency_included_when_set(self):
        h = ProviderHealth(name="x", available=True, latency_ms=42.5)
        d = h.to_dict()
        self.assertEqual(d["latency_ms"], 42.5)

    def test_to_dict_error_included_when_set(self):
        h = ProviderHealth(name="x", available=False, error="timeout")
        d = h.to_dict()
        self.assertEqual(d["error"], "timeout")

    def test_to_dict_details_included(self):
        h = ProviderHealth(
            name="groq", available=True,
            details={"available_models_count": 10},
        )
        d = h.to_dict()
        self.assertIn("details", d)
        self.assertEqual(d["details"]["available_models_count"], 10)


# ── TestLLMProviderBase ───────────────────────────────────────────────────────

class TestLLMProviderBase(unittest.TestCase):

    def test_cannot_instantiate_abstract(self):
        with self.assertRaises(TypeError):
            LLMProvider()  # type: ignore

    def test_concrete_provider_instantiates(self):
        p = AlwaysSucceedProvider()
        self.assertEqual(p.name, "always_ok")
        self.assertEqual(p.model, "test-model")

    def test_generate_returns_string(self):
        p = AlwaysSucceedProvider()
        result = asyncio.run(p.generate("test prompt"))
        self.assertIsInstance(result, str)

    def test_health_returns_provider_health(self):
        p = AlwaysSucceedProvider()
        h = asyncio.run(p.health())
        self.assertIsInstance(h, ProviderHealth)


# ── TestOllamaProvider ────────────────────────────────────────────────────────

class TestOllamaProvider(unittest.TestCase):

    def _make_provider(self) -> OllamaProvider:
        return OllamaProvider(host="http://localhost:11434", model="llama3.2:3b")

    def test_name_is_ollama(self):
        self.assertEqual(self._make_provider().name, "ollama")

    def test_model_property(self):
        p = OllamaProvider(model="test-model")
        self.assertEqual(p.model, "test-model")

    def test_generate_success_with_mock(self):
        """generate() retourne le contenu du message Ollama."""
        provider = self._make_provider()

        mock_response = MagicMock()
        mock_response.message.content = "Réponse Ollama mockée"

        with patch.object(provider._client, "chat", return_value=mock_response):
            result = asyncio.run(provider.generate("Ma question"))

        self.assertEqual(result, "Réponse Ollama mockée")

    def test_generate_returns_none_on_exception(self):
        """generate() retourne None si Ollama lève une exception."""
        provider = self._make_provider()

        with patch.object(provider._client, "chat", side_effect=ConnectionRefusedError("offline")):
            result = asyncio.run(provider.generate("test"))

        self.assertIsNone(result)

    def test_health_available_when_model_found(self):
        """health() retourne available=True si le modèle est dans la liste."""
        provider = self._make_provider()

        mock_model = MagicMock()
        mock_model.model = "llama3.2:3b"
        mock_list = MagicMock()
        mock_list.models = [mock_model]

        with patch.object(provider._client, "list", return_value=mock_list):
            h = asyncio.run(provider.health())

        self.assertTrue(h.available)
        self.assertEqual(h.name, "ollama")

    def test_health_unavailable_when_model_not_found(self):
        """health() retourne available=False si le modèle n'est pas chargé."""
        provider = self._make_provider()

        mock_model = MagicMock()
        mock_model.model = "autre-modele:7b"
        mock_list = MagicMock()
        mock_list.models = [mock_model]

        with patch.object(provider._client, "list", return_value=mock_list):
            h = asyncio.run(provider.health())

        self.assertFalse(h.available)
        self.assertIsNotNone(h.error)

    def test_health_unavailable_on_connection_error(self):
        """health() retourne available=False si Ollama est hors ligne."""
        provider = self._make_provider()

        with patch.object(provider._client, "list", side_effect=ConnectionRefusedError("offline")):
            h = asyncio.run(provider.health())

        self.assertFalse(h.available)

    def test_generate_with_context(self):
        """generate() injecte le contexte dans les messages."""
        provider = self._make_provider()
        calls: list[dict] = []

        mock_response = MagicMock()
        mock_response.message.content = "Avec mémoire"

        def capture_call(**kwargs):
            calls.append(kwargs)
            return mock_response

        with patch.object(provider._client, "chat", side_effect=capture_call):
            asyncio.run(provider.generate("question", context="mémoire"))

        self.assertTrue(len(calls) > 0)
        messages = calls[0]["messages"]
        # Le premier message non-système doit contenir le contexte
        user_msgs = [m for m in messages if m["role"] == "user"]
        self.assertTrue(any("mémoire" in m["content"] for m in user_msgs))


# ── TestGroqProvider ──────────────────────────────────────────────────────────

class TestGroqProvider(unittest.TestCase):

    def test_name_is_groq(self):
        self.assertEqual(GroqProvider().name, "groq")

    def test_model_default(self):
        p = GroqProvider()
        self.assertEqual(p.model, "llama-3.3-70b-versatile")

    def test_generate_returns_none_when_no_api_key(self):
        """generate() retourne None si GROQ_API_KEY est vide."""
        with patch("core.provider_layer.groq_provider.settings") as mock_settings:
            mock_settings.GROQ_API_KEY = ""
            result = asyncio.run(GroqProvider().generate("test"))
        self.assertIsNone(result)

    def test_generate_success_with_mock(self):
        """generate() retourne le contenu de la réponse Groq."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Réponse Groq mockée"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("core.provider_layer.groq_provider.settings") as mock_settings:
            mock_settings.GROQ_API_KEY = "gsk_test_key"
            with patch("core.provider_layer.groq_provider.GroqProvider._get_client",
                       return_value=mock_client):
                result = asyncio.run(GroqProvider().generate("Ma question"))

        self.assertEqual(result, "Réponse Groq mockée")

    def test_generate_returns_none_on_exception(self):
        """generate() retourne None si l'API Groq lève une exception."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        with patch("core.provider_layer.groq_provider.settings") as mock_settings:
            mock_settings.GROQ_API_KEY = "gsk_test"
            with patch("core.provider_layer.groq_provider.GroqProvider._get_client",
                       return_value=mock_client):
                result = asyncio.run(GroqProvider().generate("test"))

        self.assertIsNone(result)

    def test_health_unavailable_when_no_api_key(self):
        with patch("core.provider_layer.groq_provider.settings") as mock_settings:
            mock_settings.GROQ_API_KEY = ""
            h = asyncio.run(GroqProvider().health())
        self.assertFalse(h.available)
        self.assertIn("GROQ_API_KEY", h.error)

    def test_health_available_with_valid_key(self):
        mock_model    = MagicMock()
        mock_model.id = "llama-3.3-70b-versatile"
        mock_list     = MagicMock()
        mock_list.data = [mock_model]

        mock_client = MagicMock()
        mock_client.models.list.return_value = mock_list

        with patch("core.provider_layer.groq_provider.settings") as mock_settings:
            mock_settings.GROQ_API_KEY = "gsk_test"
            with patch("core.provider_layer.groq_provider.GroqProvider._get_client",
                       return_value=mock_client):
                h = asyncio.run(GroqProvider().health())

        self.assertTrue(h.available)
        self.assertEqual(h.name, "groq")


# ── TestLLMRouter ─────────────────────────────────────────────────────────────

class TestLLMRouter(unittest.TestCase):

    def _router_with(self, *providers: LLMProvider) -> LLMRouter:
        return LLMRouter(providers=list(providers))

    def test_first_provider_used_when_available(self):
        ok   = AlwaysSucceedProvider()
        fail = AlwaysFailProvider()
        router = self._router_with(ok, fail)
        result = asyncio.run(router.generate("test"))
        self.assertIn("always_ok", result)

    def test_fallback_to_second_provider(self):
        fail = AlwaysFailProvider()
        ok   = AlwaysSucceedProvider()
        router = self._router_with(fail, ok)
        result = asyncio.run(router.generate("test"))
        self.assertIn("always_ok", result)

    def test_all_providers_fail_returns_error_message(self):
        fail1 = AlwaysFailProvider()
        fail2 = AlwaysFailProvider()
        router = self._router_with(fail1, fail2)
        result = asyncio.run(router.generate("test"))
        self.assertIn("aucun provider", result.lower())

    def test_preferred_provider_tried_first(self):
        ok_first  = AlwaysSucceedProvider()
        ok_second = AlwaysSucceedProvider()

        # On remplace les noms pour simuler deux providers distincts
        ok_first._name  = "first"
        ok_second._name = "second"

        class FirstProvider(LLMProvider):
            @property
            def name(self): return "first"
            @property
            def model(self): return "m"
            async def generate(self, prompt, context="", system_prompt=None): return "first response"
            async def health(self): return ProviderHealth("first", True)

        class SecondProvider(LLMProvider):
            @property
            def name(self): return "second"
            @property
            def model(self): return "m"
            async def generate(self, prompt, context="", system_prompt=None): return "second response"
            async def health(self): return ProviderHealth("second", True)

        router = self._router_with(FirstProvider(), SecondProvider())
        result = asyncio.run(router.generate("test", preferred="second"))
        self.assertEqual(result, "second response")

    def test_health_all_returns_list(self):
        ok   = AlwaysSucceedProvider()
        fail = AlwaysFailProvider()
        router = self._router_with(ok, fail)
        healths = asyncio.run(router.health_all())
        self.assertEqual(len(healths), 2)
        self.assertIsInstance(healths[0], ProviderHealth)

    def test_health_all_reflects_provider_status(self):
        ok   = AlwaysSucceedProvider()
        fail = AlwaysFailProvider()
        router = self._router_with(ok, fail)
        healths = asyncio.run(router.health_all())
        names = {h.name: h.available for h in healths}
        self.assertTrue(names["always_ok"])
        self.assertFalse(names["always_fail"])

    def test_providers_property(self):
        ok   = AlwaysSucceedProvider()
        fail = AlwaysFailProvider()
        router = self._router_with(ok, fail)
        self.assertEqual(len(router.providers), 2)

    def test_metrics_recorded_on_success(self):
        from core.observability import get_metrics
        get_metrics().reset()

        ok = AlwaysSucceedProvider()
        router = self._router_with(ok)
        asyncio.run(router.generate("metrics test"))

        snap = get_metrics().snapshot()
        self.assertIn("always_ok", snap["providers"])
        self.assertEqual(snap["providers"]["always_ok"]["success_calls"], 1)

    def test_metrics_recorded_on_fallback(self):
        from core.observability import get_metrics
        get_metrics().reset()

        fail = AlwaysFailProvider()
        ok   = AlwaysSucceedProvider()
        router = self._router_with(fail, ok)
        asyncio.run(router.generate("fallback test"))

        snap = get_metrics().snapshot()
        # Le fallback=True est enregistré sur le provider qui a réussi (second)
        self.assertEqual(snap["providers"]["always_ok"]["fallbacks"], 1)


# ── TestGetRouterSingleton ────────────────────────────────────────────────────

class TestGetRouterSingleton(unittest.TestCase):

    def test_same_instance_returned(self):
        r1 = get_router()
        r2 = get_router()
        self.assertIs(r1, r2)

    def test_router_has_providers(self):
        router = get_router()
        self.assertTrue(len(router.providers) > 0)

    def test_singleton_thread_safe(self):
        instances: list[LLMRouter] = []
        lock = threading.Lock()

        def grab():
            r = get_router()
            with lock:
                instances.append(r)

        threads = [threading.Thread(target=grab) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = instances[0]
        self.assertTrue(all(r is first for r in instances))


if __name__ == "__main__":
    unittest.main()

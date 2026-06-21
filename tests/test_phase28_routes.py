"""Tests des champs API exposes par la Phase 2.8."""
from __future__ import annotations

import asyncio

from routers.chat import ChatResponse


def test_chat_response_reste_retrocompatible_avec_anciens_champs() -> None:
    """Les nouveaux champs doivent avoir des valeurs par defaut non cassantes."""
    response = ChatResponse(
        response="Reponse existante",
        memories_used=0,
        graph_concepts=[],
        model="local",
        provider="local",
    )

    assert response.response == "Reponse existante"
    assert response.confidence == 0.0
    assert response.risk_level == "high"
    assert response.suggested_sources == []


def test_chat_response_sources_ne_partage_pas_de_liste_mutable() -> None:
    """Chaque reponse doit posseder sa propre liste de sources suggerees."""
    first = ChatResponse(
        response="A",
        memories_used=0,
        graph_concepts=[],
        model="local",
        provider="local",
    )
    second = ChatResponse(
        response="B",
        memories_used=0,
        graph_concepts=[],
        model="local",
        provider="local",
    )

    first.suggested_sources.append({"type": "web", "query": "test"})

    assert second.suggested_sources == []


def test_brain_expert_chat_expose_les_nouveaux_champs(monkeypatch) -> None:
    """L'endpoint expert doit enrichir sa reponse sans changer les champs historiques."""
    from routers import brain
    import core.extractor
    import core.graph
    import core.memory
    import core.providers

    async def fake_search_memory(_query: str, n_results: int = 8) -> list[dict]:
        """Retourne une memoire factice sans acceder a ChromaDB."""
        return [
            {
                "content": "MakenBrain utilise une memoire personnelle.",
                "distance": 0.2,
                "metadata": {"tags": "ia"},
            }
        ]

    async def fake_groq_generate(*_args, **_kwargs) -> str:
        """Retourne une reponse factice sans appel reseau."""
        return "Reponse experte basee sur le contexte disponible."

    class FakeGraph:
        """Graphe minimal pour isoler le test de la logique exposee."""

        def explore(self, _name: str, depth: int = 2) -> dict:
            return {"found": False, "neighbors": {}}

    monkeypatch.setattr(core.memory, "search_memory", fake_search_memory)
    monkeypatch.setattr(core.providers, "groq_generate", fake_groq_generate)
    monkeypatch.setattr(core.extractor, "extract_fast", lambda _question: [])
    monkeypatch.setattr(core.graph, "neuron_graph", FakeGraph())

    result = asyncio.run(
        brain.expert_chat(
            brain.ExpertChatRequest(question="Explique MakenBrain", domain="ia")
        )
    )

    assert "response" in result
    assert "sources_used" in result
    assert "confidence" in result
    assert "risk_level" in result
    assert "suggested_sources" in result

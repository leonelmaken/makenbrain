"""Tests unitaires du routage vers sources externes Phase 2.8."""
from __future__ import annotations

from core.source_router import (
    is_complex_request,
    should_suggest_sources,
    suggest_sources,
)


def test_sources_suggerees_quand_confiance_faible() -> None:
    """Une confiance basse doit produire des pistes de verification externes."""
    sources = suggest_sources(
        question="Explique comment corriger cette erreur FastAPI",
        confidence=0.45,
        critic_issues=[],
    )

    assert [source["type"] for source in sources][:2] == ["web", "documentation_officielle"]
    assert any(source["type"] == "youtube" for source in sources)


def test_sources_suggerees_quand_incertitude_detectee() -> None:
    """Une incertitude critique doit declencher le routage meme avec confiance moyenne."""
    assert should_suggest_sources(
        question="Quelle approche choisir ?",
        confidence=0.68,
        critic_issues=["uncertainty"],
    )


def test_demande_complexe_declenche_sources_sans_baisser_la_confiance() -> None:
    """Une demande complexe doit rester verifiable meme avec un bon score."""
    question = "Propose une architecture production pour un assistant IA avec memoire, auth et observabilite"

    assert is_complex_request(question)
    assert should_suggest_sources(question=question, confidence=0.81, critic_issues=[])


def test_sources_x_suggerees_pour_tendances_recentes() -> None:
    """Les sujets dependants de tendances recentes doivent proposer X en verification."""
    sources = suggest_sources(
        question="Quelles sont les tendances IA actuelles en 2026 ?",
        confidence=0.58,
        critic_issues=[],
    )

    assert any(source["type"] == "x" for source in sources)

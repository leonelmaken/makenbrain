"""Tests de validation de la couche consciousness Phase 2.8."""
from __future__ import annotations

from core.consciousness import BrainConsciousness


def test_evaluation_sans_memoire_expose_confiance_faible_et_sources() -> None:
    """Sans memoire utilisateur, la reponse doit rester prudente et verifiable."""
    consciousness = BrainConsciousness()

    result = consciousness.evaluate_response(
        question="Quelle strategie choisir pour un sujet inconnu ?",
        answer="Cette strategie est la meilleure.",
        user_memory_count=0,
        supabase_data_count=0,
        vector_memory_count=0,
        graph_context_count=0,
    )

    assert result["confidence"] < 0.60
    assert result["risk_level"] == "high"
    assert result["suggested_sources"]
    assert result["answer"].startswith("Avec les donnees actuellement disponibles")


def test_evaluation_avec_memoire_utilisateur_augmente_la_confiance() -> None:
    """La presence de memoire utilisateur doit renforcer le score de confiance."""
    consciousness = BrainConsciousness()

    result = consciousness.evaluate_response(
        question="Resume mon projet principal.",
        answer="Ton projet principal est MakenBrain.",
        user_memory_count=4,
        supabase_data_count=2,
        vector_memory_count=2,
        graph_context_count=1,
    )

    assert result["confidence"] >= 0.75
    assert result["risk_level"] == "low"
    assert "suggested_sources" in result

"""Tests unitaires du calcul de confiance des reponses Phase 2.8."""
from __future__ import annotations

from core.response_confidence import (
    build_confidence_payload,
    calculate_confidence,
    risk_level,
)


def test_confidence_faible_sans_donnees_contextuelles() -> None:
    """Une reponse sans memoire ni donnees internes doit rester risquee."""
    confidence = calculate_confidence(
        answer="Reponse plausible mais non ancree dans une source interne.",
        user_memory_count=0,
        supabase_data_count=0,
        vector_memory_count=0,
        graph_context_count=0,
    )

    assert confidence < 0.60
    assert risk_level(confidence) == "high"


def test_confidence_moyenne_avec_contexte_interne_limite() -> None:
    """Un contexte vectoriel et graphe limite doit produire un risque moyen."""
    confidence = calculate_confidence(
        answer="Reponse appuyee sur quelques elements internes disponibles.",
        vector_memory_count=4,
        graph_context_count=3,
    )

    assert 0.60 <= confidence < 0.75
    assert risk_level(confidence) == "medium"


def test_confidence_elevee_avec_memoire_utilisateur_et_supabase() -> None:
    """La memoire utilisateur et Supabase doivent fortement renforcer la confiance."""
    confidence = calculate_confidence(
        answer="Reponse ancree dans la memoire utilisateur et les donnees internes.",
        user_memory_count=3,
        supabase_data_count=2,
        vector_memory_count=2,
        graph_context_count=2,
    )

    assert confidence >= 0.75
    assert risk_level(confidence) == "low"


def test_incertitude_reduit_le_score_de_confiance() -> None:
    """Un marqueur d'incertitude doit abaisser le score final."""
    base = calculate_confidence(
        answer="Reponse appuyee sur des donnees internes.",
        user_memory_count=2,
        supabase_data_count=1,
    )
    uncertain = calculate_confidence(
        answer="Reponse appuyee sur des donnees internes.",
        user_memory_count=2,
        supabase_data_count=1,
        uncertainty_detected=True,
    )

    assert uncertain < base


def test_payload_retrocompatible_contient_les_champs_attendus() -> None:
    """Le payload de confiance doit conserver un format dictionnaire stable."""
    payload = build_confidence_payload("Reponse finale", 0.81)

    assert payload == {
        "answer": "Reponse finale",
        "confidence": 0.81,
        "risk_level": "low",
    }

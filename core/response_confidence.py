"""Evaluation de confiance pour les reponses de MakenBrain.

Ce module fournit une heuristique deterministe et sans effet de bord pour
estimer la fiabilite d'une reponse. Il reste volontairement dans ``core``
car il encode une logique applicative, pas un DTO ni une route API.
"""
from __future__ import annotations

from typing import Any


LOW_RISK_THRESHOLD = 0.75
MEDIUM_RISK_THRESHOLD = 0.60


def calculate_confidence(
    *,
    answer: str,
    user_memory_count: int = 0,
    supabase_data_count: int = 0,
    vector_memory_count: int = 0,
    graph_context_count: int = 0,
    internal_logic_used: bool = True,
    uncertainty_detected: bool = False,
) -> float:
    """Calcule un score de confiance normalise entre 0.0 et 1.0.

    La memoire utilisateur et les donnees Supabase augmentent fortement
    la confiance car elles ancrent la reponse dans le contexte personnel
    disponible. La logique interne du modele est consideree comme neutre:
    utile pour formuler, mais insuffisante comme preuve. L'absence de
    donnees et les marqueurs d'incertitude abaissent le score.
    """
    score = 0.42

    if user_memory_count > 0:
        score += min(0.28, 0.12 + (user_memory_count * 0.04))
    if supabase_data_count > 0:
        score += min(0.18, 0.08 + (supabase_data_count * 0.03))
    if vector_memory_count > 0:
        score += min(0.12, vector_memory_count * 0.03)
    if graph_context_count > 0:
        score += min(0.08, graph_context_count * 0.02)
    if internal_logic_used:
        score += 0.02

    has_no_grounding = (
        user_memory_count == 0
        and supabase_data_count == 0
        and vector_memory_count == 0
        and graph_context_count == 0
    )
    if has_no_grounding:
        score -= 0.20
    if uncertainty_detected:
        score -= 0.18
    if len(answer.strip()) < 20:
        score -= 0.10

    return max(0.0, min(1.0, round(score, 2)))


def risk_level(confidence: float) -> str:
    """Convertit un score de confiance en niveau de risque lisible."""
    if confidence >= LOW_RISK_THRESHOLD:
        return "low"
    if confidence >= MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "high"


def build_confidence_payload(answer: str, confidence: float) -> dict[str, Any]:
    """Construit le fragment standard expose aux routes API."""
    return {
        "answer": answer,
        "confidence": confidence,
        "risk_level": risk_level(confidence),
    }

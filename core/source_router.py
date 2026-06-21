"""Routage vers sources externes lorsque la reponse est incertaine.

Ce module ne lance aucune recherche reseau. Il propose des pistes de
verification sous forme de requetes structurees afin que l'API puisse
rediriger l'utilisateur vers les bons supports sans inventer de sources.
"""
from __future__ import annotations

from typing import Any


COMPLEXITY_MARKERS = (
    "architecture",
    "compare",
    "comparaison",
    "strategie",
    "plan",
    "production",
    "securite",
    "debug",
    "erreur",
    "meilleure",
    "tendance",
    "latest",
    "recent",
    "actuel",
)


def should_suggest_sources(
    *,
    question: str,
    confidence: float,
    critic_issues: list[str] | None = None,
) -> bool:
    """Determine si des sources externes doivent etre proposees."""
    issues = critic_issues or []
    return (
        confidence < 0.60
        or "uncertainty" in issues
        or "missing_sources" in issues
        or is_complex_request(question)
    )


def suggest_sources(
    *,
    question: str,
    confidence: float,
    critic_issues: list[str] | None = None,
) -> list[dict[str, str]]:
    """Construit une liste de sources externes recommandees.

    Les suggestions restent declaratives: elles indiquent quoi verifier
    et pourquoi, sans pretendre qu'une recherche a deja ete effectuee.
    """
    if not should_suggest_sources(
        question=question,
        confidence=confidence,
        critic_issues=critic_issues,
    ):
        return []

    reason = _reason_from_state(confidence, critic_issues or [])
    query = _clean_query(question)
    suggestions: list[dict[str, str]] = [
        {
            "type": "web",
            "query": query,
            "reason": reason,
        },
        {
            "type": "documentation_officielle",
            "query": f"{query} documentation officielle",
            "reason": "verification prioritaire aupres de la source de reference",
        },
    ]

    if is_learning_or_debug_request(question):
        suggestions.append(
            {
                "type": "youtube",
                "query": f"{query} tutoriel",
                "reason": "support pedagogique utile pour completer l'explication",
            }
        )

    if is_trend_request(question):
        suggestions.append(
            {
                "type": "x",
                "query": query,
                "reason": "signal faible utile pour tendances et retours terrain recents",
            }
        )

    return suggestions


def is_complex_request(question: str) -> bool:
    """Repere les demandes qui meritent une verification externe."""
    normalized = question.lower()
    return len(normalized.split()) > 18 or any(marker in normalized for marker in COMPLEXITY_MARKERS)


def is_learning_or_debug_request(question: str) -> bool:
    """Detecte les demandes ou un tutoriel peut aider l'utilisateur."""
    normalized = question.lower()
    return any(marker in normalized for marker in ("comment", "explique", "debug", "erreur", "implementer", "tutoriel"))


def is_trend_request(question: str) -> bool:
    """Detecte les questions dependantes de tendances recentes."""
    normalized = question.lower()
    return any(marker in normalized for marker in ("tendance", "latest", "recent", "actuel", "2025", "2026"))


def build_source_payload(answer: str, confidence: float, suggested_sources: list[dict[str, str]]) -> dict[str, Any]:
    """Assemble le format attendu pour une reponse avec sources suggerees."""
    return {
        "answer": answer,
        "confidence": confidence,
        "suggested_sources": suggested_sources,
    }


def _clean_query(question: str) -> str:
    """Transforme la question utilisateur en requete de recherche concise."""
    query = " ".join(question.strip().split())
    return query[:160]


def _reason_from_state(confidence: float, critic_issues: list[str]) -> str:
    """Explique pourquoi une verification externe est recommandee."""
    if confidence < 0.60:
        return "confiance insuffisante pour reponse categorique"
    if "missing_sources" in critic_issues:
        return "sources externes manquantes"
    if "uncertainty" in critic_issues:
        return "incertitude detectee dans la reponse"
    return "demande complexe necessitant validation"

"""Auto-critique deterministe des reponses generees par MakenBrain.

Le role de ce module est de relire une reponse avant exposition API afin
de detecter les signaux simples de fragilite: contradiction apparente,
incertitude explicite ou manque de sources. Il ne remplace pas le modele
LLM; il ajoute une garde applicative stable et previsible.
"""
from __future__ import annotations

from typing import Any


UNCERTAINTY_MARKERS = (
    "je ne sais pas",
    "je ne suis pas sur",
    "je ne suis pas certain",
    "peut-etre",
    "probablement",
    "a verifier",
    "il semble",
    "hypothese",
    "je suppose",
)


def analyze_answer(
    answer: str,
    *,
    context_available: bool,
    confidence: float,
) -> dict[str, Any]:
    """Analyse une reponse et retourne une version validee ou corrigee.

    Parametres:
        answer: Texte brut genere par le provider LLM.
        context_available: Indique si la reponse dispose de memoire ou
            de donnees internes exploitables.
        confidence: Score calcule par ``response_confidence``.
    Retour:
        Dictionnaire contenant la reponse finale, les alertes detectees
        et un booleen indiquant si une correction a ete appliquee.
    """
    original = answer.strip()
    issues = _detect_issues(original, context_available)
    corrected = original

    # Confiance basse : le LLM exprime lui-même l'incertitude via le prompt système.
    # On n'ajoute aucun préambule académique automatique.

    # Sources manquantes : on laisse la réponse telle quelle.
    # Le score de confiance bas est déjà visible dans les métadonnées.

    return {
        "answer": corrected,
        "issues": issues,
        "corrected": corrected != original,
    }


def has_uncertainty(answer: str) -> bool:
    """Detecte les marqueurs textuels d'incertitude dans une reponse."""
    normalized = answer.lower()
    return any(marker in normalized for marker in UNCERTAINTY_MARKERS)


def _detect_issues(answer: str, context_available: bool) -> list[str]:
    """Retourne les problemes simples detectables sans appel LLM."""
    issues: list[str] = []
    normalized = answer.lower()

    if has_uncertainty(answer):
        issues.append("uncertainty")
    if _has_apparent_contradiction(normalized):
        issues.append("possible_contradiction")
    if not context_available and not _mentions_source_or_limit(normalized):
        issues.append("missing_sources")

    return issues


def _has_apparent_contradiction(normalized_answer: str) -> bool:
    """Repere quelques formulations contradictoires frequentes."""
    contradiction_pairs = (
        ("toujours", "jamais"),
        ("certain", "incertain"),
        ("oui", "non"),
        ("possible", "impossible"),
    )
    return any(left in normalized_answer and right in normalized_answer for left, right in contradiction_pairs)


def _mentions_source_or_limit(normalized_answer: str) -> bool:
    """Indique si la reponse expose deja ses limites ou ses sources."""
    return any(
        marker in normalized_answer
        for marker in (
            "source",
            "documentation",
            "memoire",
            "donnees disponibles",
            "a verifier",
            "je ne dispose pas",
        )
    )


def _starts_with_uncertainty_guard(answer: str) -> bool:
    """Evite de dupliquer le preambule d'incertitude."""
    normalized = answer.lower().lstrip()
    return normalized.startswith("avec les donnees actuellement disponibles")

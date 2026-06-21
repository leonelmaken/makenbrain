"""Tests unitaires du systeme d'auto-critique Phase 2.8."""
from __future__ import annotations

from core.self_critic import analyze_answer, has_uncertainty


def test_reponse_incertaine_est_detectee() -> None:
    """Les marqueurs d'incertitude doivent etre reconnus explicitement."""
    assert has_uncertainty("Il semble que cette piste soit probablement correcte.")


def test_confiance_faible_force_une_formulation_non_categorique() -> None:
    """Une faible confiance doit transformer la reponse en hypothese."""
    result = analyze_answer(
        "Cette architecture est la meilleure solution.",
        context_available=False,
        confidence=0.42,
    )

    assert result["corrected"] is True
    assert result["answer"].startswith("Avec les donnees actuellement disponibles")
    assert "missing_sources" in result["issues"]


def test_reponse_contradictoire_est_signalee() -> None:
    """Les contradictions lexicales simples doivent etre exposees aux appelants."""
    result = analyze_answer(
        "Oui, cette approche est possible, mais elle est aussi impossible.",
        context_available=True,
        confidence=0.70,
    )

    assert "possible_contradiction" in result["issues"]


def test_contexte_present_evite_le_signal_missing_sources() -> None:
    """Une reponse ancree dans le contexte interne ne doit pas demander de sources par defaut."""
    result = analyze_answer(
        "La memoire utilisateur indique que ce projet cible la fintech.",
        context_available=True,
        confidence=0.82,
    )

    assert result["corrected"] is False
    assert "missing_sources" not in result["issues"]

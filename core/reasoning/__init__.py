"""Sous-package du moteur de raisonnement expert de MakenBrain.

Phase 3.0 — Expert Reasoning Engine.

Exports publics :
- ReasoningAgent : protocole que chaque étape du pipeline doit implémenter.

Le protocole ReasoningAgent est le contrat multi-agents. Un orchestrateur
futur peut câbler des implémentations différentes sans modifier le code
des étapes existantes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from core.reasoning.context import ReasoningContext


@runtime_checkable
class ReasoningAgent(Protocol):
    """Interface que chaque étape du pipeline de raisonnement doit implémenter.

    Contrat :
    - run() reçoit le ReasoningContext courant.
    - run() retourne le ReasoningContext enrichi par cette étape.
    - run() ne modifie jamais les champs écrits par une étape précédente.
    - En cas d'erreur, run() appelle ctx.mark_degraded() et retourne ctx.

    Un orchestrateur multi-agents (Phase 4+) peut substituer n'importe quelle
    implémentation de ce protocole sans modifier le reasoning_engine.
    """

    @property
    def name(self) -> str:
        """Nom de l'étape — utilisé dans la ReasoningTrace et les logs."""
        ...  # pragma: no cover

    async def run(self, ctx: "ReasoningContext") -> "ReasoningContext":
        """Exécute l'étape et retourne le contexte enrichi.

        Args:
            ctx : Shared state courant du pipeline.

        Returns:
            ctx enrichi par les résultats de cette étape.
        """
        ...  # pragma: no cover


__all__ = ["ReasoningAgent"]

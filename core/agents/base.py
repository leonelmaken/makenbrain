"""BaseAgent — classe abstraite commune à tous les agents MakenBrain — Phase 5.

Tout futur agent DOIT hériter de BaseAgent et implémenter `run()`.
Les champs de classe définissent l'identité, les capacités et les contraintes
de l'agent — ils sont lus par AgentRegistry et TaskRouter pour le routage
automatique sans instanciation préalable.

Niveaux d'autonomie (du plus restrictif au plus large) :
    READ_ONLY        : lecture seule, aucune action.
    SUGGEST          : propose des actions, l'humain décide.
    SANDBOXED_EXECUTE: exécute dans un environnement isolé.
    AUTONOMOUS       : exécution directe sans confirmation.

Voir docs/ARCHITECTURE_VISION.md § Niveaux d'autonomie pour la politique
de gouvernance associée à chaque niveau.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from core.agents.models import AgentResult, AgentTask, ExecutionContext


class AgentAutonomy(StrEnum):
    """Niveau d'autonomie d'un agent — détermine qui valide ses actions."""
    READ_ONLY         = "read_only"
    SUGGEST           = "suggest"
    SANDBOXED_EXECUTE = "sandboxed_execute"
    AUTONOMOUS        = "autonomous"


class BaseAgent(ABC):
    """Classe de base abstraite pour tous les agents MakenBrain.

    Champs de classe à surcharger dans chaque agent concret :
        name                : Identifiant unique de l'agent ("reasoning_agent").
        description         : Description lisible des capacités de l'agent.
        capabilities        : Liste de mots-clés utilisés par TaskRouter
                              pour matcher les tâches (ex. ["reasoning", "qa"]).
        autonomy            : Niveau d'autonomie (AgentAutonomy.*).
        version             : Numéro de version sémantique de l'agent.
        cost_per_call       : Coût estimé en unités abstraites par appel.
                              Utilisé par TaskRouter pour optimiser le coût.
        confidence_threshold: Score minimum de confiance attendu en sortie.
        compatible_models   : Liste de modèles LLM préférés par cet agent.
                              Vide = compatible avec tous les providers.
        max_concurrent      : Nombre max d'exécutions parallèles (Phase 5+).
    """

    # ── Identité (à surcharger) ────────────────────────────────────────────────
    name                : str              = "base_agent"
    description         : str              = "Agent de base abstrait"
    capabilities        : list[str]        = []
    autonomy            : AgentAutonomy    = AgentAutonomy.READ_ONLY
    version             : str              = "1.0.0"

    # ── Paramètres de routage (à surcharger selon l'agent) ────────────────────
    cost_per_call       : float            = 0.0
    confidence_threshold: float            = 0.5
    compatible_models   : list[str]        = []
    max_concurrent      : int              = 1

    # ── Interface obligatoire ─────────────────────────────────────────────────

    @abstractmethod
    async def run(self, task: AgentTask, ctx: ExecutionContext) -> AgentResult:
        """Exécute la tâche et retourne un AgentResult.

        L'implémentation doit :
        - Ne jamais lever d'exception (capturer et retourner success=False).
        - Toujours remplir duration_ms.
        - Ne jamais appeler directement un autre agent.

        Args:
            task : La tâche à traiter.
            ctx  : Contexte d'exécution partagé (lecture + écriture de shared).

        Returns:
            AgentResult avec success=True/False et output/error renseignés.
        """

    # ── Interface optionnelle (override selon les besoins) ────────────────────

    async def is_available(self) -> bool:
        """Retourne True si l'agent peut accepter de nouvelles tâches.

        Override pour intégrer une logique de health check spécifique
        (ex. vérifier qu'Ollama est disponible pour ReasoningAgent).
        """
        return True

    async def health(self) -> dict[str, Any]:
        """Retourne le statut de santé de l'agent.

        Returns:
            Dict avec au minimum {"name": ..., "available": ...}.
        """
        return {
            "name"        : self.name,
            "version"     : self.version,
            "autonomy"    : self.autonomy,
            "capabilities": self.capabilities,
            "available"   : await self.is_available(),
        }

    # ── Helpers pour les implémentations concrètes ────────────────────────────

    def _make_result(
        self,
        task       : AgentTask,
        success    : bool,
        output     : str | None       = None,
        error      : str | None       = None,
        duration_ms: float            = 0.0,
        confidence : float | None     = None,
        metadata   : dict[str, Any]   | None = None,
    ) -> AgentResult:
        """Raccourci pour construire un AgentResult correctement typé."""
        return AgentResult(
            task_id    = task.task_id,
            agent_name = self.name,
            success    = success,
            output     = output,
            error      = error,
            duration_ms= duration_ms,
            confidence = confidence,
            metadata   = metadata or {},
        )

    def _timed_result(
        self,
        task      : AgentTask,
        start_time: float,
        **kwargs  : Any,
    ) -> AgentResult:
        """Construit un AgentResult en calculant duration_ms depuis start_time.

        Args:
            start_time : Valeur de time.monotonic() au début de l'exécution.
            **kwargs   : Mêmes kwargs que _make_result (sauf duration_ms).
        """
        duration_ms = round((time.monotonic() - start_time) * 1000, 1)
        return self._make_result(task, duration_ms=duration_ms, **kwargs)

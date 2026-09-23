"""AgentRegistry — registre central des agents disponibles — Phase 5.

L'Orchestrator ne doit jamais instancier directement un agent.
Il passe toujours par le Registry pour :
    - Découvrir les agents disponibles.
    - Obtenir les agents capables d'une certaine tâche.
    - Vérifier leur disponibilité.

Design :
    - Singleton global accessible via get_registry().
    - Thread-safe (RLock) : les enregistrements peuvent venir du lifespan
      FastAPI dans un thread différent de la boucle asyncio.
    - Les agents sont enregistrés à l'initialisation de l'application
      (dans le lifespan ou en appelant register_defaults()).
"""
from __future__ import annotations

import threading
from typing import Any

from core.agents.base import BaseAgent


class AgentRegistry:
    """Registre central de tous les agents MakenBrain.

    Usage :
        registry = get_registry()
        registry.register(ReasoningAgent())
        agent = registry.get("reasoning_agent")
        capable = registry.by_capability("reasoning")
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._lock   = threading.RLock()

    def register(self, agent: BaseAgent) -> None:
        """Enregistre un agent. Remplace silencieusement si le nom existe déjà.

        Args:
            agent : Instance concrète de BaseAgent à enregistrer.
        """
        with self._lock:
            self._agents[agent.name] = agent

    def unregister(self, name: str) -> None:
        """Retire un agent du registre (utile pour les tests)."""
        with self._lock:
            self._agents.pop(name, None)

    def get(self, name: str) -> BaseAgent | None:
        """Retourne l'agent par son nom, ou None si inconnu.

        Args:
            name : Nom unique de l'agent (ex. "reasoning_agent").
        """
        with self._lock:
            return self._agents.get(name)

    def list_all(self) -> list[BaseAgent]:
        """Retourne la liste de tous les agents enregistrés.

        Returns:
            Liste dans l'ordre d'enregistrement (insertion order préservé).
        """
        with self._lock:
            return list(self._agents.values())

    def by_capability(self, capability: str) -> list[BaseAgent]:
        """Retourne tous les agents ayant la capacité donnée.

        Args:
            capability : Mot-clé de capacité (ex. "reasoning", "memory").

        Returns:
            Liste des agents correspondants, dans l'ordre d'enregistrement.
        """
        with self._lock:
            return [
                a for a in self._agents.values()
                if capability in a.capabilities
            ]

    def by_task_type(self, task_type: str) -> list[BaseAgent]:
        """Retourne les agents compatibles avec un type de tâche donné.

        Un agent est compatible si son nom ou l'une de ses capacités
        correspond au type de tâche (comparaison partielle insensible à
        la casse).

        Args:
            task_type : Type de tâche (ex. "reasoning", "planning").
        """
        task_type_lower = task_type.lower()
        with self._lock:
            result: list[BaseAgent] = []
            for agent in self._agents.values():
                match = (
                    task_type_lower in agent.name.lower()
                    or any(task_type_lower in cap.lower() for cap in agent.capabilities)
                )
                if match:
                    result.append(agent)
            return result

    def snapshot(self) -> list[dict[str, Any]]:
        """Retourne une vue sérialisable de tous les agents enregistrés.

        Utilisé par GET /health/full et le futur Dashboard Super Admin.
        """
        with self._lock:
            return [
                {
                    "name"                : a.name,
                    "description"         : a.description,
                    "capabilities"        : a.capabilities,
                    "autonomy"            : a.autonomy,
                    "version"             : a.version,
                    "cost_per_call"       : a.cost_per_call,
                    "confidence_threshold": a.confidence_threshold,
                    "max_concurrent"      : a.max_concurrent,
                }
                for a in self._agents.values()
            ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._agents)


# ── Singleton global ───────────────────────────────────────────────────────────

_global_registry: AgentRegistry | None = None
_global_registry_lock = threading.Lock()


def get_registry() -> AgentRegistry:
    """Retourne le singleton global AgentRegistry.

    Thread-safe. Crée l'instance au premier appel.
    """
    global _global_registry
    if _global_registry is None:
        with _global_registry_lock:
            if _global_registry is None:
                _global_registry = AgentRegistry()
    return _global_registry


def register_defaults() -> None:
    """Enregistre les agents par défaut (Phases 5, 6 et 10).

    À appeler une seule fois dans le lifespan FastAPI, après l'initialisation
    de la mémoire vectorielle.

    Agents enregistrés :
        - ReasoningAgent  (pipeline 3.1→3.5)                    [Phase 5]
        - MemoryAgent     (lecture/écriture ChromaDB)            [Phase 6]
        - PlanningAgent   (décomposition de tâches)              [Phase 6]
        - ResearchAgent   (recherche web multi-sources)          [Phase 7]
        - ArchitectAgent  (conception & architecture)            [Phase 10]
        - CodingAgent     (implémentation, debug, refactoring)   [Phase 10]
        - TestAgent       (stratégie de test & tests exécutables)[Phase 10]
        - PresenterAgent  (présentations PowerPoint .pptx)       [Phase 10]
    """
    from core.agents.reasoning_agent    import ReasoningAgent
    from core.agents.memory_agent       import MemoryAgent
    from core.agents.planning_agent     import PlanningAgent
    from core.agents.research_agent     import ResearchAgent
    from core.agents.engineering_agents import ArchitectAgent, CodingAgent, TestAgent
    from core.agents.presenter_agent    import PresenterAgent

    registry = get_registry()
    registry.register(ReasoningAgent())
    registry.register(MemoryAgent())
    registry.register(PlanningAgent())
    registry.register(ResearchAgent())
    registry.register(ArchitectAgent())
    registry.register(CodingAgent())
    registry.register(TestAgent())
    registry.register(PresenterAgent())

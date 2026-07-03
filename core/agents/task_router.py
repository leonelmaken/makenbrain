"""TaskRouter — sélectionne les meilleurs agents pour une tâche — Phase 5.

Le TaskRouter est la couche de décision entre l'Orchestrator et le Registry.
Il reçoit une AgentTask et retourne une liste ordonnée d'agents à exécuter.

Critères de sélection (dans cet ordre) :
    1. Capacités    : l'agent doit supporter le type de tâche.
    2. Disponibilité: l'agent doit être disponible (is_available()).
    3. Priorité     : les agents sont triés par coût_per_call croissant
                      (économiser les ressources pour les tâches normales)
                      et par confidence_threshold décroissant (agents les
                      plus fiables en premier).
    4. Autonomie    : pour les tâches CRITICAL, les agents AUTONOMOUS sont
                      préférés ; pour les tâches normales, l'ordre standard
                      s'applique.

Parallélisme :
    Pour la Phase 5, le Router retourne une liste séquentielle.
    La structure `RoutingPlan.parallel_groups` est déjà définie pour
    permettre à l'Orchestrator Phase 6+ de basculer en mode parallèle
    sans changer l'interface du Router.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.agents.base import AgentAutonomy
from core.agents.models import AgentTask, TaskPriority

if TYPE_CHECKING:
    from core.agents.base import BaseAgent
    from core.agents.registry import AgentRegistry


@dataclass
class RoutingPlan:
    """Plan d'exécution produit par le TaskRouter.

    Attributes:
        sequential     : Agents à exécuter l'un après l'autre (Phase 5).
        parallel_groups: Groupes d'agents à exécuter en parallèle (Phase 6+).
                         Chaque groupe est une liste d'agents exécutables
                         simultanément. Vide en Phase 5.
        fallback       : Agent de secours si tous les agents principaux échouent.
                         None si aucun fallback disponible.
    """
    sequential     : list["BaseAgent"]        = field(default_factory=list)
    parallel_groups: list[list["BaseAgent"]]  = field(default_factory=list)
    fallback       : "BaseAgent | None"        = None

    @property
    def all_agents(self) -> list["BaseAgent"]:
        """Retourne tous les agents du plan (séquentiels + parallèles)."""
        agents = list(self.sequential)
        for group in self.parallel_groups:
            agents.extend(group)
        return agents

    @property
    def is_empty(self) -> bool:
        return not self.sequential and not self.parallel_groups


class TaskRouter:
    """Routeur de tâches vers les agents appropriés.

    Instancié et utilisé par BrainOrchestrator.
    Prend ses décisions uniquement à partir du Registry et des métadonnées
    de la tâche — jamais de logique métier ici.
    """

    def __init__(self, registry: "AgentRegistry") -> None:
        self._registry = registry

    async def route(self, task: AgentTask) -> RoutingPlan:
        """Calcule le plan d'exécution optimal pour une tâche donnée.

        Phase 5 : un seul agent → séquentiel.
        Phase 8 : plusieurs agents primaires → parallel_groups ;
                  agents de fallback → meilleur en séquentiel + reste en parallèle.

        Args:
            task : La tâche à router.

        Returns:
            RoutingPlan avec sequential et/ou parallel_groups renseignés.
        """
        candidates = self._registry.by_task_type(task.type)

        available: list["BaseAgent"] = []
        for agent in candidates:
            if await agent.is_available():
                available.append(agent)

        fallback_used = False
        if not available:
            fallback_used = True
            all_agents = self._registry.list_all()
            for agent in all_agents:
                if await agent.is_available():
                    available.append(agent)

        ranked = self._rank(available, task)

        if not ranked:
            return RoutingPlan()

        if len(ranked) == 1:
            return RoutingPlan(sequential=ranked)

        if fallback_used:
            # Pas de correspondance par type : meilleur agent en séquentiel,
            # les autres s'exécutent en parallèle comme exploration complémentaire.
            return RoutingPlan(
                sequential     = [ranked[0]],
                parallel_groups= [ranked[1:]],
                fallback       = ranked[-1],
            )

        # Plusieurs agents primaires pour ce type de tâche : exécution parallèle.
        return RoutingPlan(
            parallel_groups= [ranked],
            fallback       = ranked[-1],
        )

    def _rank(
        self,
        agents: list["BaseAgent"],
        task  : AgentTask,
    ) -> list["BaseAgent"]:
        """Trie les agents selon les critères de priorité.

        Critères (ordre décroissant d'importance) :
            1. Pour les tâches CRITICAL, les agents AUTONOMOUS passent en tête.
            2. confidence_threshold décroissant (les plus fiables d'abord).
            3. cost_per_call croissant (les moins chers d'abord à égalité).
        """
        is_critical = task.priority >= TaskPriority.HIGH

        def sort_key(agent: "BaseAgent") -> tuple:
            autonomy_score = 0
            if is_critical and agent.autonomy == AgentAutonomy.AUTONOMOUS:
                autonomy_score = -1
            return (
                autonomy_score,
                -agent.confidence_threshold,
                agent.cost_per_call,
            )

        return sorted(agents, key=sort_key)

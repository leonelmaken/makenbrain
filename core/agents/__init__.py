"""Package multi-agents MakenBrain — Phase 5.

Architecture :
    BaseAgent         : Classe abstraite parente de tout agent.
    AgentAutonomy     : Enum des niveaux d'autonomie (READ_ONLY→AUTONOMOUS).
    AgentRegistry     : Registre central des agents disponibles.
    TaskRouter        : Sélectionne les meilleurs agents pour une tâche.
    BrainOrchestrator : Cœur de l'orchestration (point d'entrée HTTP).
    ReasoningAgent    : Premier agent — wraps le pipeline Phase 3.1→3.5.

Objets de communication :
    AgentTask         : Tâche à traiter (input de l'Orchestrator).
    AgentResult       : Résultat d'un agent (output vers Orchestrator).
    ExecutionContext   : Contexte partagé pendant une exécution.
    ExecutionReport   : Rapport final sérialisable vers le client HTTP.

Usage rapide :
    from core.agents import get_orchestrator
    from core.agents.models import AgentTask, TaskType

    report = await get_orchestrator().execute(
        AgentTask(type=TaskType.REASONING, input="Quelle est ma stratégie ?",
                  user_id="user-123")
    )
"""
from core.agents.base import AgentAutonomy, BaseAgent
from core.agents.models import (
    AgentResult,
    AgentTask,
    ExecutionContext,
    ExecutionReport,
    TaskPriority,
    TaskType,
)
from core.agents.registry import AgentRegistry, get_registry, register_defaults
from core.agents.task_router import RoutingPlan, TaskRouter
from core.agents.orchestrator import BrainOrchestrator, get_orchestrator
from core.agents.reasoning_agent import ReasoningAgent

__all__ = [
    # Base
    "BaseAgent",
    "AgentAutonomy",
    # Modèles
    "AgentTask",
    "AgentResult",
    "ExecutionContext",
    "ExecutionReport",
    "TaskType",
    "TaskPriority",
    # Registry
    "AgentRegistry",
    "get_registry",
    "register_defaults",
    # Router
    "TaskRouter",
    "RoutingPlan",
    # Orchestrator
    "BrainOrchestrator",
    "get_orchestrator",
    # Agents
    "ReasoningAgent",
]

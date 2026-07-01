"""Router HTTP /agents — Phase 5.

Endpoints :
    POST /agents/execute    → Soumet une tâche à l'Orchestrator.
    GET  /agents/registry   → Liste les agents enregistrés.
    GET  /agents/{name}     → Détails + health d'un agent spécifique.

Authentification :
    POST /agents/execute est public pour la Phase 5 (utilisateur authentifié
    via Supabase JWT géré par le middleware existant).
    Aucune dépendance SECURE supplémentaire pour l'instant.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.agents.models import AgentTask, TaskPriority, TaskType
from core.agents.orchestrator import get_orchestrator
from core.agents.registry import get_registry
from core.observability import get_request_context

router = APIRouter()


# ── DTOs ──────────────────────────────────────────────────────────────────────

class ExecuteRequest(BaseModel):
    """Corps de requête pour POST /agents/execute."""
    input      : str  = Field(..., min_length=1, description="La question ou tâche à traiter.")
    type       : str  = Field(TaskType.REASONING, description="Type de tâche.")
    session_id : str | None = Field(None, description="ID de session de conversation.")
    priority   : int  = Field(TaskPriority.NORMAL, ge=1, le=10)
    context    : dict = Field(default_factory=dict, description="Contexte RAG optionnel.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/execute", summary="Soumettre une tâche à l'Orchestrator")
async def execute_task(body: ExecuteRequest):
    """Soumet une tâche à BrainOrchestrator et retourne l'ExecutionReport.

    L'Orchestrator sélectionne automatiquement le(s) agent(s) approprié(s)
    via le TaskRouter, les exécute dans l'ordre optimal et agrège les résultats.

    Returns:
        ExecutionReport sérialisé (status, final_output, agents_used…).
    """
    ctx = get_request_context()

    task = AgentTask(
        type      = body.type,
        input     = body.input,
        user_id   = ctx.get("user_id"),
        session_id= body.session_id or ctx.get("session_id"),
        priority  = body.priority,
        context   = body.context,
    )

    report = await get_orchestrator().execute(
        task      = task,
        request_id= ctx.get("request_id"),
    )

    status_code = 200 if report.status != "failed" else 503
    return report.to_dict() if status_code == 200 else HTTPException(
        status_code= 503,
        detail     = report.to_dict(),
    )


@router.get("/registry", summary="Lister les agents enregistrés")
async def list_agents():
    """Retourne la liste de tous les agents enregistrés avec leurs métadonnées.

    Returns:
        Liste de dicts décrivant chaque agent (nom, capacités, autonomie…).
    """
    registry = get_registry()
    return {
        "total"  : len(registry),
        "agents" : registry.snapshot(),
    }


@router.get("/{agent_name}", summary="Détails et health d'un agent")
async def get_agent(agent_name: str):
    """Retourne les détails et le statut de santé d'un agent spécifique.

    Args:
        agent_name : Nom de l'agent (ex. "reasoning_agent").

    Returns:
        Dict avec les métadonnées et le statut de santé de l'agent.

    Raises:
        404 si l'agent n'est pas trouvé dans le Registry.
    """
    registry = get_registry()
    agent    = registry.get(agent_name)

    if agent is None:
        raise HTTPException(
            status_code= 404,
            detail     = f"Agent '{agent_name}' non trouvé dans le Registry.",
        )

    health = await agent.health()
    return health

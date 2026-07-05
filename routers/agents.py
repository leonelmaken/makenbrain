"""Router HTTP /agents — Phase 5, sécurité et Salle des Agents Phase 10.

Endpoints :
    POST /agents/execute           → Soumet une tâche à l'Orchestrator (auto-routage).
    GET  /agents/registry          → Liste les agents enregistrés.
    GET  /agents/{name}            → Détails + health d'un agent spécifique.
    POST /agents/{name}/ask        → Questionne UN agent précis directement
                                      (bypass le TaskRouter) — SuperAdmin uniquement,
                                      pour la Salle des Agents.

Authentification :
    POST /agents/execute exige désormais un utilisateur authentifié et respecte
    son quota journalier (Phase 9) — jusqu'ici public et sans limite, ce qui
    permettait de consommer le quota Groq du projet sans compte ni quota.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.agents.models import AgentTask, ExecutionContext, TaskPriority, TaskType
from core.agents.orchestrator import get_orchestrator
from core.agents.registry import get_registry
from core.auth import require_chat_user
from core.observability import get_request_context
from core.usage import enforce_quota
from models.user import User, UserRole

router = APIRouter()


# ── DTOs ──────────────────────────────────────────────────────────────────────

class ExecuteRequest(BaseModel):
    """Corps de requête pour POST /agents/execute."""
    input      : str  = Field(..., min_length=1, description="La question ou tâche à traiter.")
    type       : str  = Field(TaskType.REASONING, description="Type de tâche.")
    session_id : str | None = Field(None, description="ID de session de conversation.")
    priority   : int  = Field(TaskPriority.NORMAL, ge=1, le=10)
    context    : dict = Field(default_factory=dict, description="Contexte RAG optionnel.")


class AskAgentRequest(BaseModel):
    """Corps de requête pour POST /agents/{name}/ask."""
    message: str = Field(..., min_length=1, max_length=8000)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/execute", summary="Soumettre une tâche à l'Orchestrator")
async def execute_task(
    body: ExecuteRequest,
    current_user: User = Depends(require_chat_user),
):
    """Soumet une tâche à BrainOrchestrator et retourne l'ExecutionReport.

    L'Orchestrator sélectionne automatiquement le(s) agent(s) approprié(s)
    via le TaskRouter, les exécute dans l'ordre optimal et agrège les résultats.
    Compte dans le quota journalier de l'utilisateur, comme /chat/.

    Returns:
        ExecutionReport sérialisé (status, final_output, agents_used…).
    """
    enforce_quota(current_user)
    ctx = get_request_context()

    task = AgentTask(
        type      = body.type,
        input     = body.input,
        user_id   = str(current_user.id),
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


@router.post("/{agent_name}/ask", summary="Questionner un agent précis directement (Salle des Agents)")
async def ask_agent_directly(
    agent_name: str,
    body: AskAgentRequest,
    current_user: User = Depends(require_chat_user),
):
    """Envoie un message à UN agent précis, sans passer par le TaskRouter.

    Réservé au SuperAdmin : c'est le point d'entrée de la Salle des Agents,
    qui permet de dialoguer directement avec chaque agent pour comprendre
    son comportement, le tester ou lui donner un ordre ciblé — contrairement
    à /agents/execute qui laisse le TaskRouter choisir l'agent automatiquement.

    Returns:
        Dict avec agent_name, success, output/error, duration_ms, confidence.

    Raises:
        403 si l'appelant n'est pas SuperAdmin.
        404 si l'agent n'existe pas dans le Registry.
    """
    if current_user.role != UserRole.SUPERADMIN:
        raise HTTPException(status_code=403, detail="Accès réservé au SuperAdmin.")

    agent = get_registry().get(agent_name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' non trouvé dans le Registry.")

    request_id = str(uuid.uuid4())
    task = AgentTask(
        type   = agent.capabilities[0] if agent.capabilities else TaskType.GENERAL,
        input  = body.message,
        user_id= str(current_user.id),
    )
    ctx = ExecutionContext(request_id=request_id, task=task, user_id=str(current_user.id))

    try:
        result = await agent.run(task, ctx)
    except Exception as exc:  # noqa: BLE001 — un agent ne doit jamais faire tomber l'API.
        raise HTTPException(status_code=500, detail=f"L'agent '{agent_name}' a échoué : {exc}") from exc

    return result.to_dict()


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

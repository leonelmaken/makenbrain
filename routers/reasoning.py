"""Router FastAPI pour le moteur de raisonnement expert — Phase 3.0.

Responsabilités de ce module :
- Validation HTTP des requêtes entrantes (déléguée à Pydantic via FastAPI).
- Injection de la dépendance d'authentification Supabase.
- Délégation au reasoning_engine (aucune logique métier ici).
- Sérialisation de la réponse.
- Audit log des appels.

Ce module ne contient aucune logique métier.

Règle stricte : ce module importe depuis core/ et models/ uniquement.
Jamais l'inverse.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from core.audit import audit_event
from core.auth import require_supabase_user
from core.reasoning.reasoning_engine import context_to_response, run_reasoning
from models.reasoning import ReasoningRequest, ReasoningResponse
from models.user import User

router = APIRouter()
logger = logging.getLogger("makenbrain.routers.reasoning")


@router.post(
    "/analyze",
    response_model=ReasoningResponse,
    summary="Raisonnement expert",
    description=(
        "Lance le pipeline de raisonnement expert complet : analyse de la question, "
        "génération d'hypothèses, collecte de preuves, évaluation et synthèse. "
        "Retourne une réponse ancrée dans les données disponibles avec un score "
        "de confiance et une trace optionnelle du raisonnement."
    ),
)
async def analyze(
    request      : ReasoningRequest,
    current_user : User = Depends(require_supabase_user),
) -> ReasoningResponse:
    """Endpoint principal du moteur de raisonnement expert.

    Args:
        request      : Corps de la requête validé par Pydantic.
        current_user : Utilisateur Supabase authentifié via Bearer token.

    Returns:
        ReasoningResponse avec réponse, confiance, sources et trace optionnelle.

    Raises:
        HTTPException 401 : Token Bearer absent ou invalide.
        HTTPException 422 : Corps de requête invalide (géré automatiquement par FastAPI).
        HTTPException 500 : Erreur interne non récupérable du pipeline.
    """
    try:
        ctx = await run_reasoning(
            request = request,
            user_id = str(current_user.id),
        )
        response = context_to_response(ctx)

        audit_event(
            action   = "reasoning.analyze",
            tool     = "reasoning",
            endpoint = "/reasoning/analyze",
            result   = (
                f"confidence={response.confidence:.2f} "
                f"degraded={ctx.pipeline_degraded} "
                f"evidence={response.evidence_used}"
            ),
            success  = True,
        )

        return response

    except HTTPException:
        # Re-raise les HTTPException (ex. 401) sans les wrapper.
        raise

    except Exception as exc:
        logger.exception("[REASONING] Erreur non récupérée dans le pipeline.")
        audit_event(
            action   = "reasoning.analyze",
            tool     = "reasoning",
            endpoint = "/reasoning/analyze",
            result   = str(exc),
            success  = False,
        )
        raise HTTPException(
            status_code = 500,
            detail      = "Erreur interne du moteur de raisonnement.",
        ) from exc

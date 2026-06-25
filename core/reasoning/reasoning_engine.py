"""Orchestrateur du pipeline de raisonnement expert de MakenBrain.

Phase 3.0 — Sprint 1 : infrastructure du pipeline (stub).

Responsabilités de ce module :
- Instancier et exécuter les agents dans l'ordre du pipeline.
- Alimenter le ReasoningTrace à chaque étape (timing, résumé, succès).
- Gérer les erreurs partielles (mode dégradé : le pipeline continue).
- Sérialiser le ReasoningContext en ReasoningResponse via context_to_response().

Ce module NE contient PAS de logique métier. Toute logique métier appartient
aux modules d'étape : question_analyzer, hypothesis_engine, evidence_collector,
evidence_evaluator, synthesizer. Ils seront branchés à partir du Sprint 2.

Dépendances autorisées :
✓ core.reasoning.context
✓ models.reasoning
✗ routers/  (jamais)
✗ Logique métier directe (jamais dans cet orchestrateur)
"""
from __future__ import annotations

import logging
import time

from core.reasoning.context import ReasoningContext
from models.reasoning import (
    QuestionType,
    ReasoningRequest,
    ReasoningResponse,
    ReasoningTrace,
)

logger = logging.getLogger("makenbrain.reasoning_engine")

# ── Pipeline ──────────────────────────────────────────────────────────────────
# Liste des agents à exécuter dans l'ordre.
# Vide en Sprint 1 — les agents seront ajoutés au fil des sprints :
#   Sprint 2 : QuestionAnalyzerAgent, HypothesisEngineAgent
#   Sprint 3 : EvidenceCollectorAgent, EvidenceEvaluatorAgent
#   Sprint 4 : SynthesizerAgent
DEFAULT_PIPELINE: list = []  # type: ignore[type-arg]


async def run_reasoning(
    request : ReasoningRequest,
    user_id : str,
) -> ReasoningContext:
    """Exécute le pipeline de raisonnement complet.

    Sprint 1 : le pipeline est vide (DEFAULT_PIPELINE = []). La fonction
    initialise le contexte, parcourt le pipeline (aucune itération),
    finalise la trace et retourne le contexte.

    Les agents seront branchés à partir du Sprint 2.

    Args:
        request  : Requête de raisonnement validée par Pydantic.
        user_id  : Identifiant Supabase de l'utilisateur authentifié.

    Returns:
        ReasoningContext peuplé avec les résultats disponibles.
    """
    ctx = ReasoningContext(request=request, user_id=user_id)
    ctx.initialize_trace()

    logger.info(
        "[REASONING] Démarrage pipeline | user=%s | question='%.60s'",
        user_id,
        request.question,
    )

    pipeline_start = time.monotonic()

    for agent in DEFAULT_PIPELINE:
        step_start = time.monotonic()
        try:
            ctx = await agent.run(ctx)
        except Exception as exc:  # noqa: BLE001
            step_ms = (time.monotonic() - step_start) * 1000
            logger.error(
                "[REASONING] Étape '%s' en erreur (%.0fms) : %s",
                agent.name,
                step_ms,
                exc,
            )
            ctx.mark_degraded(agent.name, str(exc))
            # Le pipeline continue avec les données partielles disponibles.

    ctx.trace.total_duration_ms      = (time.monotonic() - pipeline_start) * 1000
    ctx.trace.evidence_count         = len(ctx.evidence)
    ctx.trace.hypotheses_count       = len(ctx.hypotheses)
    ctx.trace.evidence_quality_score = ctx.evidence_quality_score

    logger.info(
        "[REASONING] Pipeline terminé | %.0fms | dégradé=%s | preuves=%d | hypothèses=%d",
        ctx.trace.total_duration_ms,
        ctx.pipeline_degraded,
        len(ctx.evidence),
        len(ctx.hypotheses),
    )

    return ctx


def context_to_response(ctx: ReasoningContext) -> ReasoningResponse:
    """Sérialise un ReasoningContext en ReasoningResponse exposable via l'API.

    Utilisé par routers/reasoning.py pour construire la réponse HTTP.
    N'effectue aucun appel réseau ni traitement métier.

    Args:
        ctx : Contexte de raisonnement issu de run_reasoning().

    Returns:
        ReasoningResponse prêt à être retourné par le router.
    """
    analysis = ctx.analysis_or_fallback

    return ReasoningResponse(
        answer                 = ctx.final_answer or _stub_answer(ctx),
        confidence             = ctx.confidence,
        risk_level             = _compute_risk_level(ctx.confidence),
        suggested_sources      = [],
        reasoning_summary      = _build_summary(ctx),
        question_type          = analysis.question_type,
        complexity_score       = analysis.complexity_score,
        hypotheses_considered  = [h.content for h in ctx.hypotheses],
        best_hypothesis        = _best_hypothesis(ctx),
        evidence_used          = len(ctx.evidence),
        evidence_quality_score = ctx.evidence_quality_score,
        knowledge_gaps         = ctx.evaluation.knowledge_gaps if ctx.evaluation else [],
        provider               = ctx.request.provider,
        model                  = "pending",
        trace                  = ctx.trace if ctx.request.include_trace else None,
    )


# ── Helpers privés ────────────────────────────────────────────────────────────

def _stub_answer(ctx: ReasoningContext) -> str:
    """Réponse temporaire tant que les agents du pipeline ne sont pas implémentés.

    Remplacée dès que le Synthesizer (Sprint 4) est branché.
    """
    return (
        f"Moteur de raisonnement initialisé (Phase 3.0 — Sprint 1). "
        f"Pipeline en cours d'implémentation. "
        f"Question reçue : {ctx.request.question}"
    )


def _compute_risk_level(confidence: float) -> str:
    """Convertit un score de confiance en niveau de risque lisible.

    Seuils alignés sur core/response_confidence.py existant :
    - >= 0.75 → low
    - >= 0.60 → medium
    - <  0.60 → high
    """
    if confidence >= 0.75:
        return "low"
    if confidence >= 0.60:
        return "medium"
    return "high"


def _build_summary(ctx: ReasoningContext) -> str:
    """Construit un résumé en prose du raisonnement effectué."""
    parts: list[str] = []

    if ctx.analysis:
        parts.append(
            f"Question de type '{ctx.analysis.question_type.value}' "
            f"(complexité : {ctx.analysis.complexity_score:.2f})."
        )
    if ctx.hypotheses:
        parts.append(f"{len(ctx.hypotheses)} hypothèse(s) évaluée(s).")
    if ctx.evidence:
        parts.append(f"{len(ctx.evidence)} preuve(s) collectée(s).")
    if ctx.pipeline_degraded:
        parts.append("Avertissement : pipeline en mode dégradé.")

    return " ".join(parts) or "Pipeline Sprint 1 — agents à implémenter (Sprint 2+)."


def _best_hypothesis(ctx: ReasoningContext) -> str:
    """Retourne le contenu de la meilleure hypothèse identifiée par l'évaluateur."""
    if not ctx.hypotheses:
        return ""
    if ctx.evaluation and ctx.evaluation.best_hypothesis_id:
        for h in ctx.hypotheses:
            if h.hypothesis_id == ctx.evaluation.best_hypothesis_id:
                return h.content
    # Fallback : première hypothèse de la liste
    return ctx.hypotheses[0].content

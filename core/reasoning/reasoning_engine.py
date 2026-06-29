"""Orchestrateur du pipeline de raisonnement expert de MakenBrain.

Phase 3.3 — EvidenceEngineAgent ajouté au pipeline.

État du pipeline par phase :
    Phase 3.0 : []
    Phase 3.1 : [QuestionAnalyzerAgent()]
    Phase 3.2 : [QuestionAnalyzerAgent(), HypothesisEngineAgent()]
    Phase 3.3 : [QuestionAnalyzerAgent(), HypothesisEngineAgent(), EvidenceEngineAgent()]
    Phase 3.4 : + SynthesizerAgent()
"""
from __future__ import annotations

import logging
import time

from core.reasoning.context import ReasoningContext
from core.reasoning.evidence_engine import EvidenceEngineAgent
from core.reasoning.hypothesis_engine import HypothesisEngineAgent
from core.reasoning.question_analyzer import QuestionAnalyzerAgent
from models.reasoning import ReasoningRequest, ReasoningResponse

logger = logging.getLogger("makenbrain.reasoning_engine")

DEFAULT_PIPELINE = [
    QuestionAnalyzerAgent(),  # Phase 3.1 — analyse de la question
    HypothesisEngineAgent(),  # Phase 3.2 — génération d'hypothèses
    EvidenceEngineAgent(),    # Phase 3.3 — collecte et évaluation de preuves
    # SynthesizerAgent(),     # Phase 3.4 — synthèse raisonnée
]


async def run_reasoning(
    request : ReasoningRequest,
    user_id : str,
) -> ReasoningContext:
    """Exécute le pipeline complet.

    Args:
        request : Requête validée par Pydantic.
        user_id : Identifiant Supabase de l'utilisateur.

    Returns:
        ReasoningContext peuplé par toutes les étapes.
    """
    ctx = ReasoningContext(request=request, user_id=user_id)
    ctx.initialize_trace()

    logger.info(
        "[REASONING] Démarrage | user=%s | question='%.60s'",
        user_id, request.question,
    )

    pipeline_start = time.monotonic()

    for agent in DEFAULT_PIPELINE:
        try:
            ctx = await agent.run(ctx)
        except Exception as exc:  # noqa: BLE001
            logger.error("[REASONING] Agent '%s' — erreur : %s", agent.name, exc)
            ctx.mark_degraded(agent.name, str(exc))

    ctx.trace.total_duration_ms      = (time.monotonic() - pipeline_start) * 1000
    ctx.trace.evidence_count         = len(ctx.evidence)
    ctx.trace.hypotheses_count       = len(ctx.hypotheses)
    ctx.trace.evidence_quality_score = ctx.evidence_quality_score

    logger.info(
        "[REASONING] Terminé | %.0fms | hyp=%d | preuves=%d | qualité=%.2f | dégradé=%s",
        ctx.trace.total_duration_ms, len(ctx.hypotheses),
        len(ctx.evidence), ctx.evidence_quality_score, ctx.pipeline_degraded,
    )
    return ctx


def context_to_response(ctx: ReasoningContext) -> ReasoningResponse:
    """Sérialise le contexte en réponse API."""
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
    analysis = ctx.analysis_or_fallback
    parts = [
        f"[Synthesizer non implémenté — Phase 3.4]",
        f"Type={analysis.question_type.value}, domaine={analysis.domain}.",
    ]
    if ctx.hypotheses:
        parts.append(f"{len(ctx.hypotheses)} hypothèse(s) générée(s).")
    if ctx.evidence:
        best = ctx.evaluation.best_hypothesis_id if ctx.evaluation else None
        parts.append(
            f"{len(ctx.evidence)} preuve(s) collectée(s)."
            + (f" Meilleure hypothèse : {best}." if best else "")
        )
    return " ".join(parts)


def _compute_risk_level(confidence: float) -> str:
    if confidence >= 0.75: return "low"
    if confidence >= 0.60: return "medium"
    return "high"


def _build_summary(ctx: ReasoningContext) -> str:
    """Résumé en prose du raisonnement — message stable sans référence de phase."""
    analysis = ctx.analysis
    parts: list[str] = []

    if analysis:
        parts.append(
            f"Question de type '{analysis.question_type.value}' "
            f"dans le domaine '{analysis.domain}' "
            f"(complexité : {analysis.complexity_score:.2f}, risque : {analysis.risk_level})."
        )
        if analysis.requires_memory:
            parts.append("Mémoire utilisateur consultée.")
        if analysis.requires_external_search:
            parts.append("Recherche externe recommandée.")
        if analysis.requires_deep_reasoning:
            parts.append("Raisonnement approfondi requis.")

    if ctx.hypotheses:
        parts.append(f"{len(ctx.hypotheses)} hypothèse(s) générée(s).")
    if ctx.evidence:
        parts.append(f"{len(ctx.evidence)} preuve(s) collectée(s).")
        if ctx.evaluation and ctx.evaluation.contradictions:
            parts.append(f"{len(ctx.evaluation.contradictions)} contradiction(s) détectée(s).")
        if ctx.evaluation and ctx.evaluation.knowledge_gaps:
            parts.append(f"{len(ctx.evaluation.knowledge_gaps)} lacune(s) identifiée(s).")
    if ctx.pipeline_degraded:
        parts.append("⚠ Pipeline en mode dégradé.")

    return " ".join(parts) or "Analyse en cours."


def _best_hypothesis(ctx: ReasoningContext) -> str:
    if not ctx.hypotheses:
        return ""
    if ctx.evaluation and ctx.evaluation.best_hypothesis_id:
        for h in ctx.hypotheses:
            if h.hypothesis_id == ctx.evaluation.best_hypothesis_id:
                return h.content
    return ctx.hypotheses[0].content

"""Orchestrateur du pipeline de raisonnement expert de MakenBrain.

Phase 3.2 — HypothesisEngineAgent ajouté au pipeline.

Responsabilités de ce module :
- Instancier et exécuter les agents dans l'ordre du pipeline.
- Alimenter le ReasoningTrace à chaque étape (timing, résumé, succès).
- Gérer les erreurs partielles (mode dégradé : le pipeline continue).
- Sérialiser le ReasoningContext en ReasoningResponse via context_to_response().

Ce module NE contient PAS de logique métier. Toute logique métier appartient
aux modules d'étape individuels.

Dépendances autorisées :
    ✓ core.reasoning.context
    ✓ core.reasoning.question_analyzer   (Phase 3.1)
    ✓ core.reasoning.hypothesis_engine   (Phase 3.2)
    ✓ models.reasoning
    ✗ routers/  (jamais)

État du pipeline par phase :
    Phase 3.0 : DEFAULT_PIPELINE = []
    Phase 3.1 : DEFAULT_PIPELINE = [QuestionAnalyzerAgent()]
    Phase 3.2 : DEFAULT_PIPELINE = [QuestionAnalyzerAgent(), HypothesisEngineAgent()]
    Phase 3.3 : + EvidenceCollectorAgent(), EvidenceEvaluatorAgent()
    Phase 3.4 : + SynthesizerAgent()
"""
from __future__ import annotations

import logging
import time

from core.reasoning.context import ReasoningContext
from core.reasoning.hypothesis_engine import HypothesisEngineAgent
from core.reasoning.question_analyzer import QuestionAnalyzerAgent
from models.reasoning import (
    ReasoningRequest,
    ReasoningResponse,
)

logger = logging.getLogger("makenbrain.reasoning_engine")

# ── Pipeline ──────────────────────────────────────────────────────────────────

DEFAULT_PIPELINE = [
    QuestionAnalyzerAgent(),   # Phase 3.1 — étape 1 : analyse de la question
    HypothesisEngineAgent(),   # Phase 3.2 — étape 2 : génération d'hypothèses
    # EvidenceCollectorAgent(),# Phase 3.3 — étape 3 : collecte de preuves
    # EvidenceEvaluatorAgent(),# Phase 3.3 — étape 4 : évaluation des preuves
    # SynthesizerAgent(),      # Phase 3.4 — étape 5 : synthèse raisonnée
]


async def run_reasoning(
    request : ReasoningRequest,
    user_id : str,
) -> ReasoningContext:
    """Exécute le pipeline de raisonnement complet.

    Args:
        request : Requête validée par Pydantic.
        user_id : Identifiant Supabase de l'utilisateur authentifié.

    Returns:
        ReasoningContext peuplé avec les résultats de toutes les étapes.
    """
    ctx = ReasoningContext(request=request, user_id=user_id)
    ctx.initialize_trace()

    logger.info(
        "[REASONING] Démarrage | user=%s | question='%.60s'",
        user_id,
        request.question,
    )

    pipeline_start = time.monotonic()

    for agent in DEFAULT_PIPELINE:
        try:
            ctx = await agent.run(ctx)
        except Exception as exc:  # noqa: BLE001
            logger.error("[REASONING] Agent '%s' — erreur non gérée : %s", agent.name, exc)
            ctx.mark_degraded(agent.name, str(exc))

    ctx.trace.total_duration_ms      = (time.monotonic() - pipeline_start) * 1000
    ctx.trace.evidence_count         = len(ctx.evidence)
    ctx.trace.hypotheses_count       = len(ctx.hypotheses)
    ctx.trace.evidence_quality_score = ctx.evidence_quality_score

    logger.info(
        "[REASONING] Terminé | %.0fms | dégradé=%s | hypothèses=%d | preuves=%d",
        ctx.trace.total_duration_ms,
        ctx.pipeline_degraded,
        len(ctx.hypotheses),
        len(ctx.evidence),
    )

    return ctx


def context_to_response(ctx: ReasoningContext) -> ReasoningResponse:
    """Sérialise un ReasoningContext en ReasoningResponse exposable via l'API.

    Args:
        ctx : Contexte issu de run_reasoning().

    Returns:
        ReasoningResponse prêt pour le router.
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
    """Réponse temporaire tant que le Synthesizer (Phase 3.4) n'est pas branché."""
    analysis = ctx.analysis_or_fallback
    hyp_summary = (
        f" {len(ctx.hypotheses)} hypothèse(s) générée(s)."
        if ctx.hypotheses else ""
    )
    return (
        f"[Synthesizer non encore implémenté — Phase 3.4] "
        f"Type={analysis.question_type.value}, "
        f"domaine={analysis.domain}, "
        f"complexité={analysis.complexity_score:.2f}."
        f"{hyp_summary}"
    )


def _compute_risk_level(confidence: float) -> str:
    """Convertit un score de confiance en niveau de risque lisible.

    Seuils alignés sur core/response_confidence.py :
        >= 0.75 → low | >= 0.60 → medium | < 0.60 → high
    """
    if confidence >= 0.75:
        return "low"
    if confidence >= 0.60:
        return "medium"
    return "high"


def _build_summary(ctx: ReasoningContext) -> str:
    """Construit un résumé en prose du raisonnement effectué.

    Le message par défaut est intentionnellement générique (sans référence
    à un sprint ou une phase) pour rester stable d'une phase à l'autre.
    """
    analysis = ctx.analysis
    parts: list[str] = []

    if analysis:
        parts.append(
            f"Question de type '{analysis.question_type.value}' "
            f"dans le domaine '{analysis.domain}' "
            f"(complexité : {analysis.complexity_score:.2f}, "
            f"risque : {analysis.risk_level})."
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
    if ctx.pipeline_degraded:
        parts.append("⚠ Pipeline en mode dégradé.")

    return " ".join(parts) or "Analyse en cours."


def _best_hypothesis(ctx: ReasoningContext) -> str:
    """Retourne le contenu de la meilleure hypothèse identifiée."""
    if not ctx.hypotheses:
        return ""
    if ctx.evaluation and ctx.evaluation.best_hypothesis_id:
        for h in ctx.hypotheses:
            if h.hypothesis_id == ctx.evaluation.best_hypothesis_id:
                return h.content
    return ctx.hypotheses[0].content

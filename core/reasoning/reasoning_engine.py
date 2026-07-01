"""Orchestrateur du pipeline de raisonnement expert de MakenBrain.

Phase 3.5 — SynthesizerAgent ajouté. Pipeline complet opérationnel.

État du pipeline par phase :
    Phase 3.0 : []
    Phase 3.1 : [QuestionAnalyzerAgent()]
    Phase 3.2 : [QuestionAnalyzerAgent(), HypothesisEngineAgent()]
    Phase 3.3 : + EvidenceEngineAgent()
    Phase 3.4 : + DecisionEngineAgent()
    Phase 3.5 : + SynthesizerAgent()  ← Pipeline complet ✓

Notes Phase 3.5 :
    ctx.final_answer est désormais peuplé par SynthesizerAgent.
    context_to_response() utilise ctx.final_answer directement (plus de _stub_answer).
    ctx.synthesis_result expose les métadonnées de synthèse pour l'observabilité.
"""
from __future__ import annotations

import logging
import time

from core.reasoning.context import ReasoningContext
from core.reasoning.decision_engine import DecisionEngineAgent
from core.reasoning.evidence_engine import EvidenceEngineAgent
from core.reasoning.hypothesis_engine import HypothesisEngineAgent
from core.reasoning.question_analyzer import QuestionAnalyzerAgent
from core.reasoning.synthesizer import SynthesizerAgent
from models.reasoning import ReasoningRequest, ReasoningResponse

logger = logging.getLogger("makenbrain.reasoning_engine")

DEFAULT_PIPELINE = [
    QuestionAnalyzerAgent(),  # Phase 3.1 — analyse de la question
    HypothesisEngineAgent(),  # Phase 3.2 — génération d'hypothèses
    EvidenceEngineAgent(),    # Phase 3.3 — collecte et évaluation de preuves
    DecisionEngineAgent(),    # Phase 3.4 — décision finale argumentée
    SynthesizerAgent(),       # Phase 3.5 — synthèse en langage naturel ✓
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
        "[REASONING] Terminé | %.0fms | hyp=%d | preuves=%d | "
        "confiance=%.2f | dégradé=%s",
        ctx.trace.total_duration_ms, len(ctx.hypotheses),
        len(ctx.evidence), ctx.confidence, ctx.pipeline_degraded,
    )
    return ctx


def context_to_response(ctx: ReasoningContext) -> ReasoningResponse:
    """Sérialise le contexte en réponse API.

    Phase 3.5 : ctx.final_answer est désormais produit par SynthesizerAgent.
    La réponse reflète l'intégralité du pipeline (3.1 → 3.5).
    """
    analysis = ctx.analysis_or_fallback
    return ReasoningResponse(
        answer                 = ctx.final_answer or _fallback_answer(ctx),
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

def _fallback_answer(ctx: ReasoningContext) -> str:
    """Réponse de dernier recours si le Synthesizer a échoué sans produire de texte.

    Ce cas ne devrait jamais se produire en fonctionnement normal car
    le Synthesizer dispose lui-même d'un fallback déterministe. Cette
    fonction est conservée comme filet de sécurité supplémentaire.
    """
    if ctx.hypotheses and ctx.decision and ctx.decision.selected_hypothesis_id:
        best = next(
            (c for c in ctx.decision.candidates
             if c.hypothesis_id == ctx.decision.selected_hypothesis_id),
            None,
        )
        if best:
            return best.content
    if ctx.hypotheses:
        return ctx.hypotheses[0].content
    return "L'analyse n'a pas pu produire de réponse. Veuillez reformuler votre question."


def _compute_risk_level(confidence: float) -> str:
    if confidence >= 0.75: return "low"
    if confidence >= 0.60: return "medium"
    return "high"


def _build_summary(ctx: ReasoningContext) -> str:
    """Résumé en prose du raisonnement — champ reasoning_summary de la réponse API.

    Phase 3.5 : intègre la stratégie et le ton du Synthesizer quand disponible.
    """
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
    if ctx.decision:
        parts.append(f"Décision : qualité {ctx.decision.decision_quality}.")
        if ctx.decision.reason.residual_risks:
            parts.append(
                f"{len(ctx.decision.reason.residual_risks)} risque(s) résiduel(s) identifié(s)."
            )
    if ctx.synthesis_result:
        parts.append(
            f"Synthèse : stratégie={ctx.synthesis_result.strategy.value}, "
            f"ton={ctx.synthesis_result.tone.value}."
        )
    if ctx.pipeline_degraded:
        parts.append("⚠ Pipeline en mode dégradé.")

    return " ".join(parts) or "Analyse en cours."


def _best_hypothesis(ctx: ReasoningContext) -> str:
    """Retourne le contenu de la meilleure hypothèse.

    Phase 3.4 : priorise ctx.decision.selected_hypothesis_id quand
    disponible (source de vérité la plus fiable), puis retombe sur
    ctx.evaluation.best_hypothesis_id (Phase 3.3), puis la première
    hypothèse de la liste (fallback historique).
    """
    if not ctx.hypotheses:
        return ""

    if ctx.decision and ctx.decision.selected_hypothesis_id:
        for h in ctx.hypotheses:
            if h.hypothesis_id == ctx.decision.selected_hypothesis_id:
                return h.content

    if ctx.evaluation and ctx.evaluation.best_hypothesis_id:
        for h in ctx.hypotheses:
            if h.hypothesis_id == ctx.evaluation.best_hypothesis_id:
                return h.content

    return ctx.hypotheses[0].content

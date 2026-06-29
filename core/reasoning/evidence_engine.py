"""Moteur de collecte et d'évaluation de preuves — Phase 3.3.

Ce module orchestre la collecte parallèle de preuves depuis plusieurs
stratégies, leur scoring par l'EvidenceRanker, puis leur validation
par l'EvidenceValidator qui produit l'EvidenceEvaluation finale.

Flux d'exécution :
    ctx (avec analysis + hypotheses) →
        [stratégies collectées en parallèle via asyncio.gather()] →
        fusion de toutes les preuves →
        EvidenceRanker (scoring + tri + top-N) →
        EvidenceValidator (contradictions + gaps + support_scores) →
        ctx.evidence peuplé
        ctx.evaluation peuplé
        ctx.evidence_quality_score mis à jour

Architecture :
    EvidenceEngineAgent        ← implémente ReasoningAgent (protocole Phase 3.0)
    collect_evidence()         ← fonction publique injectable/testable
    DEFAULT_STRATEGIES         ← configuration par défaut du moteur

Extensibilité (Phase 3.4+) :
    Ajouter une stratégie : implémenter EvidenceCollectionStrategy et
    l'inscrire dans DEFAULT_STRATEGIES. Aucune autre modification nécessaire.

Dépendances autorisées :
    ✓ core.reasoning.context
    ✓ core.reasoning.evidence_collector
    ✓ core.reasoning.evidence_ranker
    ✓ core.reasoning.evidence_validator
    ✓ models.reasoning
    ✗ routers/
    ✗ core.reasoning.reasoning_engine  (évite le cycle)
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from models.reasoning import Evidence, ReasoningStepRecord
from core.reasoning.context import ReasoningContext
from core.reasoning.evidence_collector import (
    ConceptEvidenceCollector,
    CounterEvidenceCollector,
    DomainEvidenceCollector,
    EvidenceCollectionStrategy,
    HypothesisEvidenceCollector,
    LLMCallable,
    LLMEvidenceCollector,
)
from core.reasoning.evidence_ranker import EvidenceRanker
from core.reasoning.evidence_validator import EvidenceValidator

logger = logging.getLogger("makenbrain.reasoning.evidence_engine")

# ── Stratégies par défaut ─────────────────────────────────────────────────────
#
# Toutes s'exécutent en parallèle. Chaque échec est isolé (retourne []).
# Pour ajouter une stratégie → l'implémenter et l'inscrire ici.

DEFAULT_STRATEGIES: list[EvidenceCollectionStrategy] = [
    HypothesisEvidenceCollector(),   # soutien depuis les hypothèses
    CounterEvidenceCollector(),      # contre-preuves systématiques
    ConceptEvidenceCollector(),      # preuves conceptuelles (key_concepts)
    DomainEvidenceCollector(),       # faits généraux du domaine
    LLMEvidenceCollector(),          # preuves LLM (skip si LLM absent)
]

# Nombre maximum de preuves à conserver après scoring
_DEFAULT_MAX_EVIDENCE: int = 20

# Seuil composite minimum pour garder une preuve
_DEFAULT_MIN_COMPOSITE: float = 0.20


# ── Agent (protocole ReasoningAgent) ─────────────────────────────────────────

class EvidenceEngineAgent:
    """Agent de collecte et d'évaluation de preuves — étape 3 du pipeline.

    Implémente le protocole ``ReasoningAgent`` (core.reasoning.__init__).

    Responsabilités :
        - Orchestrer les stratégies de collecte en parallèle.
        - Scorer les preuves collectées (EvidenceRanker).
        - Valider et évaluer la cohérence (EvidenceValidator).
        - Peupler ctx.evidence, ctx.evaluation, ctx.evidence_quality_score.
        - Alimenter ctx.trace avec un ReasoningStepRecord.

    Le moteur fonctionne même sans hypothèses, même sans LLM, même si
    toutes les sources échouent simultanément (retourne [] + evaluation vide).

    Args :
        strategies    : Stratégies à utiliser. Défaut : DEFAULT_STRATEGIES.
        llm_generate  : Fonction LLM injectable pour LLMEvidenceCollector.
        ranker        : EvidenceRanker à utiliser. Défaut : EvidenceRanker().
        validator     : EvidenceValidator. Défaut : EvidenceValidator().
        max_evidence  : Nombre max de preuves conservées. Défaut : 20.
        min_composite : Score composite minimum. Défaut : 0.20.
    """

    name: str = "evidence_engine"

    def __init__(
        self,
        strategies    : Optional[list[EvidenceCollectionStrategy]] = None,
        llm_generate  : Optional[LLMCallable]                      = None,
        ranker        : Optional[EvidenceRanker]                   = None,
        validator     : Optional[EvidenceValidator]                = None,
        max_evidence  : int                                        = _DEFAULT_MAX_EVIDENCE,
        min_composite : float                                      = _DEFAULT_MIN_COMPOSITE,
    ) -> None:
        self._strategies    = strategies
        self._llm_generate  = llm_generate
        self._ranker        = ranker or EvidenceRanker()
        self._validator     = validator or EvidenceValidator()
        self._max_evidence  = max_evidence
        self._min_composite = min_composite

    async def run(self, ctx: ReasoningContext) -> ReasoningContext:
        """Exécute la collecte et l'évaluation de preuves.

        Délègue à ``collect_evidence()`` pour rester testable
        indépendamment du protocole agent.

        Args:
            ctx : Contexte de raisonnement courant.

        Returns:
            ctx enrichi avec evidence, evaluation et evidence_quality_score.
        """
        return await collect_evidence(
            ctx,
            strategies    = self._strategies,
            llm_generate  = self._llm_generate,
            ranker        = self._ranker,
            validator     = self._validator,
            max_evidence  = self._max_evidence,
            min_composite = self._min_composite,
        )


# ── Point d'entrée public ─────────────────────────────────────────────────────

async def collect_evidence(
    ctx           : ReasoningContext,
    strategies    : Optional[list[EvidenceCollectionStrategy]] = None,
    llm_generate  : Optional[LLMCallable]                      = None,
    ranker        : Optional[EvidenceRanker]                   = None,
    validator     : Optional[EvidenceValidator]                = None,
    max_evidence  : int                                        = _DEFAULT_MAX_EVIDENCE,
    min_composite : float                                      = _DEFAULT_MIN_COMPOSITE,
) -> ReasoningContext:
    """Collecte, score, filtre et évalue les preuves.

    Étapes :
        1. Résoudre les stratégies (paramètre ou DEFAULT_STRATEGIES).
        2. Exécuter toutes les stratégies en parallèle (asyncio.gather).
        3. Fusionner toutes les preuves collectées.
        4. Scorer avec EvidenceRanker (relevance + credibility).
        5. Filtrer par score composite minimum.
        6. Sélectionner les N meilleures.
        7. Évaluer avec EvidenceValidator (contradictions, gaps, support).
        8. Peupler ctx.evidence, ctx.evaluation, ctx.evidence_quality_score.
        9. Enregistrer dans ctx.trace.

    Args:
        ctx           : Contexte de raisonnement courant.
        strategies    : Stratégies à utiliser (None → DEFAULT_STRATEGIES).
        llm_generate  : Fonction LLM injectable pour LLMEvidenceCollector.
        ranker        : EvidenceRanker (None → EvidenceRanker()).
        validator     : EvidenceValidator (None → EvidenceValidator()).
        max_evidence  : Nombre max de preuves conservées.
        min_composite : Score composite minimum pour garder une preuve.

    Returns:
        ctx enrichi.
    """
    step_start   = time.monotonic()
    _ranker      = ranker    or EvidenceRanker()
    _validator   = validator or EvidenceValidator()
    _strategies  = _resolve_strategies(strategies, llm_generate)

    logger.info(
        "[EvidenceEngine] Collecte | %d stratégie(s) | %d hypothèse(s)",
        len(_strategies),
        len(ctx.hypotheses),
    )

    # ── Collecte parallèle ────────────────────────────────────────────────────
    raw_batches: list[list[Evidence]] = await asyncio.gather(
        *[_safe_collect(s, ctx) for s in _strategies],
        return_exceptions=False,  # _safe_collect ne lève jamais
    )

    all_evidence: list[Evidence] = [
        ev for batch in raw_batches for ev in batch
    ]
    logger.debug("[EvidenceEngine] %d preuves brutes collectées", len(all_evidence))

    # ── Scoring + filtrage + top-N ────────────────────────────────────────────
    scored   = _ranker.score(all_evidence, ctx)
    filtered = _ranker.filter_by_threshold(scored, min_composite)
    final    = _ranker.top_n(filtered, max_evidence)

    logger.debug(
        "[EvidenceEngine] %d brutes → %d après filtre → %d sélectionnées",
        len(all_evidence),
        len(filtered),
        len(final),
    )

    # ── Évaluation ────────────────────────────────────────────────────────────
    evaluation = _validator.build_evaluation(ctx.hypotheses, final)

    # ── Mise à jour du contexte ───────────────────────────────────────────────
    ctx.evidence               = final
    ctx.evaluation             = evaluation
    ctx.evidence_quality_score = evaluation.overall_quality_score
    ctx.trace.evidence_count   = len(final)
    ctx.trace.evidence_quality_score = evaluation.overall_quality_score

    duration_ms = (time.monotonic() - step_start) * 1000

    # ── Trace ────────────────────────────────────────────────────────────────
    strategy_names = ", ".join(s.name for s in _strategies)
    ctx.trace.steps.append(ReasoningStepRecord(
        step_name      = "evidence_engine",
        duration_ms    = round(duration_ms, 2),
        success        = True,
        input_summary  = (
            f"{len(ctx.hypotheses)} hypothèse(s) | "
            f"stratégies=[{strategy_names}]"
        ),
        output_summary = (
            f"{len(all_evidence)} brutes → {len(final)} sélectionnées | "
            f"qualité={evaluation.overall_quality_score:.2f} | "
            f"contradictions={len(evaluation.contradictions)} | "
            f"lacunes={len(evaluation.knowledge_gaps)} | "
            f"meilleure_hyp={evaluation.best_hypothesis_id or 'aucune'}"
        ),
        degraded       = len(final) == 0 and bool(ctx.hypotheses),
        error_message  = (
            "Aucune preuve collectée malgré des hypothèses présentes."
            if len(final) == 0 and ctx.hypotheses
            else None
        ),
    ))

    logger.info(
        "[EvidenceEngine] Terminé %.0fms | %d preuves | qualité=%.2f | "
        "meilleure_hyp=%s",
        duration_ms,
        len(final),
        evaluation.overall_quality_score,
        evaluation.best_hypothesis_id or "aucune",
    )

    return ctx


# ── Helpers privés ────────────────────────────────────────────────────────────

def _resolve_strategies(
    strategies   : Optional[list[EvidenceCollectionStrategy]],
    llm_generate : Optional[LLMCallable],
) -> list[EvidenceCollectionStrategy]:
    """Résout la liste de stratégies à utiliser.

    Si ``strategies`` est fourni, l'utilise tel quel.
    Sinon, utilise DEFAULT_STRATEGIES en remplaçant LLMEvidenceCollector
    par une instance configurée avec ``llm_generate`` si fourni.

    Args:
        strategies   : Stratégies fournies par l'appelant (ou None).
        llm_generate : Fonction LLM injectable (ou None).

    Returns:
        Liste de stratégies résolue.
    """
    if strategies is not None:
        return strategies

    if llm_generate is None:
        return DEFAULT_STRATEGIES

    # Injecter llm_generate dans LLMEvidenceCollector
    resolved: list[EvidenceCollectionStrategy] = []
    for s in DEFAULT_STRATEGIES:
        if isinstance(s, LLMEvidenceCollector):
            resolved.append(LLMEvidenceCollector(llm_generate=llm_generate))
        else:
            resolved.append(s)
    return resolved


async def _safe_collect(
    strategy : EvidenceCollectionStrategy,
    ctx      : ReasoningContext,
) -> list[Evidence]:
    """Wrapper sécurisé autour de strategy.collect().

    Garantit que toute exception non gérée dans une stratégie
    retourne [] sans interrompre le pipeline.

    Args:
        strategy : Stratégie à exécuter.
        ctx      : Contexte courant.

    Returns:
        Liste de preuves (vide si erreur).
    """
    try:
        return await strategy.collect(ctx)
    except Exception as exc:
        logger.error(
            "[EvidenceEngine] Stratégie '%s' — exception non gérée : %s",
            strategy.name,
            exc,
        )
        return []

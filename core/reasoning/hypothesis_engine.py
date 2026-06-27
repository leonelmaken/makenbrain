"""Moteur de génération d'hypothèses — Phase 3.2.

Ce module orchestre la génération d'hypothèses en combinant plusieurs
stratégies indépendantes (Pattern Strategy) et en appliquant un classement
pour ne retenir que les meilleures.

Architecture :
    HypothesisEngineAgent      ← implémente ReasoningAgent (étape 2 du pipeline)
    generate_hypotheses()      ← fonction publique injectable/testable
    DEFAULT_STRATEGIES         ← configuration par défaut du moteur

Flux d'exécution :
    ctx (avec analysis) →
        [stratégie 1, …, stratégie N] (exécutées en parallèle) →
        fusion de toutes les hypothèses →
        HypothesisRanker (dédup + score + top-N) →
        ctx.hypotheses peuplé

Extensibilité :
    Pour ajouter une stratégie : implémenter ``HypothesisStrategy``
    (défini dans hypothesis_strategies.py) et l'ajouter à ``DEFAULT_STRATEGIES``.
    Aucune autre modification n'est nécessaire.

Dépendances autorisées :
    ✓ core.reasoning.context
    ✓ core.reasoning.hypothesis_strategies
    ✓ core.reasoning.hypothesis_ranker
    ✓ models.reasoning
    ✗ routers/  (jamais)
    ✗ core.reasoning.reasoning_engine  (jamais — évite le cycle)
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from models.reasoning import Hypothesis, ReasoningStepRecord
from core.reasoning.context import ReasoningContext
from core.reasoning.hypothesis_ranker import HypothesisRanker
from core.reasoning.hypothesis_strategies import (
    ConceptualStrategy,
    DecompositionStrategy,
    HeuristicStrategy,
    HypothesisStrategy,
    LLMCallable,
    LLMStrategy,
)

logger = logging.getLogger("makenbrain.reasoning.hypothesis_engine")

# ── Pipeline de stratégies par défaut ─────────────────────────────────────────
#
# Ordre de déclaration sans importance : toutes les stratégies sont exécutées
# en parallèle (asyncio.gather). Seul le classement final compte.
#
# Pour désactiver une stratégie, la retirer de cette liste.
# Pour en ajouter une, l'implémenter et l'inscrire ici.

DEFAULT_STRATEGIES: list[HypothesisStrategy] = [
    HeuristicStrategy(),      # déterministe, toujours disponible
    DecompositionStrategy(),  # basée sur sub_questions du QuestionAnalyzer
    ConceptualStrategy(),     # basée sur key_concepts du QuestionAnalyzer
    LLMStrategy(),            # optionnel — skippé si LLM indisponible
]


# ── Agent (protocole ReasoningAgent) ─────────────────────────────────────────

class HypothesisEngineAgent:
    """Agent de génération d'hypothèses — étape 2 du pipeline de raisonnement.

    Implémente le protocole ``ReasoningAgent`` défini dans
    ``core.reasoning.__init__``.

    Responsabilités :
        - Orchestrer les stratégies de génération d'hypothèses.
        - Collecter et fusionner les hypothèses de toutes les stratégies.
        - Déléguer le classement/filtrage à HypothesisRanker.
        - Peupler ``ctx.hypotheses`` et mettre à jour ``ctx.trace``.

    Le moteur fonctionne même sans LLM, même avec une mémoire vide,
    même si une ou plusieurs stratégies retournent [].

    Usage dans le pipeline :
        Instancié une fois dans ``DEFAULT_PIPELINE`` de reasoning_engine.py.
        Peut être configuré avec des stratégies et un LLM spécifiques.

    Args :
        strategies    : Stratégies à utiliser. Par défaut : DEFAULT_STRATEGIES.
        llm_generate  : Fonction LLM injectable pour LLMStrategy.
                        Si None, LLMStrategy tente core.llm.generate.
        ranker        : HypothesisRanker à utiliser. Par défaut : HypothesisRanker().
    """

    name: str = "hypothesis_engine"

    def __init__(
        self,
        strategies   : Optional[list[HypothesisStrategy]] = None,
        llm_generate : Optional[LLMCallable]              = None,
        ranker       : Optional[HypothesisRanker]         = None,
    ) -> None:
        self._strategies    = strategies
        self._llm_generate  = llm_generate
        self._ranker        = ranker or HypothesisRanker()

    async def run(self, ctx: ReasoningContext) -> ReasoningContext:
        """Exécute la génération d'hypothèses et enrichit le contexte.

        Délègue à ``generate_hypotheses()`` pour rester testable
        indépendamment du protocole.

        Args:
            ctx : Contexte de raisonnement courant.

        Returns:
            ctx enrichi avec ctx.hypotheses peuplé et ctx.trace mis à jour.
        """
        return await generate_hypotheses(
            ctx,
            strategies   = self._strategies,
            llm_generate = self._llm_generate,
            ranker       = self._ranker,
        )


# ── Point d'entrée public ─────────────────────────────────────────────────────

async def generate_hypotheses(
    ctx          : ReasoningContext,
    strategies   : Optional[list[HypothesisStrategy]] = None,
    llm_generate : Optional[LLMCallable]              = None,
    ranker       : Optional[HypothesisRanker]         = None,
) -> ReasoningContext:
    """Génère, classe et stocke les hypothèses dans le contexte.

    Fonction publique du module, testable sans instancier l'agent.

    Étapes :
        1. Résoudre les stratégies (paramètre ou DEFAULT_STRATEGIES).
        2. Injecter ``llm_generate`` dans LLMStrategy si fourni.
        3. Exécuter toutes les stratégies en parallèle (asyncio.gather).
        4. Fusionner toutes les hypothèses collectées.
        5. Classer via HypothesisRanker (dédup + score + top-N).
        6. Peupler ctx.hypotheses et ctx.trace.

    Args:
        ctx          : Contexte de raisonnement courant.
        strategies   : Stratégies à utiliser. Si None → DEFAULT_STRATEGIES.
        llm_generate : Fonction LLM injectable pour LLMStrategy.
        ranker       : HypothesisRanker à utiliser. Si None → HypothesisRanker().

    Returns:
        ctx enrichi.
    """
    step_start  = time.monotonic()
    _ranker     = ranker or HypothesisRanker()
    _strategies = _resolve_strategies(strategies, llm_generate)
    n_target    = ctx.request.max_hypotheses

    logger.info(
        "[HypothesisEngine] Génération | %d stratégie(s) | max=%d",
        len(_strategies),
        n_target,
    )

    # ── Exécution parallèle de toutes les stratégies ──────────────────────────
    raw_results: list[list[Hypothesis]] = await asyncio.gather(
        *[_safe_generate(strategy, ctx) for strategy in _strategies],
        return_exceptions=False,  # _safe_generate ne lève jamais
    )

    # Fusion de toutes les hypothèses
    all_hypotheses: list[Hypothesis] = [
        h for batch in raw_results for h in batch
    ]

    logger.debug(
        "[HypothesisEngine] %d hypothèses brutes collectées",
        len(all_hypotheses),
    )

    # ── Classement : déduplication → scoring → top-N ─────────────────────────
    final_hypotheses = _ranker.rank(all_hypotheses, ctx, n=n_target)

    ctx.hypotheses = final_hypotheses
    ctx.trace.hypotheses_count = len(final_hypotheses)

    duration_ms = (time.monotonic() - step_start) * 1000

    # ── Enregistrement dans la trace ──────────────────────────────────────────
    strategy_names = ", ".join(s.name for s in _strategies)
    ctx.trace.steps.append(ReasoningStepRecord(
        step_name      = "hypothesis_engine",
        duration_ms    = round(duration_ms, 2),
        success        = True,
        input_summary  = (
            f"question='{ctx.request.question[:60]}' "
            f"stratégies=[{strategy_names}]"
        ),
        output_summary = (
            f"{len(all_hypotheses)} hypothèses brutes → "
            f"{len(final_hypotheses)} après dédup+score+top-{n_target} | "
            f"scores=[{', '.join(f'{h.initial_score:.2f}' for h in final_hypotheses)}]"
        ),
        degraded       = len(final_hypotheses) == 0,
        error_message  = "Aucune hypothèse générée." if not final_hypotheses else None,
    ))

    logger.info(
        "[HypothesisEngine] Terminé %.0fms | %d hypothèses finales",
        duration_ms,
        len(final_hypotheses),
    )

    return ctx


# ── Helpers privés ────────────────────────────────────────────────────────────

def _resolve_strategies(
    strategies   : Optional[list[HypothesisStrategy]],
    llm_generate : Optional[LLMCallable],
) -> list[HypothesisStrategy]:
    """Résout la liste de stratégies à utiliser.

    Si ``strategies`` est fourni, l'utilise tel quel.
    Sinon, utilise DEFAULT_STRATEGIES en remplaçant LLMStrategy par une
    instance configurée avec ``llm_generate`` si fourni.

    Args:
        strategies   : Liste de stratégies fournie par l'appelant (ou None).
        llm_generate : Fonction LLM injectable (ou None).

    Returns:
        Liste de stratégies résolue, jamais None.
    """
    if strategies is not None:
        return strategies

    if llm_generate is None:
        return DEFAULT_STRATEGIES

    # Remplacer LLMStrategy par une instance avec llm_generate injecté
    resolved: list[HypothesisStrategy] = []
    for s in DEFAULT_STRATEGIES:
        if isinstance(s, LLMStrategy):
            resolved.append(LLMStrategy(llm_generate=llm_generate))
        else:
            resolved.append(s)
    return resolved


async def _safe_generate(
    strategy : HypothesisStrategy,
    ctx      : ReasoningContext,
) -> list[Hypothesis]:
    """Wrapper sécurisé autour de strategy.generate().

    Garantit que même si une stratégie lève une exception non gérée,
    le pipeline continue avec une liste vide pour cette stratégie.

    Args:
        strategy : Stratégie à exécuter.
        ctx      : Contexte courant.

    Returns:
        Liste d'hypothèses (vide en cas d'erreur).
    """
    try:
        return await strategy.generate(ctx)
    except Exception as exc:
        logger.error(
            "[HypothesisEngine] Stratégie '%s' — exception non gérée : %s",
            strategy.name,
            exc,
        )
        return []

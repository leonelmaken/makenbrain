"""ReasoningAgent — premier agent MakenBrain — Phase 5.

Encapsule le pipeline de raisonnement expert Phase 3.1 → 3.5 :
    QuestionAnalyzer → HypothesisEngine → EvidenceEngine
    → DecisionEngine → Synthesizer

Ce n'est pas une réécriture du pipeline — c'est un adaptateur qui
présente l'interface BaseAgent autour du code existant de Phase 3.x.

Autonomie : READ_ONLY — l'agent raisonne et propose des réponses,
il n'écrit pas en mémoire et ne prend aucune action externe.
"""
from __future__ import annotations

import time
from typing import Any

from core.agents.base import AgentAutonomy, BaseAgent
from core.agents.models import AgentResult, AgentTask, ExecutionContext
from core.reasoning.reasoning_engine import run_reasoning
from models.reasoning import ReasoningRequest


class ReasoningAgent(BaseAgent):
    """Agent de raisonnement expert (pipeline Phase 3.1→3.5).

    Capabilities :
        "reasoning" : questions analytiques, hypothèses, décisions.
        "qa"        : questions-réponses générales.
        "analysis"  : analyse de texte, code, données.
    """

    name                : str           = "reasoning_agent"
    description         : str           = (
        "Agent de raisonnement expert qui orchestre le pipeline Phase 3.1→3.5 "
        "(QuestionAnalyzer → HypothesisEngine → EvidenceEngine → "
        "DecisionEngine → Synthesizer) pour produire des réponses analytiques "
        "fondées sur la mémoire personnelle de MAKEN."
    )
    capabilities        : list[str]     = ["reasoning", "qa", "analysis", "general"]
    autonomy            : AgentAutonomy = AgentAutonomy.READ_ONLY
    version             : str           = "1.0.0"

    cost_per_call       : float         = 1.0
    confidence_threshold: float         = 0.60
    compatible_models   : list[str]     = []  # compatible avec tout provider

    async def run(self, task: AgentTask, ctx: ExecutionContext) -> AgentResult:
        """Exécute le pipeline de raisonnement sur la tâche donnée.

        Extrait la question depuis task.input, transmet le contexte
        mémoire depuis task.context.get("memory", ""), et retourne
        la réponse synthétisée par le Synthesizer.

        Args:
            task : La tâche de raisonnement.
            ctx  : Contexte d'exécution (shared["memory"] utilisé si présent).

        Returns:
            AgentResult avec success=True et la réponse synthétisée dans output.
        """
        t0 = time.monotonic()

        request = ReasoningRequest(
            question  = task.input,
            session_id= task.session_id,
        )

        try:
            reasoning_ctx = await run_reasoning(request, user_id=task.user_id or "anonymous")

            duration_ms = round((time.monotonic() - t0) * 1000, 1)

            final_answer = reasoning_ctx.final_answer
            confidence   = reasoning_ctx.confidence
            strategy     = (
                reasoning_ctx.synthesis_result.strategy.value
                if reasoning_ctx.synthesis_result else "unknown"
            )

            if not final_answer:
                return self._make_result(
                    task       = task,
                    success    = False,
                    error      = "Le pipeline de raisonnement n'a produit aucune réponse.",
                    duration_ms= duration_ms,
                )

            return self._make_result(
                task       = task,
                success    = True,
                output     = final_answer,
                duration_ms= duration_ms,
                confidence = confidence,
                metadata   = {
                    "strategy"      : strategy,
                    "hypotheses"    : len(reasoning_ctx.hypotheses),
                    "evidence_count": len(reasoning_ctx.evidence),
                    "pipeline_steps": len(reasoning_ctx.reasoning_trace),
                },
            )

        except Exception as exc:
            duration_ms = round((time.monotonic() - t0) * 1000, 1)
            return self._make_result(
                task       = task,
                success    = False,
                error      = str(exc),
                duration_ms= duration_ms,
            )

    async def is_available(self) -> bool:
        """Toujours disponible — le pipeline fonctionne avec ou sans LLM."""
        return True

    async def health(self) -> dict[str, Any]:
        base = await super().health()
        base["pipeline"] = "Phase 3.1→3.5 (QuestionAnalyzer→Synthesizer)"
        return base

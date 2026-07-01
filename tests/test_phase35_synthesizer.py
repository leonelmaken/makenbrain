"""Tests Phase 3.5 — Synthesizer.

Couvre :
    1.  Modèles Phase 3.5 (SynthesisStrategy, SynthesisTone, SynthesisResult).
    2.  _select_tone() — seuils, valeurs limites.
    3.  _get_selected_hypothesis_content() — sélection, fallbacks.
    4.  _select_top_evidence() — tri, limite, sources prioritaires.
    5.  _template_synthesis() — ton affirmative, balanced, cautious ; gaps,
        risques, type de question.
    6.  _build_llm_prompt() — structure, contenu, sections optionnelles.
    7.  synthesize() — stratégie LLM (mock), fallback template, ctx enrichi,
        trace, dégradation gracieuse.
    8.  SynthesizerAgent — protocole run(), délégation.
    9.  Pipeline intégration (QA → HE → EE → DE → Synthesizer) complet.
    10. Régressions Phase 3.0 → 3.4 — aucun changement de comportement.
    11. Stabilité (déterminisme — mêmes entrées, même résultat).

Contraintes :
    - Aucun appel Ollama, réseau ou Supabase.
    - Tous les tests isolés (pas d'état partagé mutable).
    - asyncio.run() pour les coroutines.

Commandes :
    pytest tests/test_phase35_synthesizer.py -v
    pytest -q
"""
from __future__ import annotations

import asyncio
from typing import Optional
from unittest.mock import patch, AsyncMock

import pytest

from models.reasoning import (
    DecisionCandidate,
    DecisionReason,
    DecisionResult,
    DecisionScore,
    Evidence,
    EvidenceEvaluation,
    EvidenceRelation,
    EvidenceSourceType,
    EvidenceType,
    Hypothesis,
    QuestionAnalysis,
    QuestionType,
    ReasoningRequest,
    SynthesisResult,
    SynthesisStrategy,
    SynthesisTone,
)
from core.reasoning.context import ReasoningContext
from core.reasoning.synthesizer import (
    SynthesizerAgent,
    _build_llm_prompt,
    _get_selected_hypothesis_content,
    _select_tone,
    _select_top_evidence,
    _template_synthesis,
    synthesize,
)
from core.reasoning.reasoning_engine import (
    DEFAULT_PIPELINE,
    context_to_response,
    run_reasoning,
)


# ── LLM mocks ─────────────────────────────────────────────────────────────────
#
# Ne jamais passer llm_generate=None dans les tests unitaires : le code ferait
# alors un import lazy de core.llm.generate et tenterait de joindre Ollama,
# ce qui ralentit chaque test d'une durée égale au timeout réseau.
#
# Deux helpers :
#   _fast_fail_llm  : échoue instantanément → force le fallback template.
#   _fast_ok_llm    : retourne une réponse mock immédiatement → teste la branche LLM.

async def _fast_fail_llm(prompt: str, **kwargs) -> str:
    """Mock LLM qui échoue instantanément — force le fallback template."""
    raise RuntimeError("LLM simulé indisponible (test isolé)")


async def _fast_ok_llm(prompt: str, **kwargs) -> str:
    """Mock LLM qui répond immédiatement — teste la branche LLM."""
    return "Réponse générée par le LLM en mode mock."


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_request(question: str = "Quelle est la meilleure architecture pour une API ?") -> ReasoningRequest:
    return ReasoningRequest(question=question, include_trace=True)


def _make_context(
    question      : str  = "Test question",
    confidence    : float = 0.80,
    with_decision : bool  = True,
    with_evidence : bool  = True,
    with_gaps     : bool  = False,
) -> ReasoningContext:
    """Construit un contexte de test complet."""
    request = _make_request(question)
    ctx = ReasoningContext(request=request, user_id="test_user")
    ctx.initialize_trace()

    ctx.analysis = QuestionAnalysis(
        original_question  = question,
        question_type      = QuestionType.ANALYTICAL,
        complexity_score   = 0.7,
        domain             = "technique",
        risk_level         = "medium",
        sub_questions      = ["Sous-question 1"],
        key_entities       = ["API"],
        key_concepts       = ["architecture", "API"],
        requires_memory    = True,
        confidence         = 0.8,
    )

    h1 = Hypothesis(
        hypothesis_id = "hyp_001",
        content       = "L'architecture hexagonale est la plus adaptée.",
        initial_score = 0.85,
        justification = "Séparation stricte des couches.",
        strategy_name = "llm",
    )
    h2 = Hypothesis(
        hypothesis_id = "hyp_002",
        content       = "Une architecture microservices est préférable.",
        initial_score = 0.60,
        justification = "Scalabilité native.",
        strategy_name = "decomposition",
    )
    ctx.hypotheses = [h1, h2]

    if with_evidence:
        ctx.evidence = [
            Evidence(
                evidence_id     = "ev_001",
                content         = "L'architecture hexagonale facilite les tests unitaires.",
                source_type     = EvidenceSourceType.VECTOR_MEMORY,
                source_ref      = "doc_001",
                relevance_score = 0.90,
                relations       = {"hyp_001": EvidenceRelation.SUPPORTS},
                credibility_score = 0.85,
                evidence_type   = EvidenceType.FACT,
            ),
            Evidence(
                evidence_id     = "ev_002",
                content         = "La mémoire utilisateur confirme l'usage de FastAPI.",
                source_type     = EvidenceSourceType.USER_MEMORY,
                source_ref      = "user_mem_001",
                relevance_score = 0.80,
                relations       = {"hyp_001": EvidenceRelation.SUPPORTS},
                credibility_score = 0.90,
                evidence_type   = EvidenceType.FACT,
            ),
            Evidence(
                evidence_id     = "ev_003",
                content         = "Les microservices ajoutent de la complexité opérationnelle.",
                source_type     = EvidenceSourceType.NEURON_GRAPH,
                source_ref      = "graph_001",
                relevance_score = 0.70,
                relations       = {"hyp_002": EvidenceRelation.CONTRADICTS},
                credibility_score = 0.75,
                evidence_type   = EvidenceType.FACT,
            ),
        ]
        ctx.evidence_quality_score = 0.82

    gaps = ["Documentation des performances en charge manquante."] if with_gaps else []
    ctx.evaluation = EvidenceEvaluation(
        total_evidence_count  = len(ctx.evidence),
        contradictions        = [],
        knowledge_gaps        = gaps,
        overall_quality_score = 0.82,
        best_hypothesis_id    = "hyp_001",
        support_scores        = {"hyp_001": 0.85, "hyp_002": 0.40},
    )

    if with_decision:
        selected_score = DecisionScore(
            evidence_score             = 0.85,
            confidence                 = 0.80,
            contradiction_penalty      = 0.0,
            missing_information_penalty = 0.0,
            memory_bonus               = 0.20,
            graph_bonus                = 0.0,
            global_score               = 0.78,
        )
        ctx.decision = DecisionResult(
            candidates = [
                DecisionCandidate(
                    hypothesis_id = "hyp_001",
                    content       = "L'architecture hexagonale est la plus adaptée.",
                    score         = selected_score,
                    selected      = True,
                ),
                DecisionCandidate(
                    hypothesis_id = "hyp_002",
                    content       = "Une architecture microservices est préférable.",
                    score         = DecisionScore(global_score=0.40),
                    selected      = False,
                ),
            ],
            selected_hypothesis_id = "hyp_001",
            reason = DecisionReason(
                selection_reasons   = [
                    "Score global le plus élevé (0.78).",
                    "Soutenue par la mémoire utilisateur.",
                ],
                rejection_reasons   = {
                    "hyp_002": ["Score global inférieur (0.40 contre 0.78)."],
                },
                missing_information = [],
                residual_risks      = [
                    "Vérification recommandée en contexte de production."
                ],
            ),
            overall_confidence = confidence,
            decision_quality   = "high" if confidence >= 0.75 else "medium",
        )

    ctx.confidence = confidence
    return ctx


# ── 1. Modèles Phase 3.5 ──────────────────────────────────────────────────────

class TestPhase35Models:

    def test_synthesis_strategy_values(self):
        assert SynthesisStrategy.LLM.value      == "llm"
        assert SynthesisStrategy.TEMPLATE.value == "template"

    def test_synthesis_tone_values(self):
        assert SynthesisTone.AFFIRMATIVE.value == "affirmative"
        assert SynthesisTone.BALANCED.value    == "balanced"
        assert SynthesisTone.CAUTIOUS.value    == "cautious"

    def test_synthesis_result_defaults(self):
        result = SynthesisResult()
        assert result.strategy                  == SynthesisStrategy.TEMPLATE
        assert result.tone                      == SynthesisTone.CAUTIOUS
        assert result.sources_cited             == 0
        assert result.has_uncertainty_statement is False
        assert result.has_gap_statement         is False
        assert result.has_risk_warning          is False
        assert result.llm_prompt_tokens         == 0

    def test_synthesis_result_fully_populated(self):
        result = SynthesisResult(
            strategy                  = SynthesisStrategy.LLM,
            tone                      = SynthesisTone.AFFIRMATIVE,
            sources_cited             = 3,
            has_uncertainty_statement = False,
            has_gap_statement         = True,
            has_risk_warning          = False,
            llm_prompt_tokens         = 512,
        )
        assert result.strategy                  == SynthesisStrategy.LLM
        assert result.tone                      == SynthesisTone.AFFIRMATIVE
        assert result.sources_cited             == 3
        assert result.has_gap_statement         is True
        assert result.llm_prompt_tokens         == 512

    def test_synthesis_result_in_context(self):
        ctx = _make_context()
        assert ctx.synthesis_result is None  # avant synthèse
        ctx.synthesis_result = SynthesisResult(strategy=SynthesisStrategy.LLM)
        assert ctx.synthesis_result.strategy == SynthesisStrategy.LLM

    def test_context_version_is_35(self):
        ctx = _make_context()
        assert ctx.context_version == "3.5"


# ── 2. _select_tone ────────────────────────────────────────────────────────────

class TestSelectTone:

    def test_affirmative_at_075(self):
        assert _select_tone(0.75) == SynthesisTone.AFFIRMATIVE

    def test_affirmative_at_1(self):
        assert _select_tone(1.0) == SynthesisTone.AFFIRMATIVE

    def test_affirmative_just_above_threshold(self):
        assert _select_tone(0.76) == SynthesisTone.AFFIRMATIVE

    def test_balanced_at_060(self):
        assert _select_tone(0.60) == SynthesisTone.BALANCED

    def test_balanced_just_below_075(self):
        assert _select_tone(0.74) == SynthesisTone.BALANCED

    def test_cautious_at_059(self):
        assert _select_tone(0.59) == SynthesisTone.CAUTIOUS

    def test_cautious_at_zero(self):
        assert _select_tone(0.0) == SynthesisTone.CAUTIOUS

    def test_threshold_boundary_060(self):
        """0.60 est BALANCED (pas CAUTIOUS)."""
        assert _select_tone(0.60) == SynthesisTone.BALANCED

    def test_threshold_boundary_075(self):
        """0.75 est AFFIRMATIVE (pas BALANCED)."""
        assert _select_tone(0.75) == SynthesisTone.AFFIRMATIVE


# ── 3. _get_selected_hypothesis_content ───────────────────────────────────────

class TestGetSelectedHypothesisContent:

    def test_returns_selected_hypothesis(self):
        ctx = _make_context()
        content = _get_selected_hypothesis_content(ctx)
        assert "hexagonale" in content

    def test_fallback_to_first_hypothesis_when_no_decision(self):
        ctx = _make_context(with_decision=False)
        content = _get_selected_hypothesis_content(ctx)
        assert content == ctx.hypotheses[0].content

    def test_no_hypotheses_returns_fallback_text(self):
        ctx = _make_context()
        ctx.hypotheses = []
        ctx.decision   = None
        content = _get_selected_hypothesis_content(ctx)
        assert "Aucune conclusion" in content

    def test_unknown_selected_id_falls_back_to_first(self):
        ctx = _make_context()
        ctx.decision = ctx.decision.model_copy(
            update={"selected_hypothesis_id": "unknown_id"}
        )
        content = _get_selected_hypothesis_content(ctx)
        assert content == ctx.hypotheses[0].content


# ── 4. _select_top_evidence ────────────────────────────────────────────────────

class TestSelectTopEvidence:

    def test_returns_at_most_max_evidence(self):
        ctx = _make_context()
        result = _select_top_evidence(ctx)
        assert len(result) <= 4  # _MAX_EVIDENCE_IN_PROMPT

    def test_user_memory_first(self):
        ctx = _make_context()
        result = _select_top_evidence(ctx)
        # USER_MEMORY doit être en premier
        assert result[0].source_type == EvidenceSourceType.USER_MEMORY

    def test_neuron_graph_before_vector_memory(self):
        ctx = _make_context()
        result = _select_top_evidence(ctx)
        types = [ev.source_type for ev in result]
        # USER_MEMORY d'abord, puis NEURON_GRAPH, puis le reste
        if EvidenceSourceType.NEURON_GRAPH in types and EvidenceSourceType.VECTOR_MEMORY in types:
            assert types.index(EvidenceSourceType.NEURON_GRAPH) < types.index(EvidenceSourceType.VECTOR_MEMORY)

    def test_empty_evidence(self):
        ctx = _make_context(with_evidence=False)
        assert _select_top_evidence(ctx) == []


# ── 5. _template_synthesis ────────────────────────────────────────────────────

class TestTemplateSynthesis:

    def test_affirmative_tone_starts_with_d_apres(self):
        ctx    = _make_context(confidence=0.85)
        result = _template_synthesis(ctx, SynthesisTone.AFFIRMATIVE)
        assert result.lower().startswith("d'après")

    def test_balanced_tone_mentions_sur_la_base(self):
        ctx    = _make_context(confidence=0.65)
        result = _template_synthesis(ctx, SynthesisTone.BALANCED)
        assert "sur la base" in result.lower()

    def test_cautious_tone_mentions_verification(self):
        ctx    = _make_context(confidence=0.40)
        result = _template_synthesis(ctx, SynthesisTone.CAUTIOUS)
        assert "vérification" in result.lower() or "limitée" in result.lower()

    def test_evidence_count_mentioned(self):
        ctx    = _make_context(with_evidence=True)
        result = _template_synthesis(ctx, SynthesisTone.AFFIRMATIVE)
        assert "3" in result  # 3 preuves dans le fixture

    def test_no_evidence_no_count_mention(self):
        ctx    = _make_context(with_evidence=False)
        result = _template_synthesis(ctx, SynthesisTone.AFFIRMATIVE)
        assert "élément" not in result

    def test_gaps_mentioned_when_present(self):
        ctx    = _make_context(with_gaps=True)
        result = _template_synthesis(ctx, SynthesisTone.BALANCED)
        assert "documentation" in result.lower() or "manquante" in result.lower()

    def test_no_gaps_when_absent(self):
        ctx    = _make_context(with_gaps=False)
        result = _template_synthesis(ctx, SynthesisTone.AFFIRMATIVE)
        assert "lacune" not in result.lower() and "manquante" not in result.lower()

    def test_non_empty_result(self):
        ctx    = _make_context()
        result = _template_synthesis(ctx, SynthesisTone.BALANCED)
        assert len(result) > 20

    def test_analytical_question_type_note(self):
        ctx    = _make_context()
        result = _template_synthesis(ctx, SynthesisTone.AFFIRMATIVE)
        assert "causalité" in result.lower()

    def test_no_internal_jargon(self):
        """Vérifier que les termes internes ne sont pas exposés."""
        ctx    = _make_context()
        result = _template_synthesis(ctx, SynthesisTone.AFFIRMATIVE)
        assert "global_score" not in result
        assert "pipeline"     not in result.lower()


# ── 6. _build_llm_prompt ──────────────────────────────────────────────────────

class TestBuildLLMPrompt:

    def test_contains_original_question(self):
        ctx    = _make_context("Quelle est la meilleure architecture ?")
        prompt = _build_llm_prompt(ctx, SynthesisTone.AFFIRMATIVE)
        assert "Quelle est la meilleure architecture ?" in prompt

    def test_contains_selected_hypothesis(self):
        ctx    = _make_context()
        prompt = _build_llm_prompt(ctx, SynthesisTone.AFFIRMATIVE)
        assert "hexagonale" in prompt

    def test_contains_domain(self):
        ctx    = _make_context()
        prompt = _build_llm_prompt(ctx, SynthesisTone.AFFIRMATIVE)
        assert "technique" in prompt

    def test_contains_confidence_percentage(self):
        ctx    = _make_context(confidence=0.80)
        prompt = _build_llm_prompt(ctx, SynthesisTone.AFFIRMATIVE)
        assert "80%" in prompt

    def test_affirmative_instruction_in_prompt(self):
        ctx    = _make_context(confidence=0.85)
        prompt = _build_llm_prompt(ctx, SynthesisTone.AFFIRMATIVE)
        assert "affirmative" in prompt.lower() or "directe" in prompt.lower()

    def test_cautious_instruction_in_prompt(self):
        ctx    = _make_context(confidence=0.40)
        prompt = _build_llm_prompt(ctx, SynthesisTone.CAUTIOUS)
        assert "prudence" in prompt.lower() or "limites" in prompt.lower()

    def test_gaps_section_present_when_gaps_exist(self):
        ctx    = _make_context(with_gaps=True)
        prompt = _build_llm_prompt(ctx, SynthesisTone.BALANCED)
        assert "documentation" in prompt.lower() or "manquante" in prompt.lower()

    def test_no_internal_field_names_exposed(self):
        ctx    = _make_context()
        prompt = _build_llm_prompt(ctx, SynthesisTone.AFFIRMATIVE)
        assert "global_score"    not in prompt
        assert "hypothesis_id"   not in prompt
        assert "evidence_score"  not in prompt

    def test_prompt_is_non_empty_string(self):
        ctx    = _make_context()
        prompt = _build_llm_prompt(ctx, SynthesisTone.BALANCED)
        assert isinstance(prompt, str) and len(prompt) > 100


# ── 7. synthesize() ───────────────────────────────────────────────────────────

class TestSynthesize:

    def _run(self, coro):
        return asyncio.run(coro)

    def test_template_fallback_when_no_llm(self):
        ctx        = _make_context()
        result_ctx = self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        assert result_ctx.final_answer is not None
        assert len(result_ctx.final_answer) > 10
        assert result_ctx.synthesis_result is not None
        assert result_ctx.synthesis_result.strategy == SynthesisStrategy.TEMPLATE

    def test_llm_strategy_when_generate_provided(self):
        ctx        = _make_context()
        result_ctx = self._run(synthesize(ctx, llm_generate=_fast_ok_llm))
        assert result_ctx.final_answer == "Réponse générée par le LLM en mode mock."
        assert result_ctx.synthesis_result.strategy == SynthesisStrategy.LLM

    def test_fallback_to_template_when_llm_raises(self):
        ctx        = _make_context()
        result_ctx = self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        assert result_ctx.final_answer is not None
        assert result_ctx.synthesis_result.strategy == SynthesisStrategy.TEMPLATE

    def test_fallback_to_template_when_llm_returns_empty(self):
        async def empty_llm(prompt: str, **kwargs) -> str:
            return "   "

        ctx        = _make_context()
        result_ctx = self._run(synthesize(ctx, llm_generate=empty_llm))
        assert result_ctx.synthesis_result.strategy == SynthesisStrategy.TEMPLATE

    def test_synthesis_result_populated(self):
        ctx        = _make_context()
        result_ctx = self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        sr = result_ctx.synthesis_result
        assert isinstance(sr, SynthesisResult)
        assert sr.tone == SynthesisTone.AFFIRMATIVE  # confidence=0.80 → affirmative

    def test_synthesis_result_tone_balanced(self):
        ctx        = _make_context(confidence=0.65)
        result_ctx = self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        assert result_ctx.synthesis_result.tone == SynthesisTone.BALANCED

    def test_synthesis_result_tone_cautious(self):
        ctx        = _make_context(confidence=0.40)
        result_ctx = self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        assert result_ctx.synthesis_result.tone == SynthesisTone.CAUTIOUS

    def test_trace_step_added(self):
        ctx        = _make_context()
        result_ctx = self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        step_names = [s.step_name for s in result_ctx.trace.steps]
        assert "synthesizer" in step_names

    def test_trace_step_success(self):
        ctx        = _make_context()
        result_ctx = self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        synth_step = next(s for s in result_ctx.trace.steps if s.step_name == "synthesizer")
        assert synth_step.success is True

    def test_has_uncertainty_false_for_affirmative(self):
        ctx        = _make_context(confidence=0.85)
        result_ctx = self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        assert result_ctx.synthesis_result.has_uncertainty_statement is False

    def test_has_uncertainty_true_for_balanced(self):
        ctx        = _make_context(confidence=0.65)
        result_ctx = self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        assert result_ctx.synthesis_result.has_uncertainty_statement is True

    def test_has_gap_statement_true_when_gaps_exist(self):
        ctx        = _make_context(with_gaps=True)
        result_ctx = self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        assert result_ctx.synthesis_result.has_gap_statement is True

    def test_has_gap_statement_false_when_no_gaps(self):
        ctx        = _make_context(with_gaps=False)
        result_ctx = self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        assert result_ctx.synthesis_result.has_gap_statement is False

    def test_sources_cited_count(self):
        ctx        = _make_context(with_evidence=True)
        result_ctx = self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        # 3 types distincts dans le fixture : VECTOR_MEMORY, USER_MEMORY, NEURON_GRAPH
        assert result_ctx.synthesis_result.sources_cited == 3

    def test_no_evidence_sources_cited_zero(self):
        ctx        = _make_context(with_evidence=False)
        result_ctx = self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        assert result_ctx.synthesis_result.sources_cited == 0

    def test_context_not_mutated_before_synthesis(self):
        """Les étapes précédentes ne sont pas altérées par le Synthesizer."""
        ctx                = _make_context()
        original_hyp_count = len(ctx.hypotheses)
        original_ev_count  = len(ctx.evidence)
        self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        assert len(ctx.hypotheses) == original_hyp_count
        assert len(ctx.evidence)   == original_ev_count

    def test_no_hypotheses_still_produces_answer(self):
        ctx            = _make_context()
        ctx.hypotheses = []
        ctx.decision   = None
        result_ctx     = self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        assert result_ctx.final_answer is not None
        assert len(result_ctx.final_answer) > 5

    def test_llm_prompt_tokens_zero_for_template(self):
        ctx        = _make_context()
        result_ctx = self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        assert result_ctx.synthesis_result.llm_prompt_tokens == 0

    def test_llm_prompt_tokens_nonzero_for_llm(self):
        ctx        = _make_context()
        result_ctx = self._run(synthesize(ctx, llm_generate=_fast_ok_llm))
        assert result_ctx.synthesis_result.llm_prompt_tokens > 0


# ── 8. SynthesizerAgent — protocole ──────────────────────────────────────────

class TestSynthesizerAgent:

    def _run(self, coro):
        return asyncio.run(coro)

    def test_agent_name(self):
        agent = SynthesizerAgent()
        assert agent.name == "synthesizer"

    def test_agent_run_returns_context(self):
        """L'agent délègue à synthesize() — on patch core.llm pour éviter Ollama."""
        ctx   = _make_context()
        agent = SynthesizerAgent()
        with patch("core.llm.generate", new=_fast_fail_llm):
            result = self._run(agent.run(ctx))
        assert isinstance(result, ReasoningContext)

    def test_agent_run_populates_final_answer(self):
        ctx    = _make_context()
        result = self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        assert result.final_answer is not None

    def test_agent_is_in_default_pipeline(self):
        agent_names = [a.name for a in DEFAULT_PIPELINE]
        assert "synthesizer" in agent_names
        assert agent_names[-1] == "synthesizer"  # toujours dernière étape

    def test_pipeline_order(self):
        expected_order = [
            "question_analyzer",
            "hypothesis_engine",
            "evidence_engine",
            "decision_engine",
            "synthesizer",
        ]
        actual_order = [a.name for a in DEFAULT_PIPELINE]
        assert actual_order == expected_order


# ── 9. Pipeline intégration complet ──────────────────────────────────────────

class TestPipelineIntegration:
    """Tests end-to-end du pipeline complet.

    core.llm.generate est patché pour éviter tout appel réseau vers Ollama.
    Chaque test tourne en quelques millisecondes même sans serveur LLM.
    """

    _PATCH_TARGET = "core.llm.generate"

    def _run(self, coro):
        return asyncio.run(coro)

    def test_run_reasoning_produces_final_answer(self):
        """Pipeline end-to-end → template fallbacks à chaque étape."""
        request = ReasoningRequest(
            question      = "Quelle est la différence entre REST et GraphQL ?",
            include_trace = True,
        )
        with patch(self._PATCH_TARGET, new=_fast_fail_llm):
            ctx = self._run(run_reasoning(request, user_id="integration_test"))
        assert ctx.final_answer is not None
        assert len(ctx.final_answer) > 10

    def test_run_reasoning_populates_synthesis_result(self):
        request = ReasoningRequest(question="Comment fonctionne Docker ?")
        with patch(self._PATCH_TARGET, new=_fast_fail_llm):
            ctx = self._run(run_reasoning(request, user_id="integration_test"))
        assert ctx.synthesis_result is not None
        assert isinstance(ctx.synthesis_result, SynthesisResult)

    def test_synthesizer_step_in_trace(self):
        request = ReasoningRequest(
            question      = "Comment optimiser une base de données PostgreSQL ?",
            include_trace = True,
        )
        with patch(self._PATCH_TARGET, new=_fast_fail_llm):
            ctx = self._run(run_reasoning(request, user_id="integration_test"))
        step_names = [s.step_name for s in ctx.trace.steps]
        assert "synthesizer" in step_names

    def test_context_to_response_uses_final_answer(self):
        request = ReasoningRequest(question="Qu'est-ce qu'une API REST ?")
        with patch(self._PATCH_TARGET, new=_fast_fail_llm):
            ctx = self._run(run_reasoning(request, user_id="integration_test"))
        response = context_to_response(ctx)
        assert response.answer == ctx.final_answer
        assert "[Synthesizer non implémenté" not in response.answer

    def test_response_has_no_stub_marker(self):
        request = ReasoningRequest(question="Quelle est la meilleure approche TDD ?")
        with patch(self._PATCH_TARGET, new=_fast_fail_llm):
            ctx = self._run(run_reasoning(request, user_id="integration_test"))
        response = context_to_response(ctx)
        assert "[Synthesizer" not in response.answer
        assert "Phase 3.5"    not in response.answer

    def test_reasoning_summary_contains_synthesis_info(self):
        request = ReasoningRequest(question="Comment structurer un projet Python ?")
        with patch(self._PATCH_TARGET, new=_fast_fail_llm):
            ctx = self._run(run_reasoning(request, user_id="integration_test"))
        response = context_to_response(ctx)
        assert "stratégie=" in response.reasoning_summary


# ── 10. Régressions Phase 3.0 → 3.4 ─────────────────────────────────────────

class TestRegressions:
    """Vérifie qu'aucun comportement des phases précédentes n'a été cassé."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_context_has_all_phase_fields(self):
        ctx = _make_context()
        # Phase 3.0 → 3.4
        assert hasattr(ctx, "analysis")
        assert hasattr(ctx, "hypotheses")
        assert hasattr(ctx, "evidence")
        assert hasattr(ctx, "evaluation")
        assert hasattr(ctx, "decision")
        assert hasattr(ctx, "confidence")
        assert hasattr(ctx, "evidence_quality_score")
        # Phase 3.5
        assert hasattr(ctx, "final_answer")
        assert hasattr(ctx, "synthesis_result")

    def test_decision_result_unaffected(self):
        ctx          = _make_context()
        old_decision = ctx.decision
        self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        assert ctx.decision == old_decision

    def test_hypotheses_unaffected(self):
        ctx      = _make_context()
        old_hyps = list(ctx.hypotheses)
        self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        assert ctx.hypotheses == old_hyps

    def test_confidence_unaffected_by_synthesizer(self):
        ctx = _make_context(confidence=0.82)
        self._run(synthesize(ctx, llm_generate=_fast_fail_llm))
        assert ctx.confidence == 0.82

    def test_reasoning_models_backward_compat(self):
        """Les modèles Phase 3.0 peuvent encore être instanciés sans les nouveaux champs."""
        analysis = QuestionAnalysis(
            original_question = "Test",
            question_type     = QuestionType.FACTUAL,
            complexity_score  = 0.3,
        )
        assert analysis.domain     == "general"
        assert analysis.risk_level == "low"
        assert analysis.confidence == 0.0

    def test_synthesis_result_is_none_if_synthesizer_not_run(self):
        ctx = _make_context()
        assert ctx.synthesis_result is None  # pas encore exécuté


# ── 11. Stabilité (déterminisme) ──────────────────────────────────────────────

class TestDeterminism:

    def _run(self, coro):
        return asyncio.run(coro)

    def test_template_synthesis_is_deterministic(self):
        """Mêmes entrées → même résultat (mode template)."""
        ctx1 = _make_context()
        ctx2 = _make_context()
        r1   = _template_synthesis(ctx1, SynthesisTone.AFFIRMATIVE)
        r2   = _template_synthesis(ctx2, SynthesisTone.AFFIRMATIVE)
        assert r1 == r2

    def test_select_tone_is_deterministic(self):
        for conf in [0.0, 0.5, 0.59, 0.60, 0.74, 0.75, 1.0]:
            t1 = _select_tone(conf)
            t2 = _select_tone(conf)
            assert t1 == t2, f"_select_tone non déterministe pour {conf}"

    def test_full_pipeline_deterministic(self):
        """Deux exécutions sans LLM sur la même question → même réponse."""
        request = ReasoningRequest(question="Qu'est-ce qu'un design pattern ?")
        with patch("core.llm.generate", new=_fast_fail_llm):
            ctx1 = self._run(run_reasoning(request, user_id="user_1"))
            ctx2 = self._run(run_reasoning(request, user_id="user_1"))
        assert ctx1.final_answer == ctx2.final_answer

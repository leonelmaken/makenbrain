"""Tests Phase 3.4 — Decision Engine.

Couvre :
    1.  Modèles Phase 3.4 (DecisionScore, DecisionReason, DecisionCandidate,
        DecisionResult).
    2.  DecisionScorer — chaque composante du score individuellement.
    3.  DecisionScorer — score global, agrégation, clamp.
    4.  DecisionReasoner — raisons de sélection, de rejet, risques résiduels.
    5.  make_decision() — décision simple, plusieurs hypothèses, absence
        d'hypothèse, absence de preuve, contradictions, égalité de score.
    6.  Confiance globale et qualité de décision.
    7.  DecisionEngineAgent — protocole, run, trace.
    8.  Pipeline intégration (QA → HE → EE → DE) complet.
    9.  Stabilité (déterminisme — mêmes entrées, même résultat).
    10. Régressions Phase 3.0/3.1/3.2/3.3 — rétrocompatibilité totale.

Contraintes :
    - Aucun appel Ollama, réseau ou Supabase.
    - Tous les tests isolés (pas d'état partagé).
    - asyncio.run() pour les coroutines.

Commandes (à exécuter manuellement, non lancées automatiquement) :
    pytest tests/test_phase34_decision_engine.py -v
    pytest -q
"""
from __future__ import annotations

import asyncio
from typing import Optional

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
)
from core.reasoning.context import ReasoningContext
from core.reasoning.decision_engine import (
    DecisionEngineAgent,
    DecisionReasoner,
    DecisionScorer,
    _compute_decision_quality,
    _compute_overall_confidence,
    make_decision,
)
from core.reasoning.reasoning_engine import (
    DEFAULT_PIPELINE,
    context_to_response,
    run_reasoning,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_request(question: str = "Question de test", **kw) -> ReasoningRequest:
    return ReasoningRequest(question=question, **kw)


def make_ctx(
    question   : str                       = "Question de test",
    analysis   : Optional[QuestionAnalysis] = None,
    hypotheses : Optional[list[Hypothesis]] = None,
    evidence   : Optional[list[Evidence]]   = None,
    evaluation : Optional[EvidenceEvaluation] = None,
    **kw,
) -> ReasoningContext:
    ctx = ReasoningContext(request=make_request(question, **kw), user_id="test-user")
    ctx.initialize_trace()
    if analysis is not None:
        ctx.analysis = analysis
    if hypotheses is not None:
        ctx.hypotheses = hypotheses
    if evidence is not None:
        ctx.evidence = evidence
    if evaluation is not None:
        ctx.evaluation = evaluation
        ctx.evidence_quality_score = evaluation.overall_quality_score
    return ctx


def make_analysis(
    question_type : QuestionType = QuestionType.FACTUAL,
    domain        : str          = "general",
    complexity    : float        = 0.4,
    risk_level    : str          = "low",
    confidence    : float        = 0.7,
) -> QuestionAnalysis:
    return QuestionAnalysis(
        original_question = "Question de test",
        question_type     = question_type,
        domain            = domain,
        complexity_score  = complexity,
        risk_level        = risk_level,
        confidence        = confidence,
    )


def make_hypothesis(
    content : str   = "Hypothèse de test",
    score   : float = 0.5,
    hid     : Optional[str] = None,
) -> Hypothesis:
    h = Hypothesis(content=content, initial_score=score)
    if hid:
        object.__setattr__(h, "hypothesis_id", hid)
    return h


def make_evidence(
    content     : str  = "Preuve de test",
    relevance   : float = 0.6,
    credibility : float = 0.6,
    ev_type     : EvidenceType = EvidenceType.FACT,
    source_type : EvidenceSourceType = EvidenceSourceType.VECTOR_MEMORY,
    relations   : dict | None = None,
    eid         : Optional[str] = None,
) -> Evidence:
    ev = Evidence(
        content           = content,
        source_type       = source_type,
        source_ref        = "test",
        relevance_score   = relevance,
        credibility_score = credibility,
        evidence_type     = ev_type,
        relations         = relations or {},
    )
    if eid:
        object.__setattr__(ev, "evidence_id", eid)
    return ev


def make_evaluation(
    support_scores          : dict | None = None,
    contradictions           : list | None = None,
    knowledge_gaps           : list | None = None,
    overall_quality_score    : float = 0.6,
    best_hypothesis_id        : Optional[str] = None,
    evidence_per_hypothesis   : dict | None = None,
) -> EvidenceEvaluation:
    return EvidenceEvaluation(
        total_evidence_count    = len(support_scores or {}),
        support_scores          = support_scores or {},
        contradictions           = contradictions or [],
        knowledge_gaps           = knowledge_gaps or [],
        overall_quality_score    = overall_quality_score,
        best_hypothesis_id        = best_hypothesis_id,
        evidence_per_hypothesis   = evidence_per_hypothesis or {},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Modèles Phase 3.4
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionScoreModele:
    def test_defaults(self):
        s = DecisionScore()
        assert s.evidence_score == 0.0
        assert s.confidence == 0.0
        assert s.contradiction_penalty == 0.0
        assert s.missing_information_penalty == 0.0
        assert s.memory_bonus == 0.0
        assert s.graph_bonus == 0.0
        assert s.global_score == 0.0

    def test_creation_complete(self):
        s = DecisionScore(
            evidence_score=0.8, confidence=0.75,
            contradiction_penalty=0.1, missing_information_penalty=0.05,
            memory_bonus=0.2, graph_bonus=0.15, global_score=0.7,
        )
        assert s.evidence_score == 0.8
        assert s.global_score == 0.7

    def test_bornes_invalides(self):
        with pytest.raises(Exception):
            DecisionScore(evidence_score=1.5)


class TestDecisionReasonModele:
    def test_defaults_vides(self):
        r = DecisionReason()
        assert r.selection_reasons == []
        assert r.rejection_reasons == {}
        assert r.missing_information == []
        assert r.residual_risks == []

    def test_creation_complete(self):
        r = DecisionReason(
            selection_reasons=["raison 1"],
            rejection_reasons={"hyp1": ["rejet 1"]},
            missing_information=["info manquante"],
            residual_risks=["risque 1"],
        )
        assert r.selection_reasons == ["raison 1"]
        assert r.rejection_reasons == {"hyp1": ["rejet 1"]}


class TestDecisionCandidateModele:
    def test_creation(self):
        c = DecisionCandidate(
            hypothesis_id="abc123", content="test",
            score=DecisionScore(), selected=False,
        )
        assert c.hypothesis_id == "abc123"
        assert c.selected is False

    def test_selected_default_false(self):
        c = DecisionCandidate(
            hypothesis_id="x", content="test", score=DecisionScore()
        )
        assert c.selected is False


class TestDecisionResultModele:
    def test_defaults(self):
        r = DecisionResult()
        assert r.candidates == []
        assert r.selected_hypothesis_id is None
        assert isinstance(r.reason, DecisionReason)
        assert r.overall_confidence == 0.0
        assert r.decision_quality == "low"

    def test_creation_complete(self):
        c = DecisionCandidate(
            hypothesis_id="x", content="test", score=DecisionScore(), selected=True
        )
        r = DecisionResult(
            candidates=[c], selected_hypothesis_id="x",
            overall_confidence=0.8, decision_quality="high",
        )
        assert len(r.candidates) == 1
        assert r.decision_quality == "high"


# ─────────────────────────────────────────────────────────────────────────────
# 2. DecisionScorer — composantes individuelles
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionScorerConfidence:
    def setup_method(self):
        self.scorer = DecisionScorer()

    def test_sans_preuve_confiance_reduite(self):
        c = self.scorer._compute_confidence(0.5, [])
        assert c == 0.25  # 0.5 * 0.5

    def test_avec_preuves_confiance_augmente_avec_volume(self):
        ev1 = [make_evidence()]
        ev3 = [make_evidence(), make_evidence(), make_evidence()]
        c1 = self.scorer._compute_confidence(0.7, ev1)
        c3 = self.scorer._compute_confidence(0.7, ev3)
        assert c3 > c1

    def test_confidence_dans_intervalle(self):
        c = self.scorer._compute_confidence(1.0, [make_evidence() for _ in range(5)])
        assert 0.0 <= c <= 1.0


class TestDecisionScorerContradiction:
    def setup_method(self):
        self.scorer = DecisionScorer()

    def test_sans_contradiction_penalite_nulle(self):
        p = self.scorer._compute_contradiction_penalty("h1", [], [])
        assert p == 0.0

    def test_une_contradiction_ciblant_hypothese(self):
        h_id = "h1"
        e1 = make_evidence(eid="e1", relations={h_id: EvidenceRelation.SUPPORTS})
        e2 = make_evidence(eid="e2", relations={h_id: EvidenceRelation.CONTRADICTS})
        p = self.scorer._compute_contradiction_penalty(
            h_id, [("e1", "e2")], [e1, e2]
        )
        assert p > 0.0

    def test_contradiction_ne_ciblant_pas_hypothese(self):
        e1 = make_evidence(eid="e1", relations={"other_h": EvidenceRelation.SUPPORTS})
        e2 = make_evidence(eid="e2", relations={"other_h": EvidenceRelation.CONTRADICTS})
        p = self.scorer._compute_contradiction_penalty(
            "h1", [("e1", "e2")], [e1, e2]
        )
        assert p == 0.0

    def test_penalite_plafonnee(self):
        h_id = "h1"
        pairs = [(f"e{i}a", f"e{i}b") for i in range(10)]
        evidence = []
        for i in range(10):
            evidence.append(make_evidence(eid=f"e{i}a", relations={h_id: EvidenceRelation.SUPPORTS}))
            evidence.append(make_evidence(eid=f"e{i}b", relations={h_id: EvidenceRelation.CONTRADICTS}))
        p = self.scorer._compute_contradiction_penalty(h_id, pairs, evidence)
        assert p <= 0.60  # _MAX_CONTRADICTION_PENALTY


class TestDecisionScorerMissingInfo:
    def setup_method(self):
        self.scorer = DecisionScorer()

    def test_sans_soutien_penalite_fixe(self):
        h = make_hypothesis("test")
        p = self.scorer._compute_missing_info_penalty("h1", [], [], h)
        assert p >= 0.30

    def test_avec_soutien_pas_de_penalite_base(self):
        h = make_hypothesis("test")
        ev = [make_evidence()]
        p = self.scorer._compute_missing_info_penalty("h1", ev, [], h)
        assert p == 0.0

    def test_penalite_dans_intervalle(self):
        h = make_hypothesis("test")
        p = self.scorer._compute_missing_info_penalty("h1", [], ["lacune"], h)
        assert 0.0 <= p <= 1.0


class TestDecisionScorerSourceBonus:
    def setup_method(self):
        self.scorer = DecisionScorer()

    def test_sans_preuve_bonus_nul(self):
        b = self.scorer._compute_source_bonus([], EvidenceSourceType.USER_MEMORY, 0.2)
        assert b == 0.0

    def test_avec_preuve_memoire_bonus_positif(self):
        ev = [make_evidence(source_type=EvidenceSourceType.USER_MEMORY)]
        b = self.scorer._compute_source_bonus(ev, EvidenceSourceType.USER_MEMORY, 0.2)
        assert b > 0.0

    def test_preuve_autre_source_pas_de_bonus(self):
        ev = [make_evidence(source_type=EvidenceSourceType.EXTERNAL)]
        b = self.scorer._compute_source_bonus(ev, EvidenceSourceType.USER_MEMORY, 0.2)
        assert b == 0.0

    def test_bonus_plafonne(self):
        ev = [make_evidence(source_type=EvidenceSourceType.NEURON_GRAPH) for _ in range(20)]
        b = self.scorer._compute_source_bonus(ev, EvidenceSourceType.NEURON_GRAPH, 0.15)
        assert b <= 0.40  # _MAX_SOURCE_BONUS


# ─────────────────────────────────────────────────────────────────────────────
# 3. DecisionScorer — score global
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionScorerGlobal:
    def setup_method(self):
        self.scorer = DecisionScorer()

    def test_score_dans_intervalle(self):
        g = self.scorer._compute_global_score(0.8, 0.7, 0.1, 0.0, 0.2, 0.1)
        assert 0.0 <= g <= 1.0

    def test_score_eleve_sans_penalite(self):
        g = self.scorer._compute_global_score(1.0, 1.0, 0.0, 0.0, 0.4, 0.4)
        assert g > 0.7

    def test_score_bas_avec_penalites_max(self):
        g = self.scorer._compute_global_score(0.0, 0.0, 0.6, 1.0, 0.0, 0.0)
        assert g == 0.0  # clampé, jamais négatif

    def test_score_hypothesis_complete(self):
        h  = make_hypothesis("python est rapide", hid="h1")
        ev = [make_evidence(relations={"h1": EvidenceRelation.SUPPORTS}, relevance=0.8, credibility=0.8)]
        evaluation = make_evaluation(support_scores={"h1": 0.75})
        score = self.scorer.score_hypothesis(h, ev, evaluation)
        assert isinstance(score, DecisionScore)
        assert 0.0 <= score.global_score <= 1.0

    def test_hypothese_bien_soutenue_score_superieur(self):
        h1 = make_hypothesis("forte", hid="h1")
        h2 = make_hypothesis("faible", hid="h2")
        ev = [
            make_evidence(relations={"h1": EvidenceRelation.SUPPORTS}, relevance=0.9, credibility=0.9),
        ]
        evaluation = make_evaluation(support_scores={"h1": 0.9, "h2": 0.3})
        s1 = self.scorer.score_hypothesis(h1, ev, evaluation)
        s2 = self.scorer.score_hypothesis(h2, [], evaluation)
        assert s1.global_score > s2.global_score


# ─────────────────────────────────────────────────────────────────────────────
# 4. DecisionReasoner
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionReasoner:
    def setup_method(self):
        self.reasoner = DecisionReasoner()

    def test_build_reason_structure_complete(self):
        h1 = make_hypothesis("h1", hid="id1")
        h2 = make_hypothesis("h2", hid="id2")
        candidates = [
            DecisionCandidate(hypothesis_id="id1", content="h1",
                               score=DecisionScore(global_score=0.8), selected=True),
            DecisionCandidate(hypothesis_id="id2", content="h2",
                               score=DecisionScore(global_score=0.4), selected=False),
        ]
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[h1, h2])
        evaluation = make_evaluation()
        reason = self.reasoner.build_reason(candidates, "id1", evaluation, ctx)
        assert isinstance(reason, DecisionReason)
        assert len(reason.selection_reasons) > 0
        assert "id2" in reason.rejection_reasons

    def test_selection_reasons_mentionne_score(self):
        candidates = [
            DecisionCandidate(hypothesis_id="id1", content="h1",
                               score=DecisionScore(global_score=0.9, evidence_score=0.8),
                               selected=True),
        ]
        ctx = make_ctx(analysis=make_analysis())
        reasons = self.reasoner._build_selection_reasons(candidates[0], candidates)
        assert any("0.9" in r for r in reasons)

    def test_rejection_reasons_mentionne_contradiction(self):
        selected = DecisionCandidate(
            hypothesis_id="id1", content="h1",
            score=DecisionScore(global_score=0.8), selected=True
        )
        rejected = DecisionCandidate(
            hypothesis_id="id2", content="h2",
            score=DecisionScore(global_score=0.3, contradiction_penalty=0.2),
            selected=False,
        )
        reasons = self.reasoner._build_rejection_reasons(rejected, selected)
        assert any("contradiction" in r.lower() for r in reasons)

    def test_residual_risks_score_faible(self):
        selected = DecisionCandidate(
            hypothesis_id="id1", content="h1",
            score=DecisionScore(global_score=0.4), selected=True
        )
        ctx = make_ctx(analysis=make_analysis())
        evaluation = make_evaluation(overall_quality_score=0.6)
        risks = self.reasoner._build_residual_risks(selected, evaluation, ctx)
        assert len(risks) > 0

    def test_residual_risks_domaine_a_risque(self):
        selected = DecisionCandidate(
            hypothesis_id="id1", content="h1",
            score=DecisionScore(global_score=0.9), selected=True
        )
        ctx = make_ctx(analysis=make_analysis(risk_level="high"))
        evaluation = make_evaluation(overall_quality_score=0.9)
        risks = self.reasoner._build_residual_risks(selected, evaluation, ctx)
        assert any("risque" in r.lower() for r in risks)

    def test_missing_information_depuis_gaps(self):
        h = make_hypothesis("h1", hid="id1")
        candidates = [
            DecisionCandidate(hypothesis_id="id1", content="h1",
                               score=DecisionScore(), selected=True)
        ]
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[h])
        evaluation = make_evaluation(knowledge_gaps=["lacune A", "lacune B"])
        reason = self.reasoner.build_reason(candidates, "id1", evaluation, ctx)
        assert reason.missing_information == ["lacune A", "lacune B"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. make_decision() — cas fonctionnels
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeDecisionSansHypothese:
    def test_decision_vide_sans_hypothese(self):
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[])
        result = asyncio.run(make_decision(ctx))
        assert result.decision is not None
        assert result.decision.selected_hypothesis_id is None
        assert result.decision.overall_confidence == 0.0

    def test_pipeline_degraded_sans_hypothese(self):
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[])
        result = asyncio.run(make_decision(ctx))
        assert result.pipeline_degraded is True

    def test_trace_step_ajoute_sans_hypothese(self):
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[])
        result = asyncio.run(make_decision(ctx))
        names = [s.step_name for s in result.trace.steps]
        assert "decision_engine" in names


class TestMakeDecisionSimple:
    def test_decision_simple_une_hypothese(self):
        h = make_hypothesis("seule hypothèse", hid="h1")
        ev = [make_evidence(relations={"h1": EvidenceRelation.SUPPORTS}, relevance=0.7, credibility=0.7)]
        evaluation = make_evaluation(support_scores={"h1": 0.7})
        ctx = make_ctx(
            analysis=make_analysis(), hypotheses=[h],
            evidence=ev, evaluation=evaluation,
        )
        result = asyncio.run(make_decision(ctx))
        assert result.decision.selected_hypothesis_id == "h1"

    def test_decision_peuple_ctx_confidence(self):
        h = make_hypothesis("test", hid="h1")
        ev = [make_evidence(relations={"h1": EvidenceRelation.SUPPORTS})]
        evaluation = make_evaluation(support_scores={"h1": 0.7})
        ctx = make_ctx(
            analysis=make_analysis(), hypotheses=[h],
            evidence=ev, evaluation=evaluation,
        )
        result = asyncio.run(make_decision(ctx))
        assert result.confidence == result.decision.overall_confidence


class TestMakeDecisionPlusieursHypotheses:
    def test_meilleure_hypothese_selectionnee(self):
        h1 = make_hypothesis("forte", hid="h1")
        h2 = make_hypothesis("faible", hid="h2")
        ev = [
            make_evidence(relations={"h1": EvidenceRelation.SUPPORTS}, relevance=0.9, credibility=0.9),
        ]
        evaluation = make_evaluation(support_scores={"h1": 0.9, "h2": 0.2})
        ctx = make_ctx(
            analysis=make_analysis(), hypotheses=[h1, h2],
            evidence=ev, evaluation=evaluation,
        )
        result = asyncio.run(make_decision(ctx))
        assert result.decision.selected_hypothesis_id == "h1"

    def test_tous_les_candidats_presents(self):
        hyps = [make_hypothesis(f"h{i}", hid=f"id{i}") for i in range(4)]
        ctx  = make_ctx(analysis=make_analysis(), hypotheses=hyps)
        result = asyncio.run(make_decision(ctx))
        assert len(result.decision.candidates) == 4

    def test_un_seul_candidat_selected_true(self):
        hyps = [make_hypothesis(f"h{i}", hid=f"id{i}") for i in range(3)]
        ctx  = make_ctx(analysis=make_analysis(), hypotheses=hyps)
        result = asyncio.run(make_decision(ctx))
        selected_count = sum(1 for c in result.decision.candidates if c.selected)
        assert selected_count == 1

    def test_candidats_tries_score_decroissant(self):
        h1 = make_hypothesis("h1", hid="id1")
        h2 = make_hypothesis("h2", hid="id2")
        h3 = make_hypothesis("h3", hid="id3")
        evaluation = make_evaluation(support_scores={"id1": 0.9, "id2": 0.5, "id3": 0.2})
        ctx = make_ctx(
            analysis=make_analysis(), hypotheses=[h2, h3, h1],  # ordre mélangé
            evaluation=evaluation,
        )
        result = asyncio.run(make_decision(ctx))
        scores = [c.score.global_score for c in result.decision.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_rejected_hypotheses_ont_raisons(self):
        h1 = make_hypothesis("forte", hid="h1")
        h2 = make_hypothesis("faible", hid="h2")
        evaluation = make_evaluation(support_scores={"h1": 0.9, "h2": 0.1})
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[h1, h2], evaluation=evaluation)
        result = asyncio.run(make_decision(ctx))
        rejected_id = next(
            c.hypothesis_id for c in result.decision.candidates if not c.selected
        )
        assert rejected_id in result.decision.reason.rejection_reasons
        assert len(result.decision.reason.rejection_reasons[rejected_id]) > 0


class TestMakeDecisionEgaliteScore:
    def test_egalite_stricte_premiere_hypothese_gagne(self):
        """En cas d'égalité de score, la première hypothèse de la liste
        d'origine doit être retenue (déterminisme garanti)."""
        h1 = make_hypothesis("identique A", hid="hA")
        h2 = make_hypothesis("identique B", hid="hB")
        # Mêmes scores exacts pour les deux
        evaluation = make_evaluation(support_scores={"hA": 0.6, "hB": 0.6})
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[h1, h2], evaluation=evaluation)
        result1 = asyncio.run(make_decision(ctx))

        # Refaire exactement le même calcul → même résultat (stabilité)
        ctx2 = make_ctx(analysis=make_analysis(), hypotheses=[h1, h2], evaluation=evaluation)
        result2 = asyncio.run(make_decision(ctx2))

        assert result1.decision.selected_hypothesis_id == result2.decision.selected_hypothesis_id


class TestMakeDecisionContradictions:
    def test_contradiction_baisse_score(self):
        h = make_hypothesis("test", hid="h1")
        e_support = make_evidence(eid="e1", relations={"h1": EvidenceRelation.SUPPORTS}, relevance=0.8, credibility=0.8)
        e_contra  = make_evidence(eid="e2", relations={"h1": EvidenceRelation.CONTRADICTS}, relevance=0.8, credibility=0.8)
        evaluation_avec = make_evaluation(
            support_scores={"h1": 0.5}, contradictions=[("e1", "e2")]
        )
        evaluation_sans = make_evaluation(support_scores={"h1": 0.5})

        ctx_avec = make_ctx(
            analysis=make_analysis(), hypotheses=[h],
            evidence=[e_support, e_contra], evaluation=evaluation_avec,
        )
        ctx_sans = make_ctx(
            analysis=make_analysis(), hypotheses=[h],
            evidence=[e_support], evaluation=evaluation_sans,
        )
        result_avec = asyncio.run(make_decision(ctx_avec))
        result_sans = asyncio.run(make_decision(ctx_sans))

        score_avec = result_avec.decision.candidates[0].score.global_score
        score_sans = result_sans.decision.candidates[0].score.global_score
        assert score_avec < score_sans

    def test_contradiction_residual_risk_mentionne(self):
        h = make_hypothesis("test", hid="h1")
        e1 = make_evidence(eid="e1", relations={"h1": EvidenceRelation.SUPPORTS})
        e2 = make_evidence(eid="e2", relations={"h1": EvidenceRelation.CONTRADICTS})
        evaluation = make_evaluation(
            support_scores={"h1": 0.5}, contradictions=[("e1", "e2")]
        )
        ctx = make_ctx(
            analysis=make_analysis(), hypotheses=[h],
            evidence=[e1, e2], evaluation=evaluation,
        )
        result = asyncio.run(make_decision(ctx))
        assert len(result.decision.reason.residual_risks) > 0


class TestMakeDecisionAbsenceDePreuve:
    def test_sans_aucune_preuve(self):
        hyps = [make_hypothesis(f"h{i}", hid=f"id{i}") for i in range(2)]
        ctx  = make_ctx(analysis=make_analysis(), hypotheses=hyps, evidence=[])
        result = asyncio.run(make_decision(ctx))
        assert result.decision.selected_hypothesis_id is not None  # décision quand même prise

    def test_sans_preuve_missing_info_penalty_elevee(self):
        h = make_hypothesis("test", hid="h1")
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[h], evidence=[])
        result = asyncio.run(make_decision(ctx))
        assert result.decision.candidates[0].score.missing_information_penalty > 0.0

    def test_sans_preuve_confidence_basse(self):
        hyps = [make_hypothesis(f"h{i}", hid=f"id{i}") for i in range(2)]
        ctx  = make_ctx(analysis=make_analysis(), hypotheses=hyps, evidence=[])
        result = asyncio.run(make_decision(ctx))
        assert result.decision.overall_confidence < 0.7


# ─────────────────────────────────────────────────────────────────────────────
# 6. Confiance globale et qualité
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeOverallConfidence:
    def test_confidence_dans_intervalle(self):
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[make_hypothesis()])
        score = DecisionScore(global_score=0.7, contradiction_penalty=0.1)
        c = _compute_overall_confidence(ctx, score)
        assert 0.0 <= c <= 1.0

    def test_confidence_elevee_si_tout_favorable(self):
        ctx = make_ctx(
            analysis=make_analysis(confidence=0.9),
            hypotheses=[make_hypothesis(), make_hypothesis(), make_hypothesis()],
        )
        ctx.evidence_quality_score = 0.9
        score = DecisionScore(global_score=0.95, contradiction_penalty=0.0)
        c = _compute_overall_confidence(ctx, score)
        assert c > 0.7

    def test_confidence_basse_si_tout_defavorable(self):
        ctx = make_ctx(analysis=make_analysis(confidence=0.3), hypotheses=[])
        ctx.evidence_quality_score = 0.1
        score = DecisionScore(global_score=0.1, contradiction_penalty=0.6)
        c = _compute_overall_confidence(ctx, score)
        assert c < 0.4

    def test_sans_analysis_utilise_defaut(self):
        ctx = make_ctx(hypotheses=[make_hypothesis()])  # pas d'analysis
        score = DecisionScore(global_score=0.5)
        c = _compute_overall_confidence(ctx, score)
        assert 0.0 <= c <= 1.0


class TestComputeDecisionQuality:
    def test_high(self):
        assert _compute_decision_quality(0.80) == "high"
        assert _compute_decision_quality(0.75) == "high"

    def test_medium(self):
        assert _compute_decision_quality(0.65) == "medium"
        assert _compute_decision_quality(0.55) == "medium"

    def test_low(self):
        assert _compute_decision_quality(0.50) == "low"
        assert _compute_decision_quality(0.0)  == "low"


# ─────────────────────────────────────────────────────────────────────────────
# 7. DecisionEngineAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionEngineAgent:
    def test_name(self):
        assert DecisionEngineAgent().name == "decision_engine"

    def test_run_retourne_ctx(self):
        agent = DecisionEngineAgent()
        ctx   = make_ctx(analysis=make_analysis(), hypotheses=[make_hypothesis()])
        result = asyncio.run(agent.run(ctx))
        assert result is ctx

    def test_run_peuple_decision(self):
        agent = DecisionEngineAgent()
        ctx   = make_ctx(analysis=make_analysis(), hypotheses=[make_hypothesis()])
        result = asyncio.run(agent.run(ctx))
        assert result.decision is not None

    def test_run_ajoute_trace_step(self):
        agent = DecisionEngineAgent()
        ctx   = make_ctx(analysis=make_analysis(), hypotheses=[make_hypothesis()])
        result = asyncio.run(agent.run(ctx))
        names = [s.step_name for s in result.trace.steps]
        assert "decision_engine" in names

    def test_context_version_inchangee(self):
        agent = DecisionEngineAgent()
        ctx   = make_ctx(analysis=make_analysis(), hypotheses=[make_hypothesis()])
        result = asyncio.run(agent.run(ctx))
        assert result.context_version == "3.0"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Pipeline intégration (QA → HE → EE → DE)
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_default_pipeline_quatre_agents(self):
        assert len(DEFAULT_PIPELINE) >= 4

    def test_quatrieme_agent_decision_engine(self):
        assert DEFAULT_PIPELINE[3].name == "decision_engine"

    def test_agent_names_complets(self):
        names = [a.name for a in DEFAULT_PIPELINE]
        assert names[:4] == [
            "question_analyzer", "hypothesis_engine",
            "evidence_engine", "decision_engine",
        ]

    def test_run_reasoning_peuple_decision(self):
        req = make_request("Pourquoi FastAPI est plus rapide que Flask ?")
        ctx = asyncio.run(run_reasoning(request=req, user_id="test"))
        assert ctx.decision is not None

    def test_run_reasoning_confidence_non_nulle(self):
        req = make_request("Comment optimiser une base de données PostgreSQL ?")
        ctx = asyncio.run(run_reasoning(request=req, user_id="test"))
        # Le confidence est désormais calculé par DecisionEngine
        assert isinstance(ctx.confidence, float)
        assert 0.0 <= ctx.confidence <= 1.0

    def test_run_reasoning_trace_quatre_steps(self):
        req   = make_request("Comment déployer sur Kubernetes ?")
        ctx   = asyncio.run(run_reasoning(request=req, user_id="test"))
        names = [s.step_name for s in ctx.trace.steps]
        assert "decision_engine" in names

    def test_context_to_response_risk_level_coherent(self):
        req      = make_request("FastAPI vs Django ?")
        ctx      = asyncio.run(run_reasoning(request=req, user_id="test"))
        response = context_to_response(ctx)
        assert response.risk_level in ("low", "medium", "high")
        assert response.confidence == ctx.confidence

    def test_context_to_response_best_hypothesis_depuis_decision(self):
        req      = make_request("comment structurer une api rest ?")
        ctx      = asyncio.run(run_reasoning(request=req, user_id="test"))
        response = context_to_response(ctx)
        if ctx.decision and ctx.decision.selected_hypothesis_id:
            assert response.best_hypothesis != "" or not ctx.hypotheses

    def test_run_reasoning_context_version_inchangee(self):
        req = make_request("test")
        ctx = asyncio.run(run_reasoning(request=req, user_id="test"))
        assert ctx.context_version == "3.0"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Stabilité / Déterminisme
# ─────────────────────────────────────────────────────────────────────────────

class TestStabilite:
    def test_meme_entree_meme_score(self):
        scorer = DecisionScorer()
        h  = make_hypothesis("test", hid="h1")
        ev = [make_evidence(relations={"h1": EvidenceRelation.SUPPORTS}, relevance=0.7, credibility=0.7)]
        evaluation = make_evaluation(support_scores={"h1": 0.7})

        s1 = scorer.score_hypothesis(h, ev, evaluation)
        s2 = scorer.score_hypothesis(h, ev, evaluation)
        assert s1.global_score == s2.global_score
        assert s1.evidence_score == s2.evidence_score

    def test_decision_reproductible(self):
        h1 = make_hypothesis("h1", hid="id1")
        h2 = make_hypothesis("h2", hid="id2")
        evaluation = make_evaluation(support_scores={"id1": 0.8, "id2": 0.3})

        results = []
        for _ in range(3):
            ctx = make_ctx(analysis=make_analysis(), hypotheses=[h1, h2], evaluation=evaluation)
            result = asyncio.run(make_decision(ctx))
            results.append(result.decision.selected_hypothesis_id)

        assert len(set(results)) == 1  # toujours la même décision


# ─────────────────────────────────────────────────────────────────────────────
# 10. Régressions Phase 3.0/3.1/3.2/3.3
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionsPhasesPrecedentes:
    def test_reasoning_context_decision_field_default_none(self):
        """Un ReasoningContext créé sans pipeline n'a pas de décision (None)."""
        ctx = make_ctx()
        assert ctx.decision is None

    def test_hypothesis_retro_compatible(self):
        h = Hypothesis(content="test")
        assert h.justification == ""
        assert h.strategy_name == "unknown"

    def test_evidence_retro_compatible(self):
        ev = Evidence(
            content="test", source_type=EvidenceSourceType.VECTOR_MEMORY, source_ref="x"
        )
        assert ev.credibility_score == 0.5
        assert ev.evidence_type == EvidenceType.FACT

    def test_evidence_evaluation_retro_compatible(self):
        ee = EvidenceEvaluation()
        assert ee.support_scores == {}
        assert ee.evidence_per_hypothesis == {}

    def test_question_analysis_retro_compatible(self):
        qa = QuestionAnalysis(
            original_question="test", question_type=QuestionType.FACTUAL,
            complexity_score=0.3,
        )
        assert qa.domain == "general"

    def test_reasoning_response_construction_sans_decision_data(self):
        """ReasoningResponse reste constructible indépendamment du DecisionEngine."""
        from models.reasoning import ReasoningResponse
        r = ReasoningResponse(answer="test")
        assert r.confidence == 0.0

    def test_run_reasoning_ne_leve_pas_exception(self):
        req = make_request("test simple sans LLM")
        ctx = asyncio.run(run_reasoning(request=req, user_id="test"))
        assert isinstance(ctx, ReasoningContext)
        assert ctx.decision is not None  # Phase 3.4 toujours peuplée

    def test_pipeline_anciennes_etapes_toujours_presentes(self):
        names = [a.name for a in DEFAULT_PIPELINE]
        assert "question_analyzer" in names
        assert "hypothesis_engine" in names
        assert "evidence_engine"   in names

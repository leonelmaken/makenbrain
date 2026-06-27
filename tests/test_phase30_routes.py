"""Tests Sprint 1 — Phase 3.0 Expert Reasoning Engine.

Couvre :
1. Modèles Pydantic (ReasoningRequest, ReasoningResponse, QuestionAnalysis, etc.)
2. ReasoningContext (shared state, initialize_trace, mark_degraded, fallbacks)
3. reasoning_engine (context_to_response, run_reasoning stub, risk_level)
4. Router HTTP (/reasoning/analyze — 401 sans token, 200 avec auth)

Convention du projet :
- Les tests métier utilisent des appels directs avec monkeypatch (pas TestClient).
- TestClient est réservé aux tests de comportement HTTP (auth, codes de retour).
"""
from __future__ import annotations

import asyncio

import pytest

from models.reasoning import (
    Evidence,
    EvidenceEvaluation,
    EvidenceRelation,
    EvidenceSourceType,
    Hypothesis,
    QuestionAnalysis,
    QuestionType,
    ReasoningRequest,
    ReasoningResponse,
    ReasoningStepRecord,
    ReasoningTrace,
)
from core.reasoning.context import ReasoningContext
from core.reasoning.reasoning_engine import (
    _compute_risk_level,
    _build_summary,
    _best_hypothesis,
    context_to_response,
    run_reasoning,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Modèles Pydantic
# ─────────────────────────────────────────────────────────────────────────────

class TestReasoningRequest:
    def test_valide_avec_champs_minimaux(self) -> None:
        req = ReasoningRequest(question="Qu'est-ce que MakenBrain ?")
        assert req.question == "Qu'est-ce que MakenBrain ?"
        assert req.provider == "auto"
        assert req.max_hypotheses == 3
        assert req.max_evidence_per_source == 5
        assert req.relevance_threshold == 0.70
        assert req.include_trace is False

    def test_question_vide_est_invalide(self) -> None:
        with pytest.raises(Exception):
            ReasoningRequest(question="")

    def test_max_hypotheses_hors_borne_est_invalide(self) -> None:
        with pytest.raises(Exception):
            ReasoningRequest(question="test", max_hypotheses=10)

    def test_relevance_threshold_hors_borne_est_invalide(self) -> None:
        with pytest.raises(Exception):
            ReasoningRequest(question="test", relevance_threshold=1.5)


class TestReasoningResponse:
    def test_defaults_non_cassants(self) -> None:
        """Les valeurs par défaut ne doivent pas bloquer les appels existants."""
        r = ReasoningResponse(answer="Réponse de test")
        assert r.confidence == 0.0
        assert r.risk_level == "high"
        assert r.suggested_sources == []
        assert r.hypotheses_considered == []
        assert r.evidence_used == 0
        assert r.trace is None

    def test_suggested_sources_ne_partage_pas_liste_mutable(self) -> None:
        """Chaque ReasoningResponse doit avoir sa propre liste de sources."""
        r1 = ReasoningResponse(answer="A")
        r2 = ReasoningResponse(answer="B")
        r1.suggested_sources.append({"type": "web", "query": "test"})
        assert r2.suggested_sources == []

    def test_knowledge_gaps_ne_partage_pas_liste_mutable(self) -> None:
        r1 = ReasoningResponse(answer="A")
        r2 = ReasoningResponse(answer="B")
        r1.knowledge_gaps.append("gap 1")
        assert r2.knowledge_gaps == []

    def test_trace_presente_quand_fournie(self) -> None:
        trace = ReasoningTrace(question="test")
        r = ReasoningResponse(answer="A", trace=trace)
        assert r.trace is not None
        assert r.trace.question == "test"


class TestQuestionAnalysis:
    def test_requires_reasoning_faux_par_defaut(self) -> None:
        qa = QuestionAnalysis(
            original_question="test",
            question_type=QuestionType.FACTUAL,
            complexity_score=0.3,
        )
        assert qa.requires_reasoning is False

    def test_complexity_score_hors_borne_invalide(self) -> None:
        with pytest.raises(Exception):
            QuestionAnalysis(
                original_question="test",
                question_type=QuestionType.FACTUAL,
                complexity_score=1.5,
            )


class TestHypothesis:
    def test_hypothesis_id_genere_automatiquement(self) -> None:
        h1 = Hypothesis(content="Hypothèse A")
        h2 = Hypothesis(content="Hypothèse B")
        assert h1.hypothesis_id != h2.hypothesis_id
        assert len(h1.hypothesis_id) == 8

    def test_initial_score_defaut(self) -> None:
        h = Hypothesis(content="test")
        assert h.initial_score == 0.5


class TestEvidence:
    def test_evidence_id_genere_automatiquement(self) -> None:
        e1 = Evidence(
            content="preuve 1",
            source_type=EvidenceSourceType.VECTOR_MEMORY,
            source_ref="collection:memories",
        )
        e2 = Evidence(
            content="preuve 2",
            source_type=EvidenceSourceType.NEURON_GRAPH,
            source_ref="graph:node42",
        )
        assert e1.evidence_id != e2.evidence_id

    def test_relations_vides_par_defaut(self) -> None:
        e = Evidence(
            content="test",
            source_type=EvidenceSourceType.USER_MEMORY,
            source_ref="supabase:memories",
        )
        assert e.relations == {}


class TestEvidenceEvaluation:
    def test_defaults_corrects(self) -> None:
        ev = EvidenceEvaluation()
        assert ev.total_evidence_count == 0
        assert ev.contradictions == []
        assert ev.knowledge_gaps == []
        assert ev.overall_quality_score == 0.0
        assert ev.best_hypothesis_id is None

    def test_quality_score_hors_borne_invalide(self) -> None:
        with pytest.raises(Exception):
            EvidenceEvaluation(overall_quality_score=2.0)


class TestReasoningTrace:
    def test_trace_id_genere_automatiquement(self) -> None:
        t1 = ReasoningTrace(question="Q1")
        t2 = ReasoningTrace(question="Q2")
        assert t1.trace_id != t2.trace_id

    def test_steps_vides_par_defaut(self) -> None:
        t = ReasoningTrace(question="test")
        assert t.steps == []
        assert t.total_duration_ms == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. ReasoningContext
# ─────────────────────────────────────────────────────────────────────────────

def _make_request(**kwargs) -> ReasoningRequest:
    defaults = {"question": "Question de test pour Sprint 1"}
    defaults.update(kwargs)
    return ReasoningRequest(**defaults)


class TestReasoningContext:
    def test_context_version_est_3_0(self) -> None:
        ctx = ReasoningContext(request=_make_request(), user_id="user-1")
        assert ctx.context_version == "3.0"

    def test_pipeline_non_degrade_par_defaut(self) -> None:
        ctx = ReasoningContext(request=_make_request(), user_id="user-1")
        assert ctx.pipeline_degraded is False
        assert ctx.errors == []

    def test_initialize_trace_peuple_question(self) -> None:
        ctx = ReasoningContext(request=_make_request(), user_id="user-1")
        ctx.initialize_trace()
        assert ctx.trace.question == "Question de test pour Sprint 1"
        assert ctx.trace.trace_id != ""
        assert ctx.trace.created_at is not None

    def test_initialize_trace_genere_ids_uniques(self) -> None:
        ctx1 = ReasoningContext(request=_make_request(), user_id="user-1")
        ctx2 = ReasoningContext(request=_make_request(), user_id="user-2")
        ctx1.initialize_trace()
        ctx2.initialize_trace()
        assert ctx1.trace.trace_id != ctx2.trace.trace_id

    def test_mark_degraded_positionne_flag(self) -> None:
        ctx = ReasoningContext(request=_make_request(), user_id="user-1")
        ctx.mark_degraded("question_analyzer", "LLM timeout")
        assert ctx.pipeline_degraded is True
        assert len(ctx.errors) == 1
        assert ctx.errors[0]["step"] == "question_analyzer"
        assert ctx.errors[0]["error"] == "LLM timeout"

    def test_mark_degraded_accumule_plusieurs_erreurs(self) -> None:
        ctx = ReasoningContext(request=_make_request(), user_id="user-1")
        ctx.mark_degraded("etape_1", "Erreur A")
        ctx.mark_degraded("etape_2", "Erreur B")
        assert len(ctx.errors) == 2

    def test_analysis_or_fallback_sans_analyse(self) -> None:
        """Sans étape 1, le fallback doit fournir des valeurs cohérentes."""
        ctx = ReasoningContext(request=_make_request(), user_id="user-1")
        assert ctx.analysis is None
        fallback = ctx.analysis_or_fallback
        assert fallback.question_type == QuestionType.FACTUAL
        assert fallback.complexity_score == 0.5
        assert fallback.original_question == "Question de test pour Sprint 1"
        assert len(fallback.sub_questions) == 1

    def test_analysis_or_fallback_avec_analyse_reelle(self) -> None:
        """Avec une vraie analyse, retourne l'analyse, pas le fallback."""
        ctx = ReasoningContext(request=_make_request(), user_id="user-1")
        ctx.analysis = QuestionAnalysis(
            original_question = "Question réelle",
            question_type     = QuestionType.COMPARATIVE,
            complexity_score  = 0.8,
        )
        result = ctx.analysis_or_fallback
        assert result.question_type == QuestionType.COMPARATIVE
        assert result.complexity_score == 0.8

    def test_listes_independantes_entre_contextes(self) -> None:
        """Chaque contexte doit avoir ses propres listes (pas de shared state)."""
        ctx1 = ReasoningContext(request=_make_request(), user_id="user-1")
        ctx2 = ReasoningContext(request=_make_request(), user_id="user-2")
        ctx1.hypotheses.append(Hypothesis(content="H1"))
        assert len(ctx2.hypotheses) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. reasoning_engine — helpers et context_to_response
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskLevel:
    def test_confidence_haute_donne_low(self) -> None:
        assert _compute_risk_level(0.80) == "low"
        assert _compute_risk_level(0.75) == "low"

    def test_confidence_moyenne_donne_medium(self) -> None:
        assert _compute_risk_level(0.65) == "medium"
        assert _compute_risk_level(0.60) == "medium"

    def test_confidence_basse_donne_high(self) -> None:
        assert _compute_risk_level(0.50) == "high"
        assert _compute_risk_level(0.0)  == "high"


class TestBuildSummary:
    def test_summary_sans_analyse_retourne_chaine_non_vide(self) -> None:
        """Phase 3.1+ : le message par défaut ne référence plus le Sprint 1.

        Historique :
            Phase 3.0 : _build_summary retournait 'Pipeline Sprint 1 — agents à implémenter'.
            Phase 3.1 : le pipeline contient QuestionAnalyzerAgent → le message a évolué.
        Décision : mettre à jour le test (comportement intentionnel, pas une régression).
        """
        ctx = ReasoningContext(request=_make_request(), user_id="user-1")
        summary = _build_summary(ctx)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_avec_analyse_inclut_type_et_complexite(self) -> None:
        ctx = ReasoningContext(request=_make_request(), user_id="user-1")
        ctx.analysis = QuestionAnalysis(
            original_question = "test",
            question_type     = QuestionType.ANALYTICAL,
            complexity_score  = 0.72,
        )
        summary = _build_summary(ctx)
        assert "analytical" in summary
        assert "0.72" in summary

    def test_mode_degrade_signale_dans_summary(self) -> None:
        ctx = ReasoningContext(request=_make_request(), user_id="user-1")
        ctx.mark_degraded("etape_1", "erreur")
        summary = _build_summary(ctx)
        assert "dégradé" in summary.lower()


class TestBestHypothesis:
    def test_sans_hypotheses_retourne_chaine_vide(self) -> None:
        ctx = ReasoningContext(request=_make_request(), user_id="user-1")
        assert _best_hypothesis(ctx) == ""

    def test_avec_evaluation_retourne_meilleure_hypothese(self) -> None:
        ctx = ReasoningContext(request=_make_request(), user_id="user-1")
        h1 = Hypothesis(hypothesis_id="aaa11111", content="Hypothèse A")
        h2 = Hypothesis(hypothesis_id="bbb22222", content="Hypothèse B")
        ctx.hypotheses = [h1, h2]
        ctx.evaluation = EvidenceEvaluation(best_hypothesis_id="bbb22222")
        assert _best_hypothesis(ctx) == "Hypothèse B"

    def test_sans_evaluation_retourne_premiere_hypothese(self) -> None:
        ctx = ReasoningContext(request=_make_request(), user_id="user-1")
        ctx.hypotheses = [
            Hypothesis(content="Première"),
            Hypothesis(content="Deuxième"),
        ]
        assert _best_hypothesis(ctx) == "Première"


class TestContextToResponse:
    def test_champs_obligatoires_presents(self) -> None:
        ctx = ReasoningContext(request=_make_request(), user_id="user-1")
        ctx.initialize_trace()
        response = context_to_response(ctx)
        assert response.answer != ""
        assert response.risk_level == "high"   # confidence=0.0 → high
        assert response.provider == "auto"
        assert response.model == "pending"

    def test_trace_absente_si_include_trace_faux(self) -> None:
        ctx = ReasoningContext(
            request=_make_request(include_trace=False),
            user_id="user-1",
        )
        ctx.initialize_trace()
        response = context_to_response(ctx)
        assert response.trace is None

    def test_trace_presente_si_include_trace_vrai(self) -> None:
        ctx = ReasoningContext(
            request=_make_request(include_trace=True),
            user_id="user-1",
        )
        ctx.initialize_trace()
        response = context_to_response(ctx)
        assert response.trace is not None

    def test_evidence_used_reflète_evidence_collectee(self) -> None:
        ctx = ReasoningContext(request=_make_request(), user_id="user-1")
        ctx.initialize_trace()
        ctx.evidence = [
            Evidence(
                content="p1",
                source_type=EvidenceSourceType.VECTOR_MEMORY,
                source_ref="col:memories",
            ),
            Evidence(
                content="p2",
                source_type=EvidenceSourceType.NEURON_GRAPH,
                source_ref="graph:node1",
            ),
        ]
        response = context_to_response(ctx)
        assert response.evidence_used == 2

    def test_knowledge_gaps_depuis_evaluation(self) -> None:
        ctx = ReasoningContext(request=_make_request(), user_id="user-1")
        ctx.initialize_trace()
        ctx.evaluation = EvidenceEvaluation(
            knowledge_gaps=["gap A", "gap B"]
        )
        response = context_to_response(ctx)
        assert response.knowledge_gaps == ["gap A", "gap B"]

    def test_hypotheses_considered_liste_contenu(self) -> None:
        ctx = ReasoningContext(request=_make_request(), user_id="user-1")
        ctx.initialize_trace()
        ctx.hypotheses = [
            Hypothesis(content="H1"),
            Hypothesis(content="H2"),
        ]
        response = context_to_response(ctx)
        assert response.hypotheses_considered == ["H1", "H2"]


class TestRunReasoningStub:
    def test_run_reasoning_retourne_context(self) -> None:
        """Sprint 1 : le pipeline est vide, le contexte doit quand même être retourné."""
        req = _make_request()
        ctx = asyncio.run(run_reasoning(request=req, user_id="user-test"))
        assert isinstance(ctx, ReasoningContext)
        assert ctx.context_version == "3.0"
        assert ctx.user_id == "user-test"

    def test_run_reasoning_initialise_trace(self) -> None:
        req = _make_request()
        ctx = asyncio.run(run_reasoning(request=req, user_id="user-test"))
        assert ctx.trace.question == req.question
        assert ctx.trace.trace_id != ""

    def test_run_reasoning_complete_sans_exception(self) -> None:
        """Phase 3.1+ : le pipeline contient des agents réels.
        Sans core.llm, le QuestionAnalyzer utilise le fallback déterministe
        et marque pipeline_degraded=True. C'est le comportement attendu.
        On vérifie uniquement que run_reasoning se termine sans exception.
        """
        req = _make_request()
        ctx = asyncio.run(run_reasoning(request=req, user_id="user-test"))
        # Le pipeline peut être dégradé (LLM absent en env test) mais doit compléter
        assert isinstance(ctx, ReasoningContext)
        assert ctx.context_version == "3.0"

    def test_run_reasoning_trace_duration_renseignee(self) -> None:
        req = _make_request()
        ctx = asyncio.run(run_reasoning(request=req, user_id="user-test"))
        assert ctx.trace.total_duration_ms >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Router HTTP — comportement auth et structure de réponse
# ─────────────────────────────────────────────────────────────────────────────

class TestReasoningEndpointAuth:
    def test_rejette_requete_sans_token(self, test_client) -> None:
        """Sans Bearer token, l'endpoint doit retourner 401."""
        response = test_client.post(
            "/reasoning/analyze",
            json={"question": "Test sans token"},
        )
        assert response.status_code == 401

    def test_rejette_requete_avec_mauvais_scheme(self, test_client) -> None:
        """Un header Authorization non-Bearer doit aussi être rejeté."""
        response = test_client.post(
            "/reasoning/analyze",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
            json={"question": "Test mauvais scheme"},
        )
        assert response.status_code == 401


class TestReasoningEndpointReponse:
    def test_retourne_200_avec_auth_valide(self, authenticated_client) -> None:
        response = authenticated_client.post(
            "/reasoning/analyze",
            json={"question": "Qu'est-ce que MakenBrain ?"},
        )
        assert response.status_code == 200

    def test_reponse_contient_champs_obligatoires(self, authenticated_client) -> None:
        response = authenticated_client.post(
            "/reasoning/analyze",
            json={"question": "Architecture de MakenBrain"},
        )
        body = response.json()
        champs_obligatoires = [
            "answer", "confidence", "risk_level", "suggested_sources",
            "reasoning_summary", "question_type", "complexity_score",
            "hypotheses_considered", "best_hypothesis", "evidence_used",
            "evidence_quality_score", "knowledge_gaps", "provider", "model",
        ]
        for champ in champs_obligatoires:
            assert champ in body, f"Champ manquant dans la réponse : {champ}"

    def test_trace_absente_par_defaut(self, authenticated_client) -> None:
        response = authenticated_client.post(
            "/reasoning/analyze",
            json={"question": "Test trace absente", "include_trace": False},
        )
        assert response.json()["trace"] is None

    def test_trace_presente_si_demandee(self, authenticated_client) -> None:
        response = authenticated_client.post(
            "/reasoning/analyze",
            json={"question": "Test trace présente", "include_trace": True},
        )
        assert response.json()["trace"] is not None

    def test_422_si_question_vide(self, authenticated_client) -> None:
        response = authenticated_client.post(
            "/reasoning/analyze",
            json={"question": ""},
        )
        assert response.status_code == 422

    def test_422_si_body_absent(self, authenticated_client) -> None:
        response = authenticated_client.post("/reasoning/analyze")
        assert response.status_code == 422

    def test_risk_level_high_sur_pipeline_vide(self, authenticated_client) -> None:
        """Sprint 1 : confiance = 0.0 → risk_level doit être 'high'."""
        response = authenticated_client.post(
            "/reasoning/analyze",
            json={"question": "Test risk level"},
        )
        assert response.json()["risk_level"] == "high"
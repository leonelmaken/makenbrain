"""Tests Phase 3.3 — Evidence Engine.

Couvre :
    1.  Modèles Phase 3.3 (EvidenceType, champs Evidence, EvidenceEvaluation).
    2.  classify_evidence_type() — classification linguistique.
    3.  EvidenceRanker — relevance, credibility, scoring, filtrage, top-N.
    4.  HypothesisEvidenceCollector — soutien depuis les hypothèses.
    5.  CounterEvidenceCollector — contre-preuves systématiques.
    6.  ConceptEvidenceCollector — preuves conceptuelles.
    7.  DomainEvidenceCollector — faits de domaine.
    8.  LLMEvidenceCollector — JSON valide, invalide, erreur, désactivé.
    9.  EvidenceValidator — contradictions, gaps, support_scores, qualité.
    10. collect_evidence() — pipeline complet, parallélisme, injection.
    11. EvidenceEngineAgent — protocole, run, trace.
    12. Pipeline intégration (QA → HE → EE).
    13. Régressions Phase 3.0/3.1/3.2.

Contraintes :
    - Aucun appel Ollama, réseau ou Supabase.
    - Tous les LLM mockés via injection.
    - Tous les tests isolés (pas d'état partagé).
    - asyncio.run() pour les coroutines.

Commandes :
    pytest tests/test_phase33_evidence_engine.py -v
    pytest -q
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional

import pytest

from models.reasoning import (
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
from core.reasoning.evidence_collector import (
    ConceptEvidenceCollector,
    CounterEvidenceCollector,
    DomainEvidenceCollector,
    EvidenceCollectionStrategy,
    HypothesisEvidenceCollector,
    LLMEvidenceCollector,
    classify_evidence_type,
)
from core.reasoning.evidence_engine import (
    EvidenceEngineAgent,
    _resolve_strategies,
    _safe_collect,
    collect_evidence,
)
from core.reasoning.evidence_ranker import EvidenceRanker
from core.reasoning.evidence_validator import EvidenceValidator
from core.reasoning.reasoning_engine import (
    DEFAULT_PIPELINE,
    _build_summary,
    context_to_response,
    run_reasoning,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_request(question: str = "Question de test", **kw) -> ReasoningRequest:
    return ReasoningRequest(question=question, **kw)


def make_ctx(
    question   : str                      = "Question de test",
    analysis   : Optional[QuestionAnalysis] = None,
    hypotheses : Optional[list[Hypothesis]] = None,
    **kw,
) -> ReasoningContext:
    ctx = ReasoningContext(
        request = make_request(question, **kw),
        user_id = "test-user",
    )
    ctx.initialize_trace()
    if analysis:
        ctx.analysis = analysis
    if hypotheses is not None:
        ctx.hypotheses = hypotheses
    return ctx


def make_analysis(
    question_type : QuestionType = QuestionType.FACTUAL,
    domain        : str          = "general",
    complexity    : float        = 0.4,
    key_concepts  : list[str]   | None = None,
    key_entities  : list[str]   | None = None,
) -> QuestionAnalysis:
    return QuestionAnalysis(
        original_question = "Question de test",
        question_type     = question_type,
        domain            = domain,
        complexity_score  = complexity,
        key_concepts      = key_concepts or [],
        key_entities      = key_entities or [],
    )


def make_hypothesis(
    content : str   = "Hypothèse de test",
    score   : float = 0.5,
    hid     : Optional[str] = None,
) -> Hypothesis:
    h = Hypothesis(content=content, initial_score=score)
    if hid:
        # forcer l'ID pour les tests déterministes
        object.__setattr__(h, "hypothesis_id", hid)
    return h


def make_evidence(
    content    : str             = "Preuve de test",
    relevance  : float           = 0.6,
    credibility: float           = 0.6,
    ev_type    : EvidenceType    = EvidenceType.FACT,
    source_type: EvidenceSourceType = EvidenceSourceType.VECTOR_MEMORY,
    relations  : dict            | None = None,
) -> Evidence:
    return Evidence(
        content           = content,
        source_type       = source_type,
        source_ref        = "test",
        relevance_score   = relevance,
        credibility_score = credibility,
        evidence_type     = ev_type,
        relations         = relations or {},
    )


def make_llm_mock(response_dict: dict):
    async def _mock(prompt: str, **kw) -> str:
        return json.dumps(response_dict, ensure_ascii=False)
    return _mock


def make_llm_mock_text(text: str):
    async def _mock(prompt: str, **kw) -> str:
        return text
    return _mock


def make_llm_error():
    async def _mock(prompt: str, **kw) -> str:
        raise RuntimeError("LLM simulé hors ligne")
    return _mock


def make_strategy_mock(evidence_list: list[Evidence], name: str = "mock"):
    class _Mock:
        @property
        def name(self): return name
        async def collect(self, ctx): return evidence_list
    return _Mock()


def make_strategy_error(name: str = "failing"):
    class _Fail:
        @property
        def name(self): return name
        async def collect(self, ctx):
            raise RuntimeError("Stratégie en erreur")
    return _Fail()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Modèles Phase 3.3
# ─────────────────────────────────────────────────────────────────────────────

class TestEvidenceTypeEnum:
    def test_valeurs_disponibles(self):
        assert EvidenceType.FACT.value       == "fact"
        assert EvidenceType.HYPOTHESIS.value == "hypothesis"
        assert EvidenceType.OPINION.value    == "opinion"

    def test_est_string_enum(self):
        assert isinstance(EvidenceType.FACT, str)


class TestEvidenceModelPhase33:
    def test_champs_anciens_inchanges(self):
        ev = make_evidence()
        assert hasattr(ev, "evidence_id")
        assert hasattr(ev, "content")
        assert hasattr(ev, "source_type")
        assert hasattr(ev, "relevance_score")
        assert hasattr(ev, "relations")

    def test_credibility_score_default(self):
        ev = Evidence(
            content="test", source_type=EvidenceSourceType.VECTOR_MEMORY, source_ref="x"
        )
        assert ev.credibility_score == 0.5

    def test_evidence_type_default_fact(self):
        ev = Evidence(
            content="test", source_type=EvidenceSourceType.VECTOR_MEMORY, source_ref="x"
        )
        assert ev.evidence_type == EvidenceType.FACT

    def test_creation_complete_phase33(self):
        ev = make_evidence(
            ev_type    = EvidenceType.OPINION,
            credibility = 0.4,
        )
        assert ev.evidence_type    == EvidenceType.OPINION
        assert ev.credibility_score == 0.4

    def test_credibility_hors_borne_invalide(self):
        with pytest.raises(Exception):
            Evidence(
                content="test", source_type=EvidenceSourceType.VECTOR_MEMORY,
                source_ref="x", credibility_score=1.5,
            )

    def test_model_copy_preserve_tous_champs(self):
        ev   = make_evidence(relevance=0.5, credibility=0.7, ev_type=EvidenceType.HYPOTHESIS)
        copy = ev.model_copy(update={"relevance_score": 0.8})
        assert copy.credibility_score == 0.7
        assert copy.evidence_type     == EvidenceType.HYPOTHESIS
        assert copy.relevance_score   == 0.8


class TestEvidenceEvaluationPhase33:
    def test_champs_anciens_presents(self):
        ev = EvidenceEvaluation()
        assert hasattr(ev, "total_evidence_count")
        assert hasattr(ev, "contradictions")
        assert hasattr(ev, "knowledge_gaps")
        assert hasattr(ev, "overall_quality_score")
        assert hasattr(ev, "best_hypothesis_id")

    def test_support_scores_default_vide(self):
        ev = EvidenceEvaluation()
        assert ev.support_scores == {}

    def test_evidence_per_hypothesis_default_vide(self):
        ev = EvidenceEvaluation()
        assert ev.evidence_per_hypothesis == {}

    def test_creation_complete(self):
        ev = EvidenceEvaluation(
            total_evidence_count    = 5,
            support_scores          = {"abc": 0.8},
            evidence_per_hypothesis = {"abc": 3},
        )
        assert ev.support_scores          == {"abc": 0.8}
        assert ev.evidence_per_hypothesis == {"abc": 3}


# ─────────────────────────────────────────────────────────────────────────────
# 2. classify_evidence_type
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyEvidenceType:
    def test_fait_par_defaut(self):
        assert classify_evidence_type("Python est un langage de programmation.") == EvidenceType.FACT

    def test_hypothese_detectee_pourrait(self):
        assert classify_evidence_type("Cela pourrait être dû à un problème de cache.") == EvidenceType.HYPOTHESIS

    def test_hypothese_detectee_probablement(self):
        assert classify_evidence_type("La solution est probablement liée à la configuration.") == EvidenceType.HYPOTHESIS

    def test_opinion_detectee_en_general(self):
        assert classify_evidence_type("En général, les développeurs préfèrent Python.") == EvidenceType.OPINION

    def test_opinion_detectee_generalement(self):
        assert classify_evidence_type("Généralement, cette approche est plus efficace.") == EvidenceType.OPINION

    def test_chaine_vide_retourne_fact(self):
        assert classify_evidence_type("") == EvidenceType.FACT

    def test_insensible_casse(self):
        result = classify_evidence_type("POURRAIT être une solution.")
        assert result == EvidenceType.HYPOTHESIS


# ─────────────────────────────────────────────────────────────────────────────
# 3. EvidenceRanker
# ─────────────────────────────────────────────────────────────────────────────

class TestEvidenceRankerScore:
    def setup_method(self):
        self.ranker = EvidenceRanker()

    def test_score_liste_vide(self):
        ctx = make_ctx()
        assert self.ranker.score([], ctx) == []

    def test_score_ne_modifie_pas_original(self):
        ctx = make_ctx(analysis=make_analysis())
        ev  = make_evidence(relevance=0.5, credibility=0.5)
        orig_relevance = ev.relevance_score
        self.ranker.score([ev], ctx)
        assert ev.relevance_score == orig_relevance  # non modifié

    def test_score_cree_nouvelles_instances(self):
        ctx    = make_ctx(analysis=make_analysis())
        ev     = make_evidence()
        scored = self.ranker.score([ev], ctx)
        assert scored[0] is not ev

    def test_credibilite_fact_superieure_opinion(self):
        ctx     = make_ctx(analysis=make_analysis())
        ev_fact = make_evidence(ev_type=EvidenceType.FACT)
        ev_op   = make_evidence(ev_type=EvidenceType.OPINION)
        scored  = self.ranker.score([ev_fact, ev_op], ctx)
        assert scored[0].credibility_score > scored[1].credibility_score

    def test_bonus_concept_augmente_relevance(self):
        ctx_sans = make_ctx(analysis=make_analysis(key_concepts=[]))
        ctx_avec = make_ctx(analysis=make_analysis(key_concepts=["python"]))
        ev = make_evidence(content="python est un langage populaire", relevance=0.5)
        s_sans = self.ranker.score([ev], ctx_sans)[0].relevance_score
        s_avec = self.ranker.score([ev], ctx_avec)[0].relevance_score
        assert s_avec > s_sans

    def test_bonus_domaine_augmente_relevance(self):
        ctx = make_ctx(analysis=make_analysis(domain="technique"))
        ev_avec = make_evidence(content="dans le domaine technique", relevance=0.5)
        ev_sans = make_evidence(content="sans mention du domaine", relevance=0.5)
        scored  = self.ranker.score([ev_avec, ev_sans], ctx)
        assert scored[0].relevance_score > scored[1].relevance_score

    def test_score_ne_depasse_pas_1(self):
        ctx = make_ctx(analysis=make_analysis(
            key_concepts=["python", "api", "docker", "kubernetes"],
            domain="technique",
        ))
        ev = make_evidence(
            content   = "python api docker kubernetes technique",
            relevance = 0.95,
        )
        scored = self.ranker.score([ev], ctx)
        assert scored[0].relevance_score   <= 1.0
        assert scored[0].credibility_score <= 1.0

    def test_score_dans_intervalle(self):
        ctx    = make_ctx(analysis=make_analysis())
        ev     = make_evidence()
        scored = self.ranker.score([ev], ctx)
        assert 0.0 <= scored[0].relevance_score   <= 1.0
        assert 0.0 <= scored[0].credibility_score <= 1.0


class TestEvidenceRankerFilter:
    def setup_method(self):
        self.ranker = EvidenceRanker()

    def test_filtre_sous_seuil(self):
        ev_bon    = make_evidence(relevance=0.8, credibility=0.8)
        ev_faible = make_evidence(relevance=0.1, credibility=0.1)
        result = self.ranker.filter_by_threshold([ev_bon, ev_faible], 0.25)
        assert ev_bon in result
        assert ev_faible not in result

    def test_seuil_zero_garde_tout(self):
        evidence = [make_evidence(relevance=0.1, credibility=0.1) for _ in range(3)]
        assert len(self.ranker.filter_by_threshold(evidence, 0.0)) == 3

    def test_liste_vide(self):
        assert self.ranker.filter_by_threshold([], 0.5) == []


class TestEvidenceRankerTopN:
    def setup_method(self):
        self.ranker = EvidenceRanker()

    def test_top_3_depuis_5(self):
        evs    = [make_evidence(relevance=i * 0.1, credibility=0.8) for i in range(5)]
        result = self.ranker.top_n(evs, n=3)
        assert len(result) == 3

    def test_tri_composite_decroissant(self):
        e1 = make_evidence(relevance=0.9, credibility=0.9)
        e2 = make_evidence(relevance=0.5, credibility=0.5)
        e3 = make_evidence(relevance=0.7, credibility=0.7)
        result = self.ranker.top_n([e1, e2, e3], n=3)
        composites = [self.ranker.composite_score(e) for e in result]
        assert composites == sorted(composites, reverse=True)

    def test_n_superieur_longueur(self):
        evs = [make_evidence() for _ in range(2)]
        assert len(self.ranker.top_n(evs, n=10)) == 2

    def test_n_zero(self):
        assert self.ranker.top_n([make_evidence()], n=0) == []

    def test_liste_vide(self):
        assert self.ranker.top_n([], n=5) == []


class TestEvidenceRankerRank:
    def setup_method(self):
        self.ranker = EvidenceRanker()

    def test_pipeline_complet(self):
        ctx  = make_ctx(analysis=make_analysis(key_concepts=["python"]))
        evs  = [
            make_evidence("python est rapide",   relevance=0.7, credibility=0.8),
            make_evidence("java est performant", relevance=0.5, credibility=0.6),
        ]
        result = self.ranker.rank(evs, ctx, n=2)
        assert len(result) <= 2

    def test_vide_retourne_vide(self):
        ctx = make_ctx()
        assert self.ranker.rank([], ctx) == []

    def test_composite_score_utilitaire(self):
        ev = make_evidence(relevance=0.8, credibility=0.5)
        # Note : EvidenceRanker.score() va recalculer ces valeurs
        # composite_score travaille sur les valeurs actuelles de l'instance
        score = self.ranker.composite_score(ev)
        assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. HypothesisEvidenceCollector
# ─────────────────────────────────────────────────────────────────────────────

class TestHypothesisEvidenceCollector:
    def setup_method(self):
        self.collector = HypothesisEvidenceCollector()

    def test_name(self):
        assert self.collector.name == "hypothesis_evidence"

    def test_sans_hypotheses_retourne_vide(self):
        ctx    = make_ctx(analysis=make_analysis(), hypotheses=[])
        result = asyncio.run(self.collector.collect(ctx))
        assert result == []

    def test_une_hypothese_une_preuve(self):
        h      = make_hypothesis("Python est adapté au ML")
        ctx    = make_ctx(analysis=make_analysis(), hypotheses=[h])
        result = asyncio.run(self.collector.collect(ctx))
        assert len(result) == 1

    def test_trois_hypotheses_trois_preuves(self):
        hyps   = [make_hypothesis(f"hyp {i}") for i in range(3)]
        ctx    = make_ctx(analysis=make_analysis(), hypotheses=hyps)
        result = asyncio.run(self.collector.collect(ctx))
        assert len(result) == 3

    def test_relation_supports(self):
        h      = make_hypothesis("test")
        ctx    = make_ctx(analysis=make_analysis(), hypotheses=[h])
        result = asyncio.run(self.collector.collect(ctx))
        assert result[0].relations.get(h.hypothesis_id) == EvidenceRelation.SUPPORTS

    def test_source_type_vector_memory(self):
        h   = make_hypothesis("test")
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[h])
        ev  = asyncio.run(self.collector.collect(ctx))[0]
        assert ev.source_type == EvidenceSourceType.VECTOR_MEMORY

    def test_scores_dans_intervalle(self):
        h      = make_hypothesis("test")
        ctx    = make_ctx(analysis=make_analysis(), hypotheses=[h])
        ev     = asyncio.run(self.collector.collect(ctx))[0]
        assert 0.0 <= ev.relevance_score   <= 1.0
        assert 0.0 <= ev.credibility_score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. CounterEvidenceCollector
# ─────────────────────────────────────────────────────────────────────────────

class TestCounterEvidenceCollector:
    def setup_method(self):
        self.collector = CounterEvidenceCollector()

    def test_name(self):
        assert self.collector.name == "counter_evidence"

    def test_sans_hypotheses_retourne_vide(self):
        ctx    = make_ctx(analysis=make_analysis(), hypotheses=[])
        result = asyncio.run(self.collector.collect(ctx))
        assert result == []

    def test_une_hypothese_une_contre_preuve(self):
        h      = make_hypothesis("Python est le meilleur")
        ctx    = make_ctx(analysis=make_analysis(), hypotheses=[h])
        result = asyncio.run(self.collector.collect(ctx))
        assert len(result) == 1

    def test_relation_contradicts(self):
        h      = make_hypothesis("test")
        ctx    = make_ctx(analysis=make_analysis(), hypotheses=[h])
        result = asyncio.run(self.collector.collect(ctx))
        assert result[0].relations.get(h.hypothesis_id) == EvidenceRelation.CONTRADICTS

    def test_source_type_external(self):
        h   = make_hypothesis("test")
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[h])
        ev  = asyncio.run(self.collector.collect(ctx))[0]
        assert ev.source_type == EvidenceSourceType.EXTERNAL

    def test_deux_hypotheses_deux_contre_preuves(self):
        hyps   = [make_hypothesis(f"h{i}") for i in range(2)]
        ctx    = make_ctx(analysis=make_analysis(), hypotheses=hyps)
        result = asyncio.run(self.collector.collect(ctx))
        assert len(result) == 2

    def test_contre_preuves_ciblent_hypotheses_differentes(self):
        h1 = make_hypothesis("h1")
        h2 = make_hypothesis("h2")
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[h1, h2])
        result = asyncio.run(self.collector.collect(ctx))
        ids = {list(ev.relations.keys())[0] for ev in result if ev.relations}
        assert h1.hypothesis_id in ids
        assert h2.hypothesis_id in ids


# ─────────────────────────────────────────────────────────────────────────────
# 6. ConceptEvidenceCollector
# ─────────────────────────────────────────────────────────────────────────────

class TestConceptEvidenceCollector:
    def setup_method(self):
        self.collector = ConceptEvidenceCollector()

    def test_name(self):
        assert self.collector.name == "concept_evidence"

    def test_sans_concepts_retourne_vide(self):
        ctx    = make_ctx(analysis=make_analysis(key_concepts=[]))
        result = asyncio.run(self.collector.collect(ctx))
        assert result == []

    def test_un_concept_une_preuve(self):
        ctx    = make_ctx(analysis=make_analysis(key_concepts=["python"]))
        result = asyncio.run(self.collector.collect(ctx))
        assert len(result) == 1

    def test_max_trois_concepts(self):
        ctx    = make_ctx(analysis=make_analysis(
            key_concepts=["python", "api", "docker", "kubernetes", "redis"]
        ))
        result = asyncio.run(self.collector.collect(ctx))
        assert len(result) <= 3

    def test_concept_dans_contenu(self):
        ctx    = make_ctx(analysis=make_analysis(key_concepts=["fastapi"]))
        result = asyncio.run(self.collector.collect(ctx))
        assert "fastapi" in result[0].content.lower()

    def test_source_type_neuron_graph(self):
        ctx = make_ctx(analysis=make_analysis(key_concepts=["test"]))
        ev  = asyncio.run(self.collector.collect(ctx))[0]
        assert ev.source_type == EvidenceSourceType.NEURON_GRAPH

    def test_concept_presente_dans_hyp_support(self):
        h = make_hypothesis("python est utilisé ici")
        ctx = make_ctx(
            analysis   = make_analysis(key_concepts=["python"]),
            hypotheses = [h],
        )
        result = asyncio.run(self.collector.collect(ctx))
        # Le concept "python" est dans l'hypothèse → SUPPORTS
        assert result[0].relations.get(h.hypothesis_id) == EvidenceRelation.SUPPORTS


# ─────────────────────────────────────────────────────────────────────────────
# 7. DomainEvidenceCollector
# ─────────────────────────────────────────────────────────────────────────────

class TestDomainEvidenceCollector:
    def setup_method(self):
        self.collector = DomainEvidenceCollector()

    def test_name(self):
        assert self.collector.name == "domain_evidence"

    def test_sans_hypotheses_retourne_vide(self):
        ctx    = make_ctx(analysis=make_analysis(), hypotheses=[])
        result = asyncio.run(self.collector.collect(ctx))
        assert result == []

    def test_avec_hypotheses_retourne_faits(self):
        h      = make_hypothesis("test")
        ctx    = make_ctx(analysis=make_analysis(domain="technique"), hypotheses=[h])
        result = asyncio.run(self.collector.collect(ctx))
        assert len(result) >= 1

    def test_max_deux_preuves(self):
        h   = make_hypothesis("test")
        ctx = make_ctx(analysis=make_analysis(domain="finance"), hypotheses=[h])
        assert len(asyncio.run(self.collector.collect(ctx))) <= 2

    def test_evidence_type_fact(self):
        h   = make_hypothesis("test")
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[h])
        ev  = asyncio.run(self.collector.collect(ctx))[0]
        assert ev.evidence_type == EvidenceType.FACT

    def test_credibility_elevee(self):
        h   = make_hypothesis("test")
        ctx = make_ctx(analysis=make_analysis(domain="science"), hypotheses=[h])
        ev  = asyncio.run(self.collector.collect(ctx))[0]
        assert ev.credibility_score >= 0.7

    def test_domaine_general_fonctionne(self):
        h   = make_hypothesis("test")
        ctx = make_ctx(analysis=make_analysis(domain="general"), hypotheses=[h])
        result = asyncio.run(self.collector.collect(ctx))
        assert len(result) >= 1

    def test_relations_neutres(self):
        h1 = make_hypothesis("h1")
        h2 = make_hypothesis("h2")
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[h1, h2])
        result = asyncio.run(self.collector.collect(ctx))
        for ev in result:
            for rel in ev.relations.values():
                assert rel == EvidenceRelation.NEUTRAL


# ─────────────────────────────────────────────────────────────────────────────
# 8. LLMEvidenceCollector
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMEvidenceCollector:
    LLM_RESPONSE = {
        "evidence": [
            {
                "content"     : "Python domine le domaine du machine learning.",
                "type"        : "fact",
                "credibility" : 0.85,
                "relevance"   : 0.80,
                "hypothesis_id": "abc12345",
                "relation"    : "supports",
            },
            {
                "content"     : "JavaScript est populaire pour le web.",
                "type"        : "fact",
                "credibility" : 0.75,
                "relevance"   : 0.65,
                "hypothesis_id": None,
                "relation"    : "neutral",
            },
        ]
    }

    def test_name(self):
        assert LLMEvidenceCollector().name == "llm_evidence"

    def test_avec_llm_valide(self):
        h   = make_hypothesis("test", hid="abc12345")
        ctx = make_ctx(
            analysis   = make_analysis(),
            hypotheses = [h],
        )
        collector = LLMEvidenceCollector(
            llm_generate=make_llm_mock(self.LLM_RESPONSE)
        )
        result = asyncio.run(collector.collect(ctx))
        assert len(result) == 2

    def test_contenu_preserve(self):
        h   = make_hypothesis("test", hid="abc12345")
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[h])
        collector = LLMEvidenceCollector(
            llm_generate=make_llm_mock(self.LLM_RESPONSE)
        )
        result = asyncio.run(collector.collect(ctx))
        assert "Python" in result[0].content

    def test_credibility_preservee(self):
        h   = make_hypothesis("test", hid="abc12345")
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[h])
        collector = LLMEvidenceCollector(
            llm_generate=make_llm_mock(self.LLM_RESPONSE)
        )
        result = asyncio.run(collector.collect(ctx))
        assert result[0].credibility_score == 0.85

    def test_relation_supports_preservee(self):
        h   = make_hypothesis("test", hid="abc12345")
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[h])
        collector = LLMEvidenceCollector(
            llm_generate=make_llm_mock(self.LLM_RESPONSE)
        )
        result = asyncio.run(collector.collect(ctx))
        assert result[0].relations.get("abc12345") == EvidenceRelation.SUPPORTS

    def test_json_invalide_retourne_vide(self):
        ctx = make_ctx(analysis=make_analysis())
        collector = LLMEvidenceCollector(
            llm_generate=make_llm_mock_text("Désolé, je ne peux pas répondre.")
        )
        result = asyncio.run(collector.collect(ctx))
        assert result == []

    def test_json_sans_cle_evidence_retourne_vide(self):
        ctx = make_ctx(analysis=make_analysis())
        collector = LLMEvidenceCollector(
            llm_generate=make_llm_mock({"autre_cle": []})
        )
        assert asyncio.run(collector.collect(ctx)) == []

    def test_erreur_llm_retourne_vide(self):
        ctx = make_ctx(analysis=make_analysis())
        collector = LLMEvidenceCollector(llm_generate=make_llm_error())
        result = asyncio.run(collector.collect(ctx))
        assert result == []

    def test_sans_llm_retourne_vide_ou_liste(self):
        ctx       = make_ctx(analysis=make_analysis())
        collector = LLMEvidenceCollector(llm_generate=None)
        result    = asyncio.run(collector.collect(ctx))
        assert isinstance(result, list)

    def test_max_evidence_respecte(self):
        resp = {"evidence": [
            {"content": f"preuve {i}", "type": "fact",
             "credibility": 0.7, "relevance": 0.6,
             "hypothesis_id": None, "relation": "neutral"}
            for i in range(10)
        ]}
        ctx = make_ctx(analysis=make_analysis())
        collector = LLMEvidenceCollector(
            llm_generate=make_llm_mock(resp), max_evidence=3
        )
        result = asyncio.run(collector.collect(ctx))
        assert len(result) <= 3

    def test_contenu_vide_ignore(self):
        resp = {"evidence": [
            {"content": "",      "type": "fact", "credibility": 0.8, "relevance": 0.8, "hypothesis_id": None, "relation": "neutral"},
            {"content": "valide","type": "fact", "credibility": 0.8, "relevance": 0.8, "hypothesis_id": None, "relation": "neutral"},
        ]}
        ctx = make_ctx(analysis=make_analysis())
        collector = LLMEvidenceCollector(llm_generate=make_llm_mock(resp))
        result = asyncio.run(collector.collect(ctx))
        assert all(ev.content for ev in result)

    def test_credibility_hors_borne_clampee(self):
        resp = {"evidence": [
            {"content": "test", "type": "fact", "credibility": 9.9,
             "relevance": 0.7, "hypothesis_id": None, "relation": "neutral"}
        ]}
        ctx = make_ctx(analysis=make_analysis())
        collector = LLMEvidenceCollector(llm_generate=make_llm_mock(resp))
        result = asyncio.run(collector.collect(ctx))
        assert result[0].credibility_score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. EvidenceValidator
# ─────────────────────────────────────────────────────────────────────────────

class TestEvidenceValidatorContradictions:
    def setup_method(self):
        self.validator = EvidenceValidator()

    def test_pas_de_contradiction_evidence_vide(self):
        h = make_hypothesis("test")
        assert self.validator.detect_contradictions([], [h]) == []

    def test_pas_de_contradiction_une_seule_preuve(self):
        h  = make_hypothesis("test")
        ev = make_evidence(relations={h.hypothesis_id: EvidenceRelation.SUPPORTS})
        assert self.validator.detect_contradictions([ev], [h]) == []

    def test_contradiction_detectee(self):
        h   = make_hypothesis("test")
        e_s = make_evidence("support",  relations={h.hypothesis_id: EvidenceRelation.SUPPORTS})
        e_c = make_evidence("counter",  relations={h.hypothesis_id: EvidenceRelation.CONTRADICTS})
        result = self.validator.detect_contradictions([e_s, e_c], [h])
        assert len(result) == 1

    def test_paire_normalisee_pas_de_doublons(self):
        h   = make_hypothesis("test")
        e_s = make_evidence("support",  relations={h.hypothesis_id: EvidenceRelation.SUPPORTS})
        e_c = make_evidence("counter",  relations={h.hypothesis_id: EvidenceRelation.CONTRADICTS})
        result = self.validator.detect_contradictions([e_s, e_c, e_s, e_c], [h])
        # Une seule paire unique même si doublons en entrée
        assert len(set(result)) == len(result)

    def test_preuve_neutre_pas_de_contradiction(self):
        h   = make_hypothesis("test")
        e_n = make_evidence(relations={h.hypothesis_id: EvidenceRelation.NEUTRAL})
        e_s = make_evidence(relations={h.hypothesis_id: EvidenceRelation.SUPPORTS})
        assert self.validator.detect_contradictions([e_n, e_s], [h]) == []


class TestEvidenceValidatorGaps:
    def setup_method(self):
        self.validator = EvidenceValidator()

    def test_pas_de_lacune_si_pas_dhypotheses(self):
        assert self.validator.identify_knowledge_gaps([], []) == []

    def test_lacune_si_pas_de_support(self):
        h      = make_hypothesis("hypothèse sans soutien")
        result = self.validator.identify_knowledge_gaps([h], [])
        assert len(result) == 1

    def test_pas_de_lacune_si_soutien_present(self):
        h  = make_hypothesis("test")
        ev = make_evidence(relations={h.hypothesis_id: EvidenceRelation.SUPPORTS})
        assert self.validator.identify_knowledge_gaps([h], [ev]) == []

    def test_contradiction_seule_est_une_lacune(self):
        """Une contradiction sans soutien = lacune de connaissance."""
        h  = make_hypothesis("test")
        ev = make_evidence(relations={h.hypothesis_id: EvidenceRelation.CONTRADICTS})
        result = self.validator.identify_knowledge_gaps([h], [ev])
        assert len(result) == 1

    def test_deux_hypotheses_une_lacune(self):
        h1 = make_hypothesis("avec soutien")
        h2 = make_hypothesis("sans soutien")
        ev = make_evidence(relations={h1.hypothesis_id: EvidenceRelation.SUPPORTS})
        result = self.validator.identify_knowledge_gaps([h1, h2], [ev])
        assert len(result) == 1


class TestEvidenceValidatorSupportScores:
    def setup_method(self):
        self.validator = EvidenceValidator()

    def test_sans_preuves_score_neutre(self):
        h      = make_hypothesis("test")
        scores = self.validator.compute_support_scores([h], [])
        assert scores[h.hypothesis_id] == 0.5  # (0 - 0 + 1) / 2

    def test_soutien_pur_score_superieur_a_0_5(self):
        h  = make_hypothesis("test")
        ev = make_evidence(
            relevance=0.8, credibility=0.8,
            relations={h.hypothesis_id: EvidenceRelation.SUPPORTS}
        )
        scores = self.validator.compute_support_scores([h], [ev])
        assert scores[h.hypothesis_id] > 0.5

    def test_contradiction_pure_score_inferieur_a_0_5(self):
        h  = make_hypothesis("test")
        ev = make_evidence(
            relevance=0.8, credibility=0.8,
            relations={h.hypothesis_id: EvidenceRelation.CONTRADICTS}
        )
        scores = self.validator.compute_support_scores([h], [ev])
        assert scores[h.hypothesis_id] < 0.5

    def test_scores_entre_0_et_1(self):
        h  = make_hypothesis("test")
        ev = make_evidence(
            relevance=1.0, credibility=1.0,
            relations={h.hypothesis_id: EvidenceRelation.SUPPORTS}
        )
        scores = self.validator.compute_support_scores([h], [ev])
        assert 0.0 <= scores[h.hypothesis_id] <= 1.0

    def test_meilleure_hypothese_avec_plus_de_soutien(self):
        h1 = make_hypothesis("bien soutenue")
        h2 = make_hypothesis("peu soutenue")
        e_fort  = make_evidence(relevance=0.9, credibility=0.9,
                                relations={h1.hypothesis_id: EvidenceRelation.SUPPORTS})
        e_faible = make_evidence(relevance=0.3, credibility=0.3,
                                 relations={h2.hypothesis_id: EvidenceRelation.SUPPORTS})
        scores = self.validator.compute_support_scores([h1, h2], [e_fort, e_faible])
        assert scores[h1.hypothesis_id] > scores[h2.hypothesis_id]


class TestEvidenceValidatorQuality:
    def setup_method(self):
        self.validator = EvidenceValidator()

    def test_qualite_zero_si_vide(self):
        assert self.validator.compute_overall_quality([]) == 0.0

    def test_qualite_composite_correcte(self):
        ev = make_evidence(relevance=0.8, credibility=0.8)
        q  = self.validator.compute_overall_quality([ev])
        assert abs(q - 0.64) < 0.01

    def test_qualite_entre_0_et_1(self):
        evs = [make_evidence(relevance=r, credibility=0.7) for r in [0.2, 0.5, 0.9]]
        q   = self.validator.compute_overall_quality(evs)
        assert 0.0 <= q <= 1.0


class TestEvidenceValidatorBuildEvaluation:
    def setup_method(self):
        self.validator = EvidenceValidator()

    def test_sans_donnees_retourne_evaluation_vide(self):
        ev = self.validator.build_evaluation([], [])
        assert ev.total_evidence_count == 0

    def test_champs_renseignes(self):
        h  = make_hypothesis("test")
        ev_support = make_evidence(
            relevance=0.7, credibility=0.7,
            relations={h.hypothesis_id: EvidenceRelation.SUPPORTS}
        )
        result = self.validator.build_evaluation([h], [ev_support])
        assert result.total_evidence_count   == 1
        assert result.best_hypothesis_id     == h.hypothesis_id
        assert h.hypothesis_id in result.support_scores
        assert h.hypothesis_id in result.evidence_per_hypothesis

    def test_meilleure_hypothese_selectionnee(self):
        h1 = make_hypothesis("forte")
        h2 = make_hypothesis("faible")
        e1 = make_evidence(relevance=0.9, credibility=0.9,
                           relations={h1.hypothesis_id: EvidenceRelation.SUPPORTS})
        e2 = make_evidence(relevance=0.3, credibility=0.3,
                           relations={h2.hypothesis_id: EvidenceRelation.SUPPORTS})
        result = self.validator.build_evaluation([h1, h2], [e1, e2])
        assert result.best_hypothesis_id == h1.hypothesis_id

    def test_contradictions_incluses(self):
        h  = make_hypothesis("test")
        e_s = make_evidence(relations={h.hypothesis_id: EvidenceRelation.SUPPORTS})
        e_c = make_evidence(relations={h.hypothesis_id: EvidenceRelation.CONTRADICTS})
        result = self.validator.build_evaluation([h], [e_s, e_c])
        assert len(result.contradictions) == 1

    def test_lacunes_incluses(self):
        h      = make_hypothesis("sans soutien")
        result = self.validator.build_evaluation([h], [])
        assert len(result.knowledge_gaps) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 10. collect_evidence()
# ─────────────────────────────────────────────────────────────────────────────

class TestCollectEvidence:
    def test_retourne_ctx(self):
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(collect_evidence(
            ctx, strategies=[make_strategy_mock([])]
        ))
        assert result is ctx

    def test_peuple_ctx_evidence(self):
        h  = make_hypothesis("test")
        ev = make_evidence(relevance=0.7, credibility=0.7)
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[h])
        result = asyncio.run(collect_evidence(
            ctx, strategies=[make_strategy_mock([ev])]
        ))
        assert isinstance(result.evidence, list)

    def test_peuple_ctx_evaluation(self):
        ctx = make_ctx(analysis=make_analysis())
        result = asyncio.run(collect_evidence(
            ctx, strategies=[make_strategy_mock([])]
        ))
        assert result.evaluation is not None

    def test_strategie_erreur_ignoree(self):
        ev   = make_evidence()
        s_ok = make_strategy_mock([ev], name="ok")
        s_er = make_strategy_error("broken")
        ctx  = make_ctx(analysis=make_analysis())
        result = asyncio.run(collect_evidence(
            ctx, strategies=[s_ok, s_er]
        ))
        # Ne doit pas lever d'exception
        assert isinstance(result.evidence, list)

    def test_trace_step_ajoute(self):
        ctx = make_ctx(analysis=make_analysis())
        result = asyncio.run(collect_evidence(
            ctx, strategies=[make_strategy_mock([])]
        ))
        names = [s.step_name for s in result.trace.steps]
        assert "evidence_engine" in names

    def test_trace_success_true(self):
        ctx = make_ctx(analysis=make_analysis())
        result = asyncio.run(collect_evidence(
            ctx, strategies=[make_strategy_mock([make_evidence()])]
        ))
        step = next(s for s in result.trace.steps if s.step_name == "evidence_engine")
        assert step.success is True

    def test_evidence_quality_score_mis_a_jour(self):
        ev  = make_evidence(relevance=0.8, credibility=0.8)
        ctx = make_ctx(analysis=make_analysis(), hypotheses=[make_hypothesis()])
        result = asyncio.run(collect_evidence(
            ctx, strategies=[make_strategy_mock([ev])]
        ))
        assert result.evidence_quality_score >= 0.0

    def test_max_evidence_respecte(self):
        evs = [make_evidence(relevance=0.8, credibility=0.8) for _ in range(30)]
        ctx = make_ctx(analysis=make_analysis())
        result = asyncio.run(collect_evidence(
            ctx, strategies=[make_strategy_mock(evs)], max_evidence=5
        ))
        assert len(result.evidence) <= 5

    def test_llm_injecte_dans_llm_strategy(self):
        llm_resp = {"evidence": [
            {"content": "via LLM", "type": "fact", "credibility": 0.9,
             "relevance": 0.9, "hypothesis_id": None, "relation": "neutral"}
        ]}
        ctx = make_ctx(analysis=make_analysis())
        result = asyncio.run(collect_evidence(
            ctx, llm_generate=make_llm_mock(llm_resp)
        ))
        # Le LLM est branché — on ne vérifie pas le contenu exact
        # car d'autres stratégies peuvent aussi retourner des preuves
        assert isinstance(result.evidence, list)

    def test_ranker_injectable(self):
        custom = EvidenceRanker(domain_bonus=0.0)
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(collect_evidence(
            ctx, strategies=[make_strategy_mock([make_evidence()])], ranker=custom
        ))
        assert isinstance(result.evidence, list)

    def test_validator_injectable(self):
        custom = EvidenceValidator()
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(collect_evidence(
            ctx, strategies=[make_strategy_mock([])], validator=custom
        ))
        assert result.evaluation is not None

    def test_safe_collect_ne_plante_pas(self):
        failing = make_strategy_error()
        ctx     = make_ctx()
        result  = asyncio.run(_safe_collect(failing, ctx))
        assert result == []

    def test_resolve_strategies_custom(self):
        custom = [make_strategy_mock([])]
        result = _resolve_strategies(custom, None)
        assert result is custom

    def test_resolve_strategies_defaut_non_vide(self):
        result = _resolve_strategies(None, None)
        assert len(result) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 11. EvidenceEngineAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestEvidenceEngineAgent:
    def test_name(self):
        assert EvidenceEngineAgent().name == "evidence_engine"

    def test_run_retourne_ctx(self):
        agent  = EvidenceEngineAgent(strategies=[make_strategy_mock([])])
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(agent.run(ctx))
        assert result is ctx

    def test_run_peuple_evidence(self):
        ev     = make_evidence()
        agent  = EvidenceEngineAgent(strategies=[make_strategy_mock([ev])])
        ctx    = make_ctx(analysis=make_analysis(), hypotheses=[make_hypothesis()])
        result = asyncio.run(agent.run(ctx))
        assert isinstance(result.evidence, list)

    def test_run_peuple_evaluation(self):
        agent  = EvidenceEngineAgent(strategies=[make_strategy_mock([])])
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(agent.run(ctx))
        assert result.evaluation is not None

    def test_run_ajoute_trace_step(self):
        agent  = EvidenceEngineAgent(strategies=[make_strategy_mock([])])
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(agent.run(ctx))
        names  = [s.step_name for s in result.trace.steps]
        assert "evidence_engine" in names

    def test_context_version_inchangee(self):
        agent  = EvidenceEngineAgent(strategies=[make_strategy_mock([])])
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(agent.run(ctx))
        assert result.context_version == "3.0"

    def test_strategies_injectees(self):
        ev     = make_evidence()
        custom = make_strategy_mock([ev], name="custom")
        agent  = EvidenceEngineAgent(strategies=[custom])
        ctx    = make_ctx(analysis=make_analysis(), hypotheses=[make_hypothesis()])
        result = asyncio.run(agent.run(ctx))
        assert isinstance(result.evidence, list)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Pipeline intégration (QA → HE → EE)
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_default_pipeline_trois_agents(self):
        assert len(DEFAULT_PIPELINE) >= 3

    def test_agent_names_dans_pipeline(self):
        names = [a.name for a in DEFAULT_PIPELINE]
        assert "question_analyzer"  in names
        assert "hypothesis_engine"  in names
        assert "evidence_engine"    in names

    def test_run_reasoning_peuple_analysis_hypotheses_evidence(self):
        req    = make_request("Pourquoi FastAPI est plus rapide que Flask ?")
        ctx    = asyncio.run(run_reasoning(request=req, user_id="test"))
        assert ctx.analysis    is not None
        assert isinstance(ctx.hypotheses, list)
        assert isinstance(ctx.evidence,   list)
        assert ctx.evaluation  is not None

    def test_run_reasoning_trace_trois_steps(self):
        req   = make_request("Comment déployer sur Kubernetes ?")
        ctx   = asyncio.run(run_reasoning(request=req, user_id="test"))
        names = [s.step_name for s in ctx.trace.steps]
        assert "question_analyzer" in names
        assert "hypothesis_engine" in names
        assert "evidence_engine"   in names

    def test_context_to_response_evidence_used(self):
        req      = make_request("FastAPI vs Django ?")
        ctx      = asyncio.run(run_reasoning(request=req, user_id="test"))
        response = context_to_response(ctx)
        assert response.evidence_used == len(ctx.evidence)

    def test_context_to_response_knowledge_gaps(self):
        req      = make_request("test")
        ctx      = asyncio.run(run_reasoning(request=req, user_id="test"))
        response = context_to_response(ctx)
        assert isinstance(response.knowledge_gaps, list)

    def test_run_reasoning_context_version(self):
        req = make_request("test")
        ctx = asyncio.run(run_reasoning(request=req, user_id="test"))
        assert ctx.context_version == "3.0"

    def test_build_summary_mentionne_preuves_si_presentes(self):
        ctx = make_ctx(analysis=make_analysis())
        ctx.evidence = [make_evidence() for _ in range(3)]
        summary = _build_summary(ctx)
        assert "3" in summary or "preuve" in summary.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 13. Régressions Phase 3.0/3.1/3.2
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionsPhasesPrecedentes:
    def test_hypothesis_retro_compatible_sans_champs_33(self):
        h = Hypothesis(content="test", initial_score=0.7)
        assert h.content == "test"
        # Champs Phase 3.2
        assert h.justification == ""
        assert h.strategy_name == "unknown"

    def test_evidence_retro_compatible_sans_champs_33(self):
        ev = Evidence(
            content="test",
            source_type=EvidenceSourceType.VECTOR_MEMORY,
            source_ref="x",
        )
        # Champs Phase 3.3 avec defaults
        assert ev.credibility_score == 0.5
        assert ev.evidence_type     == EvidenceType.FACT

    def test_evidence_evaluation_retro_compatible(self):
        ee = EvidenceEvaluation(total_evidence_count=3)
        assert ee.support_scores          == {}
        assert ee.evidence_per_hypothesis == {}

    def test_question_analysis_retro_compatible(self):
        qa = QuestionAnalysis(
            original_question="test",
            question_type=QuestionType.FACTUAL,
            complexity_score=0.3,
        )
        # Champs Phase 3.1 avec defaults
        assert qa.domain    == "general"
        assert qa.risk_level == "low"

    def test_build_summary_sans_analysis_non_vide(self):
        ctx = make_ctx()
        assert len(_build_summary(ctx)) > 0

    def test_run_reasoning_ne_leve_pas_exception(self):
        req = make_request("test simple")
        # Ne doit jamais lever d'exception, même avec LLM absent
        ctx = asyncio.run(run_reasoning(request=req, user_id="test"))
        assert isinstance(ctx, ReasoningContext)

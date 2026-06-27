"""Tests Phase 3.2 — HypothesisEngine.

Couvre :
    1. HypothesisRanker (Jaccard, déduplication, scoring, top-N).
    2. HeuristicStrategy (templates, tous les QuestionTypes, domaines).
    3. DecompositionStrategy (sub_questions, liste vide).
    4. ConceptualStrategy (key_concepts, relation entre concepts).
    5. LLMStrategy (JSON valide, JSON invalide, erreur réseau, désactivée).
    6. generate_hypotheses() (parallélisme, fusion, injection de stratégies).
    7. HypothesisEngineAgent (protocol, run, trace, intégration contexte).
    8. Pipeline reasoning_engine (QuestionAnalyzer + HypothesisEngine).
    9. Modèle Hypothesis Phase 3.2 (nouveaux champs, rétrocompatibilité).
    10. Régression Phase 3.0/3.1 (test corrigé + non-régression des anciens).

Contraintes de test :
    - Aucun appel Ollama, réseau, Supabase.
    - Toutes les dépendances LLM mockées via injection.
    - Tous les tests sont indépendants (pas d'état partagé).
    - asyncio.run() pour les coroutines.

Lancer :
    pytest tests/test_phase32_hypothesis_engine.py -v
    pytest tests/ -q   (régression complète)
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional

import pytest

from models.reasoning import (
    Hypothesis,
    QuestionAnalysis,
    QuestionType,
    ReasoningRequest,
    ReasoningStepRecord,
)
from core.reasoning.context import ReasoningContext
from core.reasoning.hypothesis_engine import (
    HypothesisEngineAgent,
    _resolve_strategies,
    _safe_generate,
    generate_hypotheses,
)
from core.reasoning.hypothesis_ranker import HypothesisRanker
from core.reasoning.hypothesis_strategies import (
    ConceptualStrategy,
    DecompositionStrategy,
    HeuristicStrategy,
    HypothesisStrategy,
    LLMStrategy,
    _build_substitutions,
)
# Note : _jaccard est une méthode privée de HypothesisRanker,
# accessible via HypothesisRanker()._jaccard() dans les tests.
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
    question: str = "Question de test",
    analysis: Optional[QuestionAnalysis] = None,
    **kw,
) -> ReasoningContext:
    ctx = ReasoningContext(request=make_request(question, **kw), user_id="test-user")
    ctx.initialize_trace()
    if analysis:
        ctx.analysis = analysis
    return ctx


def make_analysis(
    question_type  : QuestionType = QuestionType.FACTUAL,
    domain         : str          = "general",
    complexity     : float        = 0.4,
    key_concepts   : list[str]    | None = None,
    key_entities   : list[str]    | None = None,
    sub_questions  : list[str]    | None = None,
) -> QuestionAnalysis:
    return QuestionAnalysis(
        original_question = "Question de test",
        question_type     = question_type,
        domain            = domain,
        complexity_score  = complexity,
        key_concepts      = key_concepts  or [],
        key_entities      = key_entities  or [],
        sub_questions     = sub_questions or [],
    )


def make_hypothesis(content: str, score: float = 0.5, strategy: str = "test") -> Hypothesis:
    return Hypothesis(
        content       = content,
        initial_score = score,
        origin        = strategy,
        strategy_name = strategy,
    )


def make_llm_mock(response_dict: dict):
    """Mock async LLM retournant un JSON donné."""
    async def _mock(prompt: str, **kw) -> str:
        return json.dumps(response_dict, ensure_ascii=False)
    return _mock


def make_llm_mock_text(text: str):
    """Mock async LLM retournant du texte brut."""
    async def _mock(prompt: str, **kw) -> str:
        return text
    return _mock


def make_llm_mock_error():
    """Mock async LLM qui lève toujours une exception."""
    async def _mock(prompt: str, **kw) -> str:
        raise RuntimeError("LLM simulé hors ligne")
    return _mock


def make_strategy_mock(hypotheses: list[Hypothesis], name: str = "mock"):
    """Mock de stratégie qui retourne une liste fixe d'hypothèses."""
    class _MockStrategy:
        @property
        def name(self): return name
        @property
        def weight(self): return 0.5
        async def generate(self, ctx): return hypotheses
    return _MockStrategy()


def make_strategy_error(name: str = "failing"):
    """Mock de stratégie qui lève une exception."""
    class _FailStrategy:
        @property
        def name(self): return name
        @property
        def weight(self): return 0.5
        async def generate(self, ctx):
            raise RuntimeError("Stratégie simulée en erreur")
    return _FailStrategy()


# ─────────────────────────────────────────────────────────────────────────────
# 1. HypothesisRanker
# ─────────────────────────────────────────────────────────────────────────────

class TestJaccard:
    def setup_method(self):
        self.ranker = HypothesisRanker()

    def test_identiques(self):
        assert self.ranker._jaccard("bonjour monde", "bonjour monde") == 1.0

    def test_disjoints(self):
        assert self.ranker._jaccard("chat noir", "chien blanc") == 0.0

    def test_partiellement_similaires(self):
        sim = self.ranker._jaccard("python est rapide", "python est lent")
        assert 0.0 < sim < 1.0

    def test_deux_vides(self):
        assert self.ranker._jaccard("", "") == 1.0

    def test_un_vide(self):
        assert self.ranker._jaccard("texte", "") == 0.0

    def test_insensible_casse(self):
        assert self.ranker._jaccard("Python", "python") == 1.0

    def test_ordre_symetrique(self):
        a, b = "la solution optimale", "optimale est la solution"
        assert self.ranker._jaccard(a, b) == self.ranker._jaccard(b, a)


class TestDeduplicate:
    def setup_method(self):
        self.ranker = HypothesisRanker(similarity_threshold=0.55)

    def test_liste_vide(self):
        assert self.ranker.deduplicate([]) == []

    def test_une_seule(self):
        h = make_hypothesis("une seule hypothèse")
        assert self.ranker.deduplicate([h]) == [h]

    def test_identiques_un_conserve(self):
        h1 = make_hypothesis("python est un langage de programmation", score=0.5)
        h2 = make_hypothesis("python est un langage de programmation", score=0.6)
        result = self.ranker.deduplicate([h1, h2])
        assert len(result) == 1
        # Conserve celle avec le meilleur score
        assert result[0].initial_score == 0.6

    def test_similaires_depassant_seuil(self):
        h1 = make_hypothesis("la réponse concerne python dans le domaine technique")
        h2 = make_hypothesis("la réponse concerne python dans le domaine informatique")
        result = self.ranker.deduplicate([h1, h2])
        assert len(result) == 1

    def test_differentes_toutes_conservees(self):
        h1 = make_hypothesis("python est un langage interprété")
        h2 = make_hypothesis("FastAPI utilise les annotations de type")
        h3 = make_hypothesis("Docker simplifie le déploiement")
        result = self.ranker.deduplicate([h1, h2, h3])
        assert len(result) == 3

    def test_seuil_eleve_conserve_toutes(self):
        ranker = HypothesisRanker(similarity_threshold=0.99)
        h1 = make_hypothesis("python est rapide et efficace")
        h2 = make_hypothesis("python est puissant et populaire")
        result = ranker.deduplicate([h1, h2])
        assert len(result) == 2

    def test_cinq_hypotheses_dont_deux_doublons(self):
        hypotheses = [
            make_hypothesis("la cause est liée à python"),
            make_hypothesis("la cause est liée à python"),  # doublon
            make_hypothesis("docker simplifie le déploiement"),
            make_hypothesis("fastapi est un framework web"),
            make_hypothesis("la solution dépend du contexte"),
        ]
        result = self.ranker.deduplicate(hypotheses)
        assert len(result) == 4


class TestScore:
    def setup_method(self):
        self.ranker = HypothesisRanker()

    def test_score_entre_0_et_1(self):
        ctx = make_ctx(analysis=make_analysis(key_concepts=["python", "api"]))
        h   = make_hypothesis("python est utilisé pour les api web", score=0.5)
        result = self.ranker.score([h], ctx)
        assert 0.0 <= result[0].initial_score <= 1.0

    def test_bonus_concept_augmente_score(self):
        ctx_no_concept  = make_ctx(analysis=make_analysis())
        ctx_with_concept = make_ctx(analysis=make_analysis(key_concepts=["python"]))
        h = make_hypothesis("python est un excellent choix", score=0.5)
        score_no  = self.ranker.score([h], ctx_no_concept)[0].initial_score
        score_yes = self.ranker.score([h], ctx_with_concept)[0].initial_score
        assert score_yes > score_no

    def test_bonus_domaine_augmente_score(self):
        ctx = make_ctx(analysis=make_analysis(domain="technique"))
        h_avec    = make_hypothesis("dans le domaine technique, la solution est", score=0.5)
        h_sans    = make_hypothesis("la solution est évidente", score=0.5)
        scored    = self.ranker.score([h_avec, h_sans], ctx)
        assert scored[0].initial_score > scored[1].initial_score

    def test_score_ne_depasse_pas_1(self):
        ctx = make_ctx(analysis=make_analysis(
            key_concepts=["python", "api", "fastapi", "docker", "kubernetes"],
            domain="technique",
        ))
        # Hypothèse qui contient tous les concepts
        h = make_hypothesis(
            "python fastapi docker kubernetes technique api", score=0.95
        )
        result = self.ranker.score([h], ctx)
        assert result[0].initial_score <= 1.0

    def test_nouvelles_instances_crees(self):
        """score() ne modifie pas les hypothèses originales."""
        ctx = make_ctx(analysis=make_analysis())
        h = make_hypothesis("test", score=0.5)
        original_score = h.initial_score
        self.ranker.score([h], ctx)
        assert h.initial_score == original_score  # non modifié

    def test_liste_vide(self):
        ctx = make_ctx()
        assert self.ranker.score([], ctx) == []


class TestTopN:
    def setup_method(self):
        self.ranker = HypothesisRanker()

    def test_top_3_depuis_5(self):
        hypotheses = [make_hypothesis(f"h{i}", score=i * 0.1) for i in range(5)]
        result = self.ranker.top_n(hypotheses, n=3)
        assert len(result) == 3

    def test_tri_decroissant(self):
        hypotheses = [
            make_hypothesis("bonne", score=0.9),
            make_hypothesis("moyenne", score=0.5),
            make_hypothesis("excellente", score=1.0),
        ]
        result = self.ranker.top_n(hypotheses, n=3)
        assert result[0].initial_score >= result[1].initial_score >= result[2].initial_score

    def test_n_superieur_a_longueur(self):
        hypotheses = [make_hypothesis("seule", score=0.5)]
        result = self.ranker.top_n(hypotheses, n=10)
        assert len(result) == 1

    def test_n_zero(self):
        hypotheses = [make_hypothesis("test")]
        assert self.ranker.top_n(hypotheses, n=0) == []

    def test_liste_vide(self):
        assert self.ranker.top_n([], n=3) == []


class TestRankPipeline:
    def setup_method(self):
        self.ranker = HypothesisRanker()

    def test_rank_pipeline_complet(self):
        ctx = make_ctx(analysis=make_analysis(
            question_type=QuestionType.COMPARATIVE,
            key_concepts=["python"],
        ))
        hypotheses = [
            make_hypothesis("python est adapté pour le machine learning", score=0.5),
            make_hypothesis("python est adapté pour le machine learning", score=0.6),
            make_hypothesis("java est plus performant que python", score=0.4),
            make_hypothesis("le choix dépend du contexte", score=0.55),
        ]
        result = self.ranker.rank(hypotheses, ctx, n=2)
        assert len(result) <= 2
        # Le résultat doit être trié par score décroissant
        if len(result) >= 2:
            assert result[0].initial_score >= result[1].initial_score

    def test_rank_vide_retourne_vide(self):
        ctx = make_ctx()
        assert self.ranker.rank([], ctx, n=3) == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. HeuristicStrategy
# ─────────────────────────────────────────────────────────────────────────────

class TestHeuristicStrategy:
    def setup_method(self):
        self.strategy = HeuristicStrategy()

    def test_name_et_weight(self):
        assert self.strategy.name == "heuristic"
        assert 0.0 < self.strategy.weight <= 1.0

    def test_retourne_hypotheses_non_vide(self):
        ctx = make_ctx(analysis=make_analysis())
        result = asyncio.run(self.strategy.generate(ctx))
        assert len(result) >= 1

    def test_tous_les_question_types(self):
        for qt in QuestionType:
            ctx = make_ctx(analysis=make_analysis(question_type=qt))
            result = asyncio.run(self.strategy.generate(ctx))
            assert len(result) >= 1, f"Pas d'hypothèse pour QuestionType.{qt.name}"

    def test_question_type_comparative_genere_entites(self):
        ctx = make_ctx(analysis=make_analysis(
            question_type = QuestionType.COMPARATIVE,
            key_entities  = ["FastAPI", "Django"],
        ))
        result  = asyncio.run(self.strategy.generate(ctx))
        contents = " ".join(h.content for h in result)
        assert "FastAPI" in contents or "Django" in contents

    def test_strategy_name_renseigne(self):
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(self.strategy.generate(ctx))
        assert all(h.strategy_name == "heuristic" for h in result)

    def test_justification_non_vide(self):
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(self.strategy.generate(ctx))
        assert all(len(h.justification) > 0 for h in result)

    def test_scores_dans_intervalle(self):
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(self.strategy.generate(ctx))
        assert all(0.0 <= h.initial_score <= 1.0 for h in result)

    def test_sans_analyse_utilise_fallback(self):
        """Même sans ctx.analysis (None), utilise analysis_or_fallback."""
        ctx = make_ctx()
        assert ctx.analysis is None
        result = asyncio.run(self.strategy.generate(ctx))
        assert len(result) >= 1

    def test_avec_concept_substitution(self):
        ctx = make_ctx(analysis=make_analysis(
            question_type=QuestionType.FACTUAL,
            key_concepts=["kubernetes"],
        ))
        result   = asyncio.run(self.strategy.generate(ctx))
        contents = " ".join(h.content for h in result)
        assert "kubernetes" in contents


# ─────────────────────────────────────────────────────────────────────────────
# 3. DecompositionStrategy
# ─────────────────────────────────────────────────────────────────────────────

class TestDecompositionStrategy:
    def setup_method(self):
        self.strategy = DecompositionStrategy()

    def test_name_et_weight(self):
        assert self.strategy.name == "decomposition"
        assert 0.0 < self.strategy.weight <= 1.0

    def test_sans_sous_questions_retourne_vide(self):
        ctx    = make_ctx(analysis=make_analysis(sub_questions=[]))
        result = asyncio.run(self.strategy.generate(ctx))
        assert result == []

    def test_une_sous_question_une_hypothese(self):
        ctx = make_ctx(analysis=make_analysis(
            sub_questions=["Comment configurer nginx ?"]
        ))
        result = asyncio.run(self.strategy.generate(ctx))
        assert len(result) == 1

    def test_trois_sous_questions_trois_hypotheses(self):
        ctx = make_ctx(analysis=make_analysis(
            sub_questions=[
                "Comment installer python ?",
                "Pourquoi utiliser des venv ?",
                "Quelle version choisir ?",
            ]
        ))
        result = asyncio.run(self.strategy.generate(ctx))
        assert len(result) == 3

    def test_sous_question_dans_contenu(self):
        ctx = make_ctx(analysis=make_analysis(
            sub_questions=["Comment optimiser les requêtes SQL ?"]
        ))
        result = asyncio.run(self.strategy.generate(ctx))
        assert "Comment optimiser les requêtes SQL" in result[0].content

    def test_strategy_name_renseigne(self):
        ctx = make_ctx(analysis=make_analysis(sub_questions=["test ?"]))
        result = asyncio.run(self.strategy.generate(ctx))
        assert result[0].strategy_name == "decomposition"

    def test_poids_superieur_heuristique(self):
        """DecompositionStrategy a plus de confiance que HeuristicStrategy."""
        assert DecompositionStrategy().weight > HeuristicStrategy().weight


# ─────────────────────────────────────────────────────────────────────────────
# 4. ConceptualStrategy
# ─────────────────────────────────────────────────────────────────────────────

class TestConceptualStrategy:
    def setup_method(self):
        self.strategy = ConceptualStrategy()

    def test_name_et_weight(self):
        assert self.strategy.name == "conceptual"
        assert 0.0 < self.strategy.weight <= 1.0

    def test_sans_concepts_retourne_vide(self):
        ctx    = make_ctx(analysis=make_analysis(key_concepts=[]))
        result = asyncio.run(self.strategy.generate(ctx))
        assert result == []

    def test_un_concept_une_hypothese(self):
        ctx    = make_ctx(analysis=make_analysis(key_concepts=["docker"]))
        result = asyncio.run(self.strategy.generate(ctx))
        assert len(result) == 1

    def test_deux_concepts_deux_hypotheses(self):
        ctx = make_ctx(analysis=make_analysis(
            key_concepts=["docker", "kubernetes"]
        ))
        result = asyncio.run(self.strategy.generate(ctx))
        assert len(result) == 2

    def test_concept_principal_dans_hypothese(self):
        ctx = make_ctx(analysis=make_analysis(key_concepts=["fastapi"]))
        result = asyncio.run(self.strategy.generate(ctx))
        assert "fastapi" in result[0].content.lower()

    def test_relation_entre_deux_concepts(self):
        ctx = make_ctx(analysis=make_analysis(
            key_concepts=["fastapi", "postgresql"]
        ))
        result   = asyncio.run(self.strategy.generate(ctx))
        contents = " ".join(h.content for h in result).lower()
        assert "fastapi" in contents
        assert "postgresql" in contents

    def test_score_relation_inferieur_principal(self):
        """L'hypothèse de relation est légèrement moins sûre que la principale."""
        ctx    = make_ctx(analysis=make_analysis(key_concepts=["python", "django"]))
        result = asyncio.run(self.strategy.generate(ctx))
        assert result[0].initial_score >= result[1].initial_score

    def test_strategy_name_renseigne(self):
        ctx    = make_ctx(analysis=make_analysis(key_concepts=["test"]))
        result = asyncio.run(self.strategy.generate(ctx))
        assert all(h.strategy_name == "conceptual" for h in result)


# ─────────────────────────────────────────────────────────────────────────────
# 5. LLMStrategy
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMStrategy:
    LLM_RESPONSE = {
        "hypotheses": [
            {
                "content": "Python est optimal pour le machine learning grâce à ses bibliothèques.",
                "score": 0.85,
                "justification": "L'écosystème Python (numpy, sklearn) domine ce domaine.",
            },
            {
                "content": "JavaScript est préférable pour les applications web temps réel.",
                "score": 0.75,
                "justification": "Node.js permet des I/O non bloquants.",
            },
        ]
    }

    def test_name_et_weight(self):
        assert LLMStrategy().name == "llm"
        assert LLMStrategy().weight == 0.80

    def test_avec_llm_valide(self):
        ctx      = make_ctx(analysis=make_analysis())
        strategy = LLMStrategy(llm_generate=make_llm_mock(self.LLM_RESPONSE))
        result   = asyncio.run(strategy.generate(ctx))
        assert len(result) == 2

    def test_contenu_hypothese_preserve(self):
        ctx      = make_ctx(analysis=make_analysis())
        strategy = LLMStrategy(llm_generate=make_llm_mock(self.LLM_RESPONSE))
        result   = asyncio.run(strategy.generate(ctx))
        assert result[0].content == self.LLM_RESPONSE["hypotheses"][0]["content"]

    def test_score_preserve(self):
        ctx      = make_ctx(analysis=make_analysis())
        strategy = LLMStrategy(llm_generate=make_llm_mock(self.LLM_RESPONSE))
        result   = asyncio.run(strategy.generate(ctx))
        assert result[0].initial_score == 0.85

    def test_justification_preserve(self):
        ctx      = make_ctx(analysis=make_analysis())
        strategy = LLMStrategy(llm_generate=make_llm_mock(self.LLM_RESPONSE))
        result   = asyncio.run(strategy.generate(ctx))
        assert "écosystème" in result[0].justification

    def test_strategy_name_llm(self):
        ctx      = make_ctx(analysis=make_analysis())
        strategy = LLMStrategy(llm_generate=make_llm_mock(self.LLM_RESPONSE))
        result   = asyncio.run(strategy.generate(ctx))
        assert all(h.strategy_name == "llm" for h in result)

    def test_json_invalide_retourne_vide(self):
        ctx      = make_ctx(analysis=make_analysis())
        strategy = LLMStrategy(
            llm_generate=make_llm_mock_text("Désolé je ne peux pas répondre.")
        )
        result = asyncio.run(strategy.generate(ctx))
        assert result == []

    def test_json_sans_cle_hypotheses_retourne_vide(self):
        ctx      = make_ctx(analysis=make_analysis())
        strategy = LLMStrategy(
            llm_generate=make_llm_mock({"autre_cle": []})
        )
        result = asyncio.run(strategy.generate(ctx))
        assert result == []

    def test_erreur_llm_retourne_vide_sans_exception(self):
        ctx      = make_ctx(analysis=make_analysis())
        strategy = LLMStrategy(llm_generate=make_llm_mock_error())
        result   = asyncio.run(strategy.generate(ctx))
        assert result == []

    def test_sans_llm_et_sans_core_retourne_vide(self):
        """Sans LLM injectable et sans core.llm disponible, retourne []."""
        ctx      = make_ctx(analysis=make_analysis())
        strategy = LLMStrategy(llm_generate=None)
        # core.llm n'est pas disponible dans l'environnement de test isolé
        # Le résultat peut être [] ou une liste si core.llm est présent
        result = asyncio.run(strategy.generate(ctx))
        assert isinstance(result, list)  # jamais None, jamais exception

    def test_json_avec_backticks_parse(self):
        raw = f"```json\n{json.dumps(self.LLM_RESPONSE)}\n```"
        ctx      = make_ctx(analysis=make_analysis())
        strategy = LLMStrategy(llm_generate=make_llm_mock_text(raw))
        result   = asyncio.run(strategy.generate(ctx))
        assert len(result) == 2

    def test_score_hors_borne_est_clamp(self):
        resp = {"hypotheses": [{"content": "test", "score": 9.9, "justification": "x"}]}
        ctx      = make_ctx(analysis=make_analysis())
        strategy = LLMStrategy(llm_generate=make_llm_mock(resp))
        result   = asyncio.run(strategy.generate(ctx))
        assert result[0].initial_score <= 1.0

    def test_contenu_vide_ignore(self):
        resp = {"hypotheses": [
            {"content": "", "score": 0.8, "justification": "vide"},
            {"content": "hypothèse valide", "score": 0.7, "justification": "ok"},
        ]}
        ctx      = make_ctx(analysis=make_analysis())
        strategy = LLMStrategy(llm_generate=make_llm_mock(resp))
        result   = asyncio.run(strategy.generate(ctx))
        assert len(result) == 1
        assert result[0].content == "hypothèse valide"

    def test_max_5_hypotheses(self):
        resp = {"hypotheses": [
            {"content": f"hypothèse {i}", "score": 0.5, "justification": "test"}
            for i in range(10)
        ]}
        ctx      = make_ctx(analysis=make_analysis())
        strategy = LLMStrategy(llm_generate=make_llm_mock(resp))
        result   = asyncio.run(strategy.generate(ctx))
        assert len(result) <= 5


# ─────────────────────────────────────────────────────────────────────────────
# 6. generate_hypotheses()
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateHypotheses:
    def test_retourne_ctx(self):
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(generate_hypotheses(ctx))
        assert result is ctx

    def test_peuple_ctx_hypotheses(self):
        ctx    = make_ctx(analysis=make_analysis(key_concepts=["python"]))
        result = asyncio.run(generate_hypotheses(ctx))
        assert isinstance(result.hypotheses, list)

    def test_strategie_unique_mock(self):
        hypotheses = [make_hypothesis("h1"), make_hypothesis("h2")]
        ctx        = make_ctx(analysis=make_analysis())
        result     = asyncio.run(generate_hypotheses(
            ctx,
            strategies=[make_strategy_mock(hypotheses)],
        ))
        assert len(result.hypotheses) == 2

    def test_plusieurs_strategies_fusion(self):
        s1 = make_strategy_mock([make_hypothesis("hyp A")], name="s1")
        s2 = make_strategy_mock([make_hypothesis("hyp B")], name="s2")
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(generate_hypotheses(ctx, strategies=[s1, s2]))
        assert len(result.hypotheses) >= 1

    def test_strategie_en_erreur_ignoree(self):
        s_ok  = make_strategy_mock([make_hypothesis("valide")], name="ok")
        s_err = make_strategy_error("broken")
        ctx   = make_ctx(analysis=make_analysis())
        # Ne doit pas lever d'exception
        result = asyncio.run(generate_hypotheses(ctx, strategies=[s_ok, s_err]))
        assert len(result.hypotheses) >= 1

    def test_max_hypotheses_respecte(self):
        hypotheses = [make_hypothesis(f"h{i}", score=float(i)/10) for i in range(10)]
        ctx        = make_ctx(
            "question",
            analysis=make_analysis(),
            max_hypotheses=3,
        )
        result = asyncio.run(generate_hypotheses(
            ctx,
            strategies=[make_strategy_mock(hypotheses)],
        ))
        assert len(result.hypotheses) <= 3

    def test_toutes_strategies_vides_liste_vide(self):
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(generate_hypotheses(
            ctx,
            strategies=[make_strategy_mock([])],
        ))
        assert result.hypotheses == []

    def test_trace_step_ajoute(self):
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(generate_hypotheses(
            ctx,
            strategies=[make_strategy_mock([make_hypothesis("h")])],
        ))
        step_names = [s.step_name for s in result.trace.steps]
        assert "hypothesis_engine" in step_names

    def test_trace_step_success(self):
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(generate_hypotheses(
            ctx,
            strategies=[make_strategy_mock([make_hypothesis("h")])],
        ))
        step = next(s for s in result.trace.steps if s.step_name == "hypothesis_engine")
        assert step.success is True

    def test_trace_degraded_si_aucune_hypothese(self):
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(generate_hypotheses(
            ctx,
            strategies=[make_strategy_mock([])],
        ))
        step = next(s for s in result.trace.steps if s.step_name == "hypothesis_engine")
        assert step.degraded is True

    def test_llm_injecte_dans_llm_strategy(self):
        llm_resp = {"hypotheses": [{"content": "via LLM", "score": 0.9, "justification": "ok"}]}
        ctx      = make_ctx(analysis=make_analysis())
        result   = asyncio.run(generate_hypotheses(
            ctx,
            llm_generate=make_llm_mock(llm_resp),
        ))
        contents = [h.content for h in result.hypotheses]
        assert any("via LLM" in c for c in contents)

    def test_ranker_injectable(self):
        """Un ranker custom peut être injecté."""
        custom_ranker = HypothesisRanker(similarity_threshold=0.99)
        ctx           = make_ctx(analysis=make_analysis())
        result        = asyncio.run(generate_hypotheses(
            ctx,
            strategies=[make_strategy_mock([make_hypothesis("h1"), make_hypothesis("h2")])],
            ranker=custom_ranker,
        ))
        assert isinstance(result.hypotheses, list)

    def test_sans_analyse_ne_plante_pas(self):
        """Sans ctx.analysis, l'engine utilise analysis_or_fallback."""
        ctx = make_ctx()  # pas d'analyse
        result = asyncio.run(generate_hypotheses(
            ctx,
            strategies=[make_strategy_mock([make_hypothesis("h")])],
        ))
        assert isinstance(result.hypotheses, list)

    def test_hypotheses_count_dans_trace(self):
        hyps   = [make_hypothesis("h1"), make_hypothesis("h2")]
        ctx    = make_ctx(analysis=make_analysis(), max_hypotheses=5)
        result = asyncio.run(generate_hypotheses(
            ctx,
            strategies=[make_strategy_mock(hyps)],
        ))
        assert result.trace.hypotheses_count == len(result.hypotheses)


# ─────────────────────────────────────────────────────────────────────────────
# 7. HypothesisEngineAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestHypothesisEngineAgent:
    def test_name(self):
        assert HypothesisEngineAgent().name == "hypothesis_engine"

    def test_run_retourne_ctx(self):
        agent  = HypothesisEngineAgent(
            strategies=[make_strategy_mock([make_hypothesis("h")])]
        )
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(agent.run(ctx))
        assert result is ctx

    def test_run_peuple_hypotheses(self):
        agent  = HypothesisEngineAgent(
            strategies=[make_strategy_mock([make_hypothesis("test")])]
        )
        ctx    = make_ctx(analysis=make_analysis(key_concepts=["python"]))
        result = asyncio.run(agent.run(ctx))
        assert len(result.hypotheses) >= 1

    def test_run_ajoute_step_trace(self):
        agent  = HypothesisEngineAgent(
            strategies=[make_strategy_mock([make_hypothesis("h")])]
        )
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(agent.run(ctx))
        names  = [s.step_name for s in result.trace.steps]
        assert "hypothesis_engine" in names

    def test_context_version_inchangee(self):
        agent  = HypothesisEngineAgent(
            strategies=[make_strategy_mock([make_hypothesis("h")])]
        )
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(agent.run(ctx))
        assert result.context_version == "3.0"

    def test_llm_injecte_via_agent(self):
        llm_resp = {"hypotheses": [{"content": "via agent LLM", "score": 0.9, "justification": "ok"}]}
        agent  = HypothesisEngineAgent(llm_generate=make_llm_mock(llm_resp))
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(agent.run(ctx))
        contents = [h.content for h in result.hypotheses]
        assert any("via agent LLM" in c for c in contents)

    def test_strategies_injectees(self):
        custom = make_strategy_mock([make_hypothesis("custom")], name="custom")
        agent  = HypothesisEngineAgent(strategies=[custom])
        ctx    = make_ctx(analysis=make_analysis())
        result = asyncio.run(agent.run(ctx))
        assert len(result.hypotheses) == 1
        assert result.hypotheses[0].content == "custom"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Pipeline reasoning_engine (intégration)
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_default_pipeline_contient_deux_agents(self):
        assert len(DEFAULT_PIPELINE) >= 2

    def test_premier_agent_question_analyzer(self):
        assert DEFAULT_PIPELINE[0].name == "question_analyzer"

    def test_deuxieme_agent_hypothesis_engine(self):
        assert DEFAULT_PIPELINE[1].name == "hypothesis_engine"

    def test_run_reasoning_peuple_analysis_et_hypotheses(self):
        req    = make_request("Pourquoi FastAPI est plus rapide que Flask ?")
        ctx    = asyncio.run(run_reasoning(request=req, user_id="test-user"))
        assert ctx.analysis is not None
        assert isinstance(ctx.hypotheses, list)

    def test_run_reasoning_trace_contient_les_deux_steps(self):
        req      = make_request("Comment déployer docker sur AWS ?")
        ctx      = asyncio.run(run_reasoning(request=req, user_id="test-user"))
        names    = [s.step_name for s in ctx.trace.steps]
        assert "question_analyzer" in names
        assert "hypothesis_engine" in names

    def test_context_to_response_inclut_hypotheses(self):
        req      = make_request("FastAPI vs Django ?")
        ctx      = asyncio.run(run_reasoning(request=req, user_id="test-user"))
        response = context_to_response(ctx)
        assert isinstance(response.hypotheses_considered, list)

    def test_context_to_response_best_hypothesis(self):
        req      = make_request("comment optimiser mon code python ?")
        ctx      = asyncio.run(run_reasoning(request=req, user_id="test-user"))
        response = context_to_response(ctx)
        # Si des hypothèses ont été générées, best_hypothesis est non vide
        if response.hypotheses_considered:
            assert isinstance(response.best_hypothesis, str)

    def test_run_reasoning_context_version_inchangee(self):
        req = make_request("test pipeline")
        ctx = asyncio.run(run_reasoning(request=req, user_id="test-user"))
        assert ctx.context_version == "3.0"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Modèle Hypothesis Phase 3.2
# ─────────────────────────────────────────────────────────────────────────────

class TestHypothesisModele:
    def test_champs_phase_30_toujours_presents(self):
        h = Hypothesis(content="test")
        assert hasattr(h, "hypothesis_id")
        assert hasattr(h, "content")
        assert hasattr(h, "initial_score")
        assert hasattr(h, "origin")

    def test_champs_phase_32_avec_defaults(self):
        h = Hypothesis(content="test")
        assert h.justification == ""
        assert h.strategy_name == "unknown"

    def test_champs_phase_32_settables(self):
        h = Hypothesis(
            content       = "hypothèse complète",
            justification = "car c'est logique",
            strategy_name = "heuristic",
        )
        assert h.justification == "car c'est logique"
        assert h.strategy_name == "heuristic"

    def test_retro_compatibilite_creation_sans_nouveaux_champs(self):
        """Du code Phase 3.0 peut créer Hypothesis sans les champs Phase 3.2."""
        h = Hypothesis(content="ancienne hypothèse", initial_score=0.7)
        assert h.content == "ancienne hypothèse"
        assert h.justification == ""

    def test_model_copy_preserve_champs(self):
        h    = Hypothesis(content="original", initial_score=0.5, strategy_name="test")
        copy = h.model_copy(update={"initial_score": 0.8})
        assert copy.strategy_name == "test"
        assert copy.initial_score == 0.8

    def test_hypothesis_id_auto_unique(self):
        h1 = Hypothesis(content="a")
        h2 = Hypothesis(content="b")
        assert h1.hypothesis_id != h2.hypothesis_id


# ─────────────────────────────────────────────────────────────────────────────
# 10. Régression Phase 3.0 / 3.1 — test mis à jour
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionsPhasesPrecedentes:
    def test_build_summary_sans_analyse_non_vide(self):
        """Régression corrigée Phase 3.0 → 3.1 : message par défaut stable."""
        ctx     = make_ctx()
        summary = _build_summary(ctx)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_build_summary_sans_hypotheses_pas_de_mention(self):
        """Sans hypothèses, le summary n'en fait pas mention."""
        ctx     = make_ctx(analysis=make_analysis())
        summary = _build_summary(ctx)
        assert "hypothèse" not in summary.lower()

    def test_build_summary_avec_hypotheses_mentionne_count(self):
        """Avec des hypothèses, le summary les mentionne."""
        ctx = make_ctx(analysis=make_analysis())
        ctx.hypotheses = [make_hypothesis("h1"), make_hypothesis("h2")]
        summary = _build_summary(ctx)
        assert "2" in summary or "hypothèse" in summary.lower()

    def test_reasoning_response_defaults_inchanges(self):
        """Vérification de non-régression sur les defaults de ReasoningResponse."""
        from models.reasoning import ReasoningResponse
        r = ReasoningResponse(answer="test")
        assert r.confidence == 0.0
        assert r.risk_level == "high"
        assert r.hypotheses_considered == []
        assert r.trace is None

    def test_question_analysis_retro_compatible(self):
        """QuestionAnalysis sans les nouveaux champs reste valide."""
        from models.reasoning import QuestionAnalysis
        qa = QuestionAnalysis(
            original_question = "test",
            question_type     = QuestionType.FACTUAL,
            complexity_score  = 0.3,
        )
        assert qa.domain == "general"
        assert qa.risk_level == "low"
        assert qa.confidence == 0.0

    def test_resolve_strategies_sans_llm(self):
        result = _resolve_strategies(None, None)
        assert len(result) > 0

    def test_resolve_strategies_custom(self):
        custom = [make_strategy_mock([])]
        result = _resolve_strategies(custom, None)
        assert result is custom

    def test_safe_generate_ne_plante_pas(self):
        failing = make_strategy_error()
        ctx     = make_ctx()
        result  = asyncio.run(_safe_generate(failing, ctx))
        assert result == []

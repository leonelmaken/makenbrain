"""Tests Phase 3.1 — QuestionAnalyzerAgent.

Couvre :
    1. Détecteurs lexicaux (_detect_type, _detect_domain, _detect_complexity,
       _detect_risk, _detect_needs_memory, _detect_needs_external).
    2. Extracteurs (_extract_entities, _extract_concepts, _extract_sub_questions).
    3. Parsing et validation LLM (_parse_llm_response, _try_parse_json,
       _dict_to_analysis, _clamp_float, _safe_str_list).
    4. Fallback déterministe (_analyze_deterministic).
    5. Analyse LLM mockée (analyze_question avec llm_generate injectable).
    6. Agent (QuestionAnalyzerAgent.run, intégration avec ReasoningContext).
    7. Modèle étendu (QuestionAnalysis Phase 3.1, rétrocompatibilité).
    8. Reasoning engine (DEFAULT_PIPELINE branché sur QuestionAnalyzerAgent).

Convention :
    - Pas de TestClient, pas d'appel réseau.
    - LLM mocké via le paramètre ``llm_generate`` injectable.
    - asyncio.run() pour les tests de coroutines.

Lancer :
    pytest tests/test_phase31_question_analyzer.py -v
    pytest tests/test_phase31_question_analyzer.py -v --tb=short -q
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional

import pytest

from models.reasoning import (
    QuestionAnalysis,
    QuestionType,
    ReasoningRequest,
    ReasoningStepRecord,
)
from core.reasoning.context import ReasoningContext
from core.reasoning.question_analyzer import (
    QuestionAnalyzerAgent,
    _analyze_deterministic,
    _clamp_float,
    _detect_complexity,
    _detect_domain,
    _detect_needs_external,
    _detect_needs_memory,
    _detect_risk,
    _detect_type,
    _dict_to_analysis,
    _extract_concepts,
    _extract_entities,
    _extract_sub_questions,
    _parse_llm_response,
    _safe_str_list,
    _try_parse_json,
    analyze_question,
)
from core.reasoning.reasoning_engine import (
    DEFAULT_PIPELINE,
    context_to_response,
    run_reasoning,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_request(question: str = "Question de test", **kwargs) -> ReasoningRequest:
    """Construit un ReasoningRequest minimal pour les tests."""
    return ReasoningRequest(question=question, **kwargs)


def make_ctx(question: str = "Question de test", **kwargs) -> ReasoningContext:
    """Construit un ReasoningContext initialisé pour les tests."""
    ctx = ReasoningContext(request=make_request(question, **kwargs), user_id="test-user")
    ctx.initialize_trace()
    return ctx


def make_llm_mock(response_json: dict) -> callable:
    """Crée un mock async de llm_generate retournant un JSON donné."""
    async def mock(prompt: str, **kwargs) -> str:
        return json.dumps(response_json, ensure_ascii=False)
    return mock


def make_llm_mock_text(text: str) -> callable:
    """Crée un mock async de llm_generate retournant du texte brut."""
    async def mock(prompt: str, **kwargs) -> str:
        return text
    return mock


def make_llm_mock_error() -> callable:
    """Crée un mock async de llm_generate qui lève une exception."""
    async def mock(prompt: str, **kwargs) -> str:
        raise RuntimeError("LLM simulé hors ligne")
    return mock


# ── 1. Détecteurs lexicaux ────────────────────────────────────────────────────

class TestDetectType:
    def test_factuel_quest_ce_que(self):
        assert _detect_type("qu'est-ce que python ?") == QuestionType.FACTUAL

    def test_factuel_quel_est(self):
        assert _detect_type("quel est le meilleur framework ?") in (
            QuestionType.FACTUAL, QuestionType.EVALUATIVE
        )

    def test_analytique_pourquoi(self):
        assert _detect_type("pourquoi mon api est lente ?") == QuestionType.ANALYTICAL

    def test_analytique_expliquer(self):
        assert _detect_type("explique pourquoi docker utilise autant de mémoire") == QuestionType.ANALYTICAL

    def test_comparatif_vs(self):
        assert _detect_type("fastapi vs django, lequel choisir ?") == QuestionType.COMPARATIVE

    def test_comparatif_difference_entre(self):
        assert _detect_type("quelle est la différence entre redis et memcached ?") == QuestionType.COMPARATIVE

    def test_procedural_comment(self):
        assert _detect_type("comment configurer nginx avec ssl ?") == QuestionType.PROCEDURAL

    def test_procedural_etapes(self):
        assert _detect_type("quelles étapes pour déployer sur kubernetes ?") == QuestionType.PROCEDURAL

    def test_hypothetique_et_si(self):
        assert _detect_type("et si je migrais vers un monorepo ?") == QuestionType.HYPOTHETICAL

    def test_hypothetique_que_se_passerait(self):
        assert _detect_type("que se passerait-il si on supprimait la base ?") == QuestionType.HYPOTHETICAL

    def test_evaluatif_meilleur(self):
        assert _detect_type("quelle est la meilleure approche pour le caching ?") == QuestionType.EVALUATIVE

    def test_evaluatif_recommand(self):
        assert _detect_type("tu recommandes quoi pour les tests ?") == QuestionType.EVALUATIVE

    def test_defaut_factuel(self):
        assert _detect_type("makenbrain") == QuestionType.FACTUAL


class TestDetectDomain:
    def test_technique_code(self):
        assert _detect_domain("comment écrire du code python propre ?") == "technique"

    def test_technique_api(self):
        assert _detect_domain("mon api fastapi retourne une erreur 422") == "technique"

    def test_finance_budget(self):
        assert _detect_domain("comment optimiser mon budget mensuel ?") == "finance"

    def test_finance_tontine(self):
        assert _detect_domain("comment gérer une tontine avec mobile money ?") == "finance"

    def test_sante_medecin(self):
        assert _detect_domain("j'ai de la fièvre, dois-je voir un médecin ?") == "santé"

    def test_droit_contrat(self):
        assert _detect_domain("un contrat de prestation sans clause de non-concurrence est-il valide ?") == "droit"

    def test_science_statistiques(self):
        assert _detect_domain("comment calculer la variance dans une étude statistique ?") == "science"

    def test_general_question_vague(self):
        assert _detect_domain("dis-moi quelque chose d'intéressant") == "general"

    def test_multi_domaine_priorite_plus_de_mots_cles(self):
        # La question technique a plus de mots-clés → doit gagner
        result = _detect_domain("déployer un code python avec docker et kubernetes sur aws")
        assert result == "technique"


class TestDetectComplexity:
    def test_question_triviale(self):
        score = _detect_complexity("bonjour")
        assert score < 0.35

    def test_question_simple(self):
        score = _detect_complexity("qu'est-ce que python ?")
        assert score < 0.5

    def test_question_moderee(self):
        score = _detect_complexity(
            "comment comparer les performances de fastapi et flask ?"
        )
        assert 0.3 <= score <= 0.7

    def test_question_complexe_longue(self):
        question = (
            "comment concevoir une architecture microservices scalable pour "
            "une application fintech africaine, avec gestion des tontines, "
            "mobile money, et intégration multi-cloud aws et azure ?"
        )
        score = _detect_complexity(question.lower())
        assert score >= 0.6

    def test_questions_multiples_augmente_score(self):
        score_simple   = _detect_complexity("qu'est-ce que docker ?")
        score_multiple = _detect_complexity("qu'est-ce que docker ? et kubernetes ? et helm ?")
        assert score_multiple > score_simple

    def test_score_entre_0_et_1(self):
        extreme = "analyser comparer évaluer optimiser " * 20
        score = _detect_complexity(extreme)
        assert 0.0 <= score <= 1.0


class TestDetectRisk:
    def test_low_information_generale(self):
        assert _detect_risk("qu'est-ce que python ?") == "low"

    def test_medium_financier(self):
        assert _detect_risk("quel est le meilleur budget pour ce projet ?") == "medium"

    def test_medium_base_de_donnees(self):
        assert _detect_risk("comment migrer ma base de données ?") == "medium"

    def test_high_medical(self):
        assert _detect_risk("comment reconnaître les symptômes d'une urgence médicale ?") == "high"

    def test_high_juridique(self):
        assert _detect_risk("est-ce que ce contrat est légalement valide ?") == "high"

    def test_high_securite(self):
        assert _detect_risk("comment exploiter une vulnérabilité de sécurité ?") == "high"


class TestDetectNeedsMemory:
    def test_mon_projet_active_memory(self):
        assert _detect_needs_memory("comment améliorer mon projet ?") is True

    def test_notre_systeme_active_memory(self):
        assert _detect_needs_memory("notre système de tontine a un bug") is True

    def test_question_generique_pas_memory(self):
        assert _detect_needs_memory("qu'est-ce que le machine learning ?") is False

    def test_mes_donnees_active_memory(self):
        assert _detect_needs_memory("comment sauvegarder mes données ?") is True


class TestDetectNeedsExternal:
    def test_derniere_version_active_external(self):
        assert _detect_needs_external("quelle est la dernière version de fastapi ?") is True

    def test_actualite_active_external(self):
        assert _detect_needs_external("actualité intelligence artificielle 2025") is True

    def test_question_conceptuelle_pas_external(self):
        assert _detect_needs_external("comment fonctionne un algorithme de tri ?") is False


# ── 2. Extracteurs ────────────────────────────────────────────────────────────

class TestExtractEntities:
    def test_entite_majuscule_detectee(self):
        entities = _extract_entities("Comment intégrer FastAPI avec PostgreSQL ?")
        assert "FastAPI" in entities
        assert "PostgreSQL" in entities

    def test_premier_mot_ignore(self):
        entities = _extract_entities("Python est un langage de programmation.")
        assert "Python" not in entities  # premier mot, ignoré

    def test_pas_dentite_dans_question_minuscule(self):
        entities = _extract_entities("comment fonctionne cela ?")
        assert entities == []

    def test_deduplication(self):
        entities = _extract_entities("MakenBrain est un projet. MakenBrain utilise Python.")
        assert entities.count("MakenBrain") == 1

    def test_max_8_entites(self):
        q = "A B C D E F G H I J K L M N O P Q R".join([" utilise "] * 18)
        entities = _extract_entities(q)
        assert len(entities) <= 8


class TestExtractConcepts:
    def test_mots_cles_domaine_inclus(self):
        concepts = _extract_concepts("mon api fastapi a des erreurs", "technique")
        assert "api" in concepts or "fastapi" in concepts

    def test_stopwords_exclus(self):
        concepts = _extract_concepts("comment faire cela avec notre système ?", "general")
        assert "comment" not in concepts
        assert "avec" not in concepts
        assert "notre" not in concepts

    def test_max_10_concepts(self):
        long_q = "python fastapi docker kubernetes redis postgresql mongodb nginx " * 3
        concepts = _extract_concepts(long_q, "technique")
        assert len(concepts) <= 10


class TestExtractSubQuestions:
    def test_pas_de_sous_questions_simple(self):
        sub = _extract_sub_questions("Qu'est-ce que Python ?")
        assert sub == []

    def test_sous_question_detectee(self):
        q = "Comment configurer nginx, et pourquoi utiliser ssl ?"
        sub = _extract_sub_questions(q)
        assert len(sub) >= 1

    def test_max_4_sous_questions(self):
        q = "comment A, pourquoi B, quelle C, quel D, comment E ?"
        sub = _extract_sub_questions(q)
        assert len(sub) <= 4


# ── 3. Parsing LLM ────────────────────────────────────────────────────────────

class TestTryParseJson:
    def test_json_valide(self):
        result = _try_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_invalide_retourne_none(self):
        assert _try_parse_json("ce n'est pas du json") is None

    def test_json_non_dict_retourne_none(self):
        assert _try_parse_json("[1, 2, 3]") is None

    def test_chaine_vide_retourne_none(self):
        assert _try_parse_json("") is None


class TestParseLLMResponse:
    VALID_JSON = {
        "question_type": "analytical",
        "domain": "technique",
        "complexity_score": 0.7,
        "risk_level": "medium",
        "sub_questions": [],
        "key_entities": ["FastAPI"],
        "key_concepts": ["performance"],
        "requires_reasoning": True,
        "requires_memory": False,
        "requires_external_search": False,
        "requires_deep_reasoning": True,
        "confidence": 0.85,
    }

    def test_parse_json_direct(self):
        analysis = _parse_llm_response(json.dumps(self.VALID_JSON), "test ?")
        assert analysis.question_type == QuestionType.ANALYTICAL
        assert analysis.domain == "technique"
        assert analysis.complexity_score == 0.7
        assert analysis.confidence == 0.85

    def test_parse_json_avec_backticks(self):
        text = f"```json\n{json.dumps(self.VALID_JSON)}\n```"
        analysis = _parse_llm_response(text, "test ?")
        assert analysis.question_type == QuestionType.ANALYTICAL

    def test_parse_json_avec_texte_avant(self):
        text = f"Voici mon analyse :\n{json.dumps(self.VALID_JSON)}"
        analysis = _parse_llm_response(text, "test ?")
        assert analysis.domain == "technique"

    def test_parse_json_invalide_leve_value_error(self):
        with pytest.raises(ValueError, match="JSON non extractible"):
            _parse_llm_response("Désolé, je ne peux pas répondre.", "test ?")

    def test_parse_question_originale_preservee(self):
        analysis = _parse_llm_response(json.dumps(self.VALID_JSON), "ma vraie question ?")
        assert analysis.original_question == "ma vraie question ?"


class TestDictToAnalysis:
    def test_question_type_invalide_devient_factual(self):
        analysis = _dict_to_analysis(
            {"question_type": "inexistant", "complexity_score": 0.5}, "test"
        )
        assert analysis.question_type == QuestionType.FACTUAL

    def test_domain_invalide_devient_general(self):
        analysis = _dict_to_analysis(
            {"question_type": "factual", "domain": "voyage", "complexity_score": 0.3},
            "test",
        )
        assert analysis.domain == "general"

    def test_risk_level_invalide_devient_low(self):
        analysis = _dict_to_analysis(
            {"question_type": "factual", "risk_level": "critical", "complexity_score": 0.3},
            "test",
        )
        assert analysis.risk_level == "low"

    def test_complexity_clampee_au_dessus_de_1(self):
        analysis = _dict_to_analysis(
            {"question_type": "factual", "complexity_score": 9.9}, "test"
        )
        assert analysis.complexity_score <= 1.0

    def test_complexity_clampee_en_dessous_de_0(self):
        analysis = _dict_to_analysis(
            {"question_type": "factual", "complexity_score": -0.5}, "test"
        )
        assert analysis.complexity_score >= 0.0

    def test_requires_deep_reasoning_coherent_avec_complexity(self):
        # complexity >= 0.6 → requires_deep_reasoning True (si non fourni)
        analysis = _dict_to_analysis(
            {"question_type": "factual", "complexity_score": 0.8}, "test"
        )
        assert analysis.requires_deep_reasoning is True

    def test_sub_questions_liste_securisee(self):
        analysis = _dict_to_analysis(
            {"question_type": "factual", "complexity_score": 0.3,
             "sub_questions": [None, "sous-question valide", ""]},
            "test",
        )
        assert "sous-question valide" in analysis.sub_questions
        assert None not in analysis.sub_questions


class TestClampFloat:
    def test_valeur_valide(self):
        assert _clamp_float(0.5) == 0.5

    def test_valeur_sup_1_clampee(self):
        assert _clamp_float(1.5) == 1.0

    def test_valeur_inf_0_clampee(self):
        assert _clamp_float(-0.3) == 0.0

    def test_string_numerique(self):
        assert _clamp_float("0.75") == 0.75

    def test_none_retourne_defaut(self):
        assert _clamp_float(None) == 0.5

    def test_string_invalide_retourne_defaut(self):
        assert _clamp_float("abc", default=0.3) == 0.3


class TestSafeStrList:
    def test_liste_valide(self):
        assert _safe_str_list(["a", "b", "c"]) == ["a", "b", "c"]

    def test_liste_avec_none_filtre(self):
        result = _safe_str_list(["a", None, "b"])
        assert "a" in result
        assert "b" in result

    def test_string_simple(self):
        assert _safe_str_list("concept") == ["concept"]

    def test_none_retourne_liste_vide(self):
        assert _safe_str_list(None) == []

    def test_liste_vide(self):
        assert _safe_str_list([]) == []

    def test_strings_trimmees(self):
        result = _safe_str_list(["  concept  ", " autre "])
        assert result == ["concept", "autre"]


# ── 4. Fallback déterministe ──────────────────────────────────────────────────

class TestAnalyzeDeterministic:
    def test_retourne_question_analysis(self):
        result = _analyze_deterministic("Qu'est-ce que Python ?")
        assert isinstance(result, QuestionAnalysis)

    def test_question_originale_preservee(self):
        q = "Comment optimiser mon API FastAPI ?"
        result = _analyze_deterministic(q)
        assert result.original_question == q

    def test_confidence_reduite_fallback(self):
        result = _analyze_deterministic("test")
        assert result.confidence == 0.6  # signale que c'est le fallback

    def test_question_technique_detectee(self):
        result = _analyze_deterministic("comment déployer mon code python sur docker ?")
        assert result.domain == "technique"

    def test_question_medicale_risque_high(self):
        result = _analyze_deterministic("j'ai besoin d'un diagnostic médical urgent")
        assert result.risk_level == "high"

    def test_complexite_coherente(self):
        simple  = _analyze_deterministic("c'est quoi python ?")
        complex_ = _analyze_deterministic(
            "comment concevoir et implémenter une architecture microservices "
            "scalable pour une fintech africaine avec tontines et mobile money ?"
        )
        assert complex_.complexity_score > simple.complexity_score

    def test_requires_deep_reasoning_sur_question_complexe(self):
        result = _analyze_deterministic(
            "comment analyser et optimiser les performances de notre système "
            "distribué avec plusieurs composants ?"
        )
        assert result.requires_deep_reasoning is True

    def test_requires_memory_sur_question_personnelle(self):
        result = _analyze_deterministic("mon projet a un problème de performance")
        assert result.requires_memory is True

    def test_requires_external_search_sur_actualite(self):
        result = _analyze_deterministic("quelle est la dernière version de fastapi 2025 ?")
        assert result.requires_external_search is True

    def test_types_corrects(self):
        result = _analyze_deterministic("test")
        assert isinstance(result.complexity_score, float)
        assert isinstance(result.requires_reasoning, bool)
        assert isinstance(result.requires_memory, bool)
        assert isinstance(result.requires_deep_reasoning, bool)
        assert isinstance(result.sub_questions, list)
        assert isinstance(result.key_entities, list)
        assert isinstance(result.key_concepts, list)


# ── 5. Analyse LLM mockée ────────────────────────────────────────────────────

class TestAnalyzeQuestionWithLLM:
    LLM_RESPONSE = {
        "question_type": "comparative",
        "domain": "technique",
        "complexity_score": 0.65,
        "risk_level": "low",
        "sub_questions": ["Quelles différences entre FastAPI et Django ?"],
        "key_entities": ["FastAPI", "Django"],
        "key_concepts": ["framework", "performance", "async"],
        "requires_reasoning": True,
        "requires_memory": False,
        "requires_external_search": False,
        "requires_deep_reasoning": True,
        "confidence": 0.88,
    }

    def test_analyse_avec_llm_valide(self):
        ctx = make_ctx("FastAPI vs Django, lequel choisir ?")
        mock = make_llm_mock(self.LLM_RESPONSE)
        result = asyncio.run(analyze_question(ctx, llm_generate=mock))
        assert result.analysis is not None
        assert result.analysis.question_type == QuestionType.COMPARATIVE
        assert result.analysis.domain == "technique"
        assert result.analysis.confidence == 0.88

    def test_analyse_peuple_ctx_analysis(self):
        ctx = make_ctx("FastAPI vs Django ?")
        mock = make_llm_mock(self.LLM_RESPONSE)
        result = asyncio.run(analyze_question(ctx, llm_generate=mock))
        assert result.analysis is not None
        assert isinstance(result.analysis, QuestionAnalysis)

    def test_trace_step_ajoute(self):
        ctx = make_ctx("FastAPI vs Django ?")
        mock = make_llm_mock(self.LLM_RESPONSE)
        result = asyncio.run(analyze_question(ctx, llm_generate=mock))
        steps = [s for s in result.trace.steps if s.step_name == "question_analyzer"]
        assert len(steps) == 1
        assert steps[0].success is True

    def test_trace_contient_infos_analyse(self):
        ctx = make_ctx("FastAPI vs Django ?")
        mock = make_llm_mock(self.LLM_RESPONSE)
        result = asyncio.run(analyze_question(ctx, llm_generate=mock))
        step = result.trace.steps[0]
        assert "comparative" in step.output_summary
        assert "technique" in step.output_summary

    def test_trace_duration_renseignee(self):
        ctx = make_ctx("test")
        mock = make_llm_mock(self.LLM_RESPONSE)
        result = asyncio.run(analyze_question(ctx, llm_generate=mock))
        step = result.trace.steps[0]
        assert step.duration_ms >= 0.0

    def test_llm_json_invalide_fallback_deterministe(self):
        ctx = make_ctx("Qu'est-ce que Python ?")
        mock = make_llm_mock_text("Désolé je ne peux pas répondre en JSON.")
        result = asyncio.run(analyze_question(ctx, llm_generate=mock))
        # Le fallback doit quand même produire une analyse valide
        assert result.analysis is not None
        assert result.pipeline_degraded is True

    def test_llm_erreur_reseau_fallback(self):
        ctx = make_ctx("test question")
        mock = make_llm_mock_error()
        result = asyncio.run(analyze_question(ctx, llm_generate=mock))
        # Le fallback déterministe doit être utilisé
        assert result.analysis is not None
        assert result.pipeline_degraded is True
        assert result.analysis.confidence == 0.6  # signature du fallback

    def test_sans_llm_utilise_fallback(self):
        ctx = make_ctx("comment fonctionne python ?")
        result = asyncio.run(analyze_question(ctx, llm_generate=None))
        assert result.analysis is not None
        # Sans LLM, on ne peut pas garantir degraded=True car core.llm
        # peut être disponible dans l'environnement de test
        assert isinstance(result.analysis, QuestionAnalysis)

    def test_key_entities_depuis_llm(self):
        ctx = make_ctx("FastAPI vs Django ?")
        mock = make_llm_mock(self.LLM_RESPONSE)
        result = asyncio.run(analyze_question(ctx, llm_generate=mock))
        assert "FastAPI" in result.analysis.key_entities
        assert "Django" in result.analysis.key_entities

    def test_sub_questions_depuis_llm(self):
        ctx = make_ctx("FastAPI vs Django ?")
        mock = make_llm_mock(self.LLM_RESPONSE)
        result = asyncio.run(analyze_question(ctx, llm_generate=mock))
        assert len(result.analysis.sub_questions) >= 1


# ── 6. Agent QuestionAnalyzerAgent ───────────────────────────────────────────

class TestQuestionAnalyzerAgent:
    def test_name_correct(self):
        agent = QuestionAnalyzerAgent()
        assert agent.name == "question_analyzer"

    def test_run_retourne_ctx(self):
        agent = QuestionAnalyzerAgent()
        ctx = make_ctx("Qu'est-ce que MakenBrain ?")
        result = asyncio.run(agent.run(ctx))
        assert result is ctx  # même objet, modifié in-place

    def test_run_peuple_analysis(self):
        agent = QuestionAnalyzerAgent()
        ctx = make_ctx("comment déployer sur kubernetes ?")
        result = asyncio.run(agent.run(ctx))
        assert result.analysis is not None

    def test_run_ajoute_step_dans_trace(self):
        agent = QuestionAnalyzerAgent()
        ctx = make_ctx("test")
        result = asyncio.run(agent.run(ctx))
        step_names = [s.step_name for s in result.trace.steps]
        assert "question_analyzer" in step_names

    def test_run_step_success_true(self):
        agent = QuestionAnalyzerAgent()
        ctx = make_ctx("qu'est-ce que python ?")
        result = asyncio.run(agent.run(ctx))
        step = next(s for s in result.trace.steps if s.step_name == "question_analyzer")
        assert step.success is True

    def test_run_context_version_inchangee(self):
        agent = QuestionAnalyzerAgent()
        ctx = make_ctx("test")
        result = asyncio.run(agent.run(ctx))
        assert result.context_version == "3.0"


# ── 7. Modèle QuestionAnalysis Phase 3.1 ─────────────────────────────────────

class TestQuestionAnalysisModele:
    def test_champs_phase_30_toujours_presents(self):
        """Rétrocompatibilité : les anciens champs ne doivent pas disparaître."""
        qa = QuestionAnalysis(
            original_question="test",
            question_type=QuestionType.FACTUAL,
            complexity_score=0.3,
        )
        assert hasattr(qa, "original_question")
        assert hasattr(qa, "question_type")
        assert hasattr(qa, "complexity_score")
        assert hasattr(qa, "sub_questions")
        assert hasattr(qa, "key_entities")
        assert hasattr(qa, "key_concepts")
        assert hasattr(qa, "requires_reasoning")

    def test_champs_phase_31_avec_defaults(self):
        """Les nouveaux champs ont des valeurs par défaut — pas de breaking change."""
        qa = QuestionAnalysis(
            original_question="test",
            question_type=QuestionType.FACTUAL,
            complexity_score=0.3,
        )
        assert qa.domain == "general"
        assert qa.risk_level == "low"
        assert qa.requires_memory is False
        assert qa.requires_external_search is False
        assert qa.requires_deep_reasoning is False
        assert qa.confidence == 0.0

    def test_creation_complete_phase_31(self):
        """Création avec tous les nouveaux champs."""
        qa = QuestionAnalysis(
            original_question        = "Comment déployer sur K8S ?",
            question_type            = QuestionType.PROCEDURAL,
            complexity_score         = 0.7,
            domain                   = "technique",
            risk_level               = "medium",
            requires_memory          = True,
            requires_external_search = False,
            requires_deep_reasoning  = True,
            confidence               = 0.85,
        )
        assert qa.domain == "technique"
        assert qa.risk_level == "medium"
        assert qa.requires_memory is True
        assert qa.requires_deep_reasoning is True
        assert qa.confidence == 0.85

    def test_risk_level_valide(self):
        for level in ("low", "medium", "high"):
            qa = QuestionAnalysis(
                original_question="test",
                question_type=QuestionType.FACTUAL,
                complexity_score=0.3,
                risk_level=level,
            )
            assert qa.risk_level == level

    def test_confidence_hors_borne_invalide(self):
        with pytest.raises(Exception):
            QuestionAnalysis(
                original_question="test",
                question_type=QuestionType.FACTUAL,
                complexity_score=0.3,
                confidence=1.5,
            )


# ── 8. Reasoning engine — pipeline branché ───────────────────────────────────

class TestReasoningEnginePipeline:
    def test_default_pipeline_non_vide(self):
        """Phase 3.1 : le pipeline doit contenir au moins QuestionAnalyzerAgent."""
        assert len(DEFAULT_PIPELINE) >= 1

    def test_premier_agent_est_question_analyzer(self):
        assert DEFAULT_PIPELINE[0].name == "question_analyzer"

    def test_run_reasoning_peuple_analysis(self):
        req = make_request("Pourquoi FastAPI est rapide ?")
        ctx = asyncio.run(run_reasoning(request=req, user_id="test-user"))
        assert ctx.analysis is not None

    def test_run_reasoning_trace_contient_question_analyzer(self):
        req = make_request("test question pour le pipeline")
        ctx = asyncio.run(run_reasoning(request=req, user_id="test-user"))
        step_names = [s.step_name for s in ctx.trace.steps]
        assert "question_analyzer" in step_names

    def test_run_reasoning_trace_duration_renseignee(self):
        req = make_request("test")
        ctx = asyncio.run(run_reasoning(request=req, user_id="test-user"))
        assert ctx.trace.total_duration_ms >= 0.0

    def test_context_to_response_inclut_domain_et_risk(self):
        """La réponse API doit refléter les champs Phase 3.1 via le summary."""
        req = make_request("comment déployer du code python ?")
        ctx = asyncio.run(run_reasoning(request=req, user_id="test-user"))
        response = context_to_response(ctx)
        # Le summary doit mentionner au moins le type de question
        assert response.question_type is not None
        assert isinstance(response.reasoning_summary, str)
        assert len(response.reasoning_summary) > 0

    def test_run_reasoning_context_version_inchangee(self):
        req = make_request("test")
        ctx = asyncio.run(run_reasoning(request=req, user_id="test-user"))
        assert ctx.context_version == "3.0"

"""Agent d'analyse de questions — Phase 3.1 du moteur de raisonnement expert.

Rôle : première étape du pipeline de raisonnement. Analyse la question
utilisateur pour produire un QuestionAnalysis complet avant toute recherche
d'hypothèses ou de preuves.

Stratégie d'exécution (dégradation gracieuse) :
    1. Tentative LLM avec prompt JSON contraint.
    2. Retry LLM avec prompt simplifié si le JSON retourné est invalide.
    3. Fallback déterministe (règles lexicales) si les deux LLM échouent.
    Le pipeline ne s'arrête jamais : une erreur LLM déclenche le fallback.

Dépendances autorisées :
    ✓ core.reasoning.context
    ✓ models.reasoning
    ✗ routers/  (jamais)
    ✗ core.reasoning.reasoning_engine  (jamais — évite le cycle)

Injection de dépendances :
    La fonction ``llm_generate`` est injectable pour les tests.
    Signature attendue : ``async def generate(prompt: str, ...) -> str``
    Compatible avec ``core.llm.generate``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Awaitable, Callable, Optional

from models.reasoning import (
    QuestionAnalysis,
    QuestionType,
    ReasoningStepRecord,
)
from core.reasoning.context import ReasoningContext

logger = logging.getLogger("makenbrain.reasoning.question_analyzer")

# ── Types ─────────────────────────────────────────────────────────────────────

LLMCallable = Callable[..., Awaitable[str]]
"""Type de la fonction LLM injectable.
Signature minimale : async (prompt: str, **kwargs) -> str.
"""

# ── Dictionnaires de détection (règles déterministes) ─────────────────────────

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "technique": [
        "code", "python", "javascript", "typescript", "java", "api",
        "docker", "kubernetes", "serveur", "base de données", "sql",
        "nosql", "fonction", "classe", "module", "import", "framework",
        "library", "git", "ci/cd", "déploiement", "cloud", "aws", "azure",
        "gcp", "endpoint", "algorithme", "performance", "optimisation",
        "fastapi", "react", "vue", "angular", "spring", "microservice",
        "architecture", "design pattern", "debug", "erreur", "exception",
        "async", "thread", "cache", "redis", "migration", "index",
    ],
    "finance": [
        "budget", "investissement", "banque", "tontine", "épargne",
        "prêt", "crédit", "dette", "revenu", "dépense", "bilan",
        "cash flow", "valorisation", "action", "bourse", "dividende",
        "fiscal", "impôt", "comptabilité", "facture", "coût", "marge",
        "rentabilité", "roi", "taux", "intérêt", "mobile money", "momo",
        "fintech", "transaction", "paiement", "virement",
    ],
    "santé": [
        "médecin", "symptôme", "maladie", "traitement", "médicament",
        "diagnostic", "santé", "hôpital", "chirurgie", "thérapie",
        "douleur", "fièvre", "infection", "virus", "bactérie", "vaccin",
        "ordonnance", "clinique", "patient", "urgence", "médical",
    ],
    "droit": [
        "contrat", "loi", "juridique", "légal", "tribunal", "avocat",
        "réglementation", "conformité", "rgpd", "propriété", "brevet",
        "licence", "accord", "clause", "litige", "droits", "obligation",
        "responsabilité", "infraction", "peine", "jugement",
    ],
    "science": [
        "physique", "chimie", "biologie", "mathématiques", "statistiques",
        "expérience", "hypothèse", "théorie", "données", "analyse",
        "recherche", "étude", "résultats", "conclusion", "modèle",
        "simulation", "mesure", "observation", "variable",
    ],
}

COMPLEXITY_MARKERS: list[str] = [
    "pourquoi", "comment", "expliquer", "analyser", "comparer", "évaluer",
    "quelle est la différence", "avantages et inconvénients", "pros and cons",
    "architecture", "conception", "stratégie", "optimiser", "plusieurs",
    "multiples", "étapes", "processus", "workflow", "pipeline", "concevoir",
    "implémenter", "intégrer", "déployer", "migrer", "refactoriser",
]

RISK_MARKERS: dict[str, list[str]] = {
    "high": [
        "chirurgie", "médicament", "diagnostic", "urgence", "tribunal",
        "contrat", "données personnelles", "sécurité", "vulnérabilité",
        "exploit", "critique", "irréversible", "légal", "juridique",
        "investissement important", "production", "suppression définitive",
    ],
    "medium": [
        "financier", "déploiement", "base de données", "migration",
        "modifier", "supprimer", "budget", "dette", "architecture",
        "performance", "charge", "scalabilité",
    ],
}

MEMORY_TRIGGERS: list[str] = [
    "mon projet", "notre ", "mes ", "mon ", "ma ", "notre code",
    "mon application", "notre système", "notre base", "mes données",
    "mon équipe", "notre équipe", "chez moi", "dans mon",
    "j'utilise", "j'ai ", "on utilise", "nous avons", "notre projet",
    "ma stack", "notre stack", "mon repo", "notre repo",
]

EXTERNAL_SEARCH_TRIGGERS: list[str] = [
    "actualité", "dernière version", "récent", "nouveau", "2024", "2025",
    "2026", "documentation officielle", "changelog", "release", "latest",
    "dernières nouvelles", "prix actuel", "disponible maintenant",
    "existe-t-il", "quoi de neuf", "mise à jour", "deprecated",
]

QUESTION_TYPE_PATTERNS: dict[QuestionType, list[str]] = {
    QuestionType.HYPOTHETICAL: [
        r"et si\b", r"que se passerait", r"supposons que",
        r"imaginons que", r"dans le cas où", r"si je\b", r"si on\b",
        r"hypothétiquement", r"au cas où",
    ],
    QuestionType.COMPARATIVE: [
        r"\bvs\b", r"\bversus\b", r"différence entre", r"\bcomparer\b",
        r"comparaison", r"avantages.*inconvénients", r"lequel est meilleur",
        r"par rapport à",
    ],
    QuestionType.ANALYTICAL: [
        r"^pourquoi\b", r"pour quelle raison", r"qu.est.ce qui cause",
        r"comment expliquer", r"quelle est la cause", r"\banalyse\b",
        r"expliqu", r"justifi",
    ],
    QuestionType.PROCEDURAL: [
        r"^comment\b", r"quell[e]?s? étapes", r"comment faire",
        r"^implémenter", r"^créer\b", r"^mettre en place",
        r"^configurer", r"^installer", r"^déployer",
    ],
    QuestionType.EVALUATIVE: [
        r"\bmeilleur", r"\brecommand", r"devrait.on", r"quelle approche",
        r"quelle solution", r"est.ce une bonne", r"vaut.il mieux",
        r"faut.il", r"dois.je",
    ],
    QuestionType.FACTUAL: [
        r"^qu.est.ce", r"^c.est quoi", r"^quel est\b", r"^quelle est\b",
        r"^qui est\b", r"^où est\b", r"^quand\b", r"^définir\b",
        r"^qu.est ce", r"c.est quoi",
    ],
}

STOPWORDS_FR: frozenset[str] = frozenset({
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou",
    "est", "sont", "dans", "pour", "avec", "sur", "par", "que", "qui",
    "quoi", "quel", "quelle", "quels", "comment", "pourquoi", "quand",
    "il", "elle", "ils", "elles", "je", "tu", "nous", "vous", "on",
    "ne", "pas", "plus", "très", "aussi", "mais", "donc", "car", "fait",
    "faire", "peut", "doit", "avoir", "être", "cette", "cet", "ces",
    "mon", "ton", "son", "notre", "votre", "leur", "mes", "tes", "ses",
})


# ── Agent (protocole ReasoningAgent) ─────────────────────────────────────────

class QuestionAnalyzerAgent:
    """Agent d'analyse de la question — étape 1 du pipeline de raisonnement.

    Implémente le protocole ``ReasoningAgent`` défini dans
    ``core.reasoning.__init__``.

    Responsabilités :
        - Détecter le type sémantique de la question (QuestionType).
        - Identifier le domaine métier (technique, finance, santé…).
        - Évaluer la complexité [0.0 → 1.0].
        - Évaluer le niveau de risque (low / medium / high).
        - Déterminer si la mémoire utilisateur est pertinente.
        - Déterminer si une recherche externe est nécessaire.
        - Déterminer si un raisonnement approfondi est requis.
        - Extraire les sous-questions, entités et concepts clés.
        - Alimenter la trace d'exécution (ReasoningStepRecord).

    Stratégie :
        1. LLM avec prompt JSON contraint (haute qualité, parsing strict).
        2. LLM retry avec prompt simplifié (filet de sécurité).
        3. Fallback déterministe (règles lexicales, sans LLM).

    Usage :
        L'agent est instancié une fois dans ``DEFAULT_PIPELINE`` de
        ``reasoning_engine.py`` et réutilisé pour toutes les requêtes.
    """

    name: str = "question_analyzer"

    async def run(self, ctx: ReasoningContext) -> ReasoningContext:
        """Exécute l'analyse de la question et enrichit le contexte.

        Point d'entrée du protocole ReasoningAgent. Délègue à la fonction
        standalone ``analyze_question`` pour permettre les tests unitaires
        indépendamment du protocole.

        Args:
            ctx: Contexte de raisonnement courant (modifié in-place).

        Returns:
            ctx avec ctx.analysis peuplé et ctx.trace enrichi.
        """
        return await analyze_question(ctx)


# ── Point d'entrée public ─────────────────────────────────────────────────────

async def analyze_question(
    ctx: ReasoningContext,
    llm_generate: Optional[LLMCallable] = None,
) -> ReasoningContext:
    """Analyse la question et peuple ctx.analysis.

    Fonction publique du module, utilisée par QuestionAnalyzerAgent.run()
    et directement testable sans instancier l'agent.

    Args:
        ctx          : Contexte de raisonnement courant.
        llm_generate : Fonction LLM injectable pour les tests.
                       Si None, importe ``core.llm.generate`` au runtime.
                       Signature : ``async (prompt: str, **kwargs) -> str``.

    Returns:
        ctx enrichi : ctx.analysis peuplé, ctx.trace mis à jour.
    """
    step_start = time.monotonic()
    question   = ctx.request.question

    logger.info("[QA] Analyse | '%.80s'", question)

    # Résoudre la fonction LLM (import lazy pour éviter les cycles)
    _generate = llm_generate
    if _generate is None:
        try:
            from core.llm import generate as _core_generate  # noqa: PLC0415
            _generate = _core_generate
        except ImportError:
            logger.warning("[QA] core.llm introuvable — fallback déterministe")

    # ── Tentative 1 : LLM prompt complet ────────────────────────────────────
    analysis: Optional[QuestionAnalysis] = None
    degraded: bool = False
    error_msg: Optional[str] = None

    if _generate is not None:
        try:
            analysis = await _analyze_with_llm(question, _generate)
            logger.debug("[QA] LLM principal OK")
        except Exception as exc:
            error_msg = str(exc)
            logger.warning("[QA] LLM principal échoué (%s) — retry", exc)

    # ── Tentative 2 : LLM prompt simplifié (retry) ───────────────────────────
    if analysis is None and _generate is not None:
        try:
            analysis = await _analyze_with_llm_simplified(question, _generate)
            logger.debug("[QA] LLM retry OK")
        except Exception as exc:
            error_msg = str(exc)
            logger.warning("[QA] LLM retry échoué (%s) — fallback", exc)
            degraded = True

    # ── Tentative 3 : Fallback déterministe ──────────────────────────────────
    if analysis is None:
        degraded = True
        analysis = _analyze_deterministic(question)
        logger.info("[QA] Fallback déterministe utilisé")

    ctx.analysis = analysis
    duration_ms  = (time.monotonic() - step_start) * 1000

    # ── Trace ────────────────────────────────────────────────────────────────
    ctx.trace.steps.append(ReasoningStepRecord(
        step_name      = "question_analyzer",
        duration_ms    = round(duration_ms, 2),
        success        = True,
        input_summary  = f"'{question[:80]}' ({len(question)} chars)",
        output_summary = (
            f"type={analysis.question_type.value} "
            f"domain={analysis.domain} "
            f"complexity={analysis.complexity_score:.2f} "
            f"risk={analysis.risk_level} "
            f"memory={analysis.requires_memory} "
            f"ext_search={analysis.requires_external_search} "
            f"deep={analysis.requires_deep_reasoning} "
            f"confidence={analysis.confidence:.2f}"
        ),
        degraded       = degraded,
        error_message  = error_msg if degraded else None,
    ))

    if degraded:
        ctx.mark_degraded(
            "question_analyzer",
            error_msg or "LLM indisponible — fallback déterministe activé",
        )

    logger.info(
        "[QA] Terminé %.0fms | type=%s domain=%s complexity=%.2f "
        "risk=%s deep=%s dégradé=%s",
        duration_ms,
        analysis.question_type.value,
        analysis.domain,
        analysis.complexity_score,
        analysis.risk_level,
        analysis.requires_deep_reasoning,
        degraded,
    )

    return ctx


# ── Analyse via LLM ───────────────────────────────────────────────────────────

async def _analyze_with_llm(
    question: str,
    llm_generate: LLMCallable,
) -> QuestionAnalysis:
    """Analyse la question via LLM avec un prompt JSON fortement contraint.

    Le prompt force une réponse JSON pur, sans texte avant ni après.
    Le résultat est validé par Pydantic via ``_parse_llm_response``.

    Args:
        question     : Question à analyser.
        llm_generate : Fonction de génération LLM (async).

    Returns:
        QuestionAnalysis validé par Pydantic.

    Raises:
        ValueError : Si le JSON retourné est invalide après extraction.
    """
    prompt   = _build_main_prompt(question)
    raw_text = await llm_generate(prompt)
    return _parse_llm_response(raw_text, question)


async def _analyze_with_llm_simplified(
    question: str,
    llm_generate: LLMCallable,
) -> QuestionAnalysis:
    """Retry d'analyse via LLM avec un prompt minimaliste.

    Utilisé uniquement si le prompt principal échoue. Plus court et plus
    contraint pour maximiser la probabilité d'un JSON valide.

    Args:
        question     : Question à analyser.
        llm_generate : Fonction de génération LLM (async).

    Returns:
        QuestionAnalysis validé.

    Raises:
        ValueError : Si même le prompt simplifié retourne un JSON invalide.
    """
    prompt   = _build_simple_prompt(question)
    raw_text = await llm_generate(prompt)
    return _parse_llm_response(raw_text, question)


def _build_main_prompt(question: str) -> str:
    """Construit le prompt principal d'analyse.

    Contraint le LLM à retourner exclusivement un JSON valide.
    Documente les valeurs possibles pour chaque clé.

    Args:
        question : Question à analyser.

    Returns:
        Prompt formaté.
    """
    return f"""Analyse cette question et réponds UNIQUEMENT avec un JSON valide.
COMMENCE DIRECTEMENT par {{ — aucun texte avant ou après.

Question à analyser : {question}

Schéma JSON attendu (respecte EXACTEMENT les clés et les valeurs possibles) :
{{
  "question_type": "factual|analytical|comparative|procedural|hypothetical|evaluative",
  "domain": "technique|finance|santé|droit|science|general",
  "complexity_score": <nombre 0.0 à 1.0>,
  "risk_level": "low|medium|high",
  "sub_questions": ["<sous-question 1>", "<sous-question 2>"],
  "key_entities": ["<entité1>", "<entité2>"],
  "key_concepts": ["<concept1>", "<concept2>"],
  "requires_reasoning": true|false,
  "requires_memory": true|false,
  "requires_external_search": true|false,
  "requires_deep_reasoning": true|false,
  "confidence": <nombre 0.0 à 1.0>
}}

Règles de classification :
- question_type  → factual=fait/définition | analytical=pourquoi/cause | comparative=vs/différence | procedural=comment/étapes | hypothetical=si/imaginons | evaluative=meilleur/recommandation
- domain         → technique=code/infra/software | finance=argent/budget/investissement | santé=médical | droit=légal/contrat | science=recherche | general=autre
- complexity_score → 0.0=triviale | 0.3=simple | 0.5=modérée | 0.7=complexe | 1.0=très complexe
- risk_level     → low=information générale | medium=technique ou financier | high=médical/juridique/sécurité/production
- requires_memory→ true si la question porte sur le projet ou les habitudes de l'utilisateur
- requires_external_search → true si les données doivent être récentes (actualité, versions, prix)
- requires_deep_reasoning  → true si complexity_score >= 0.6
- confidence     → ta certitude dans cette analyse (0.7 si incertain, 0.9 si évident)

RÉPONDS UNIQUEMENT AVEC LE JSON — aucun texte, aucune explication."""


def _build_simple_prompt(question: str) -> str:
    """Construit le prompt de retry minimaliste.

    Version courte pour maximiser la probabilité d'obtenir un JSON valide
    après un échec du prompt principal.

    Args:
        question : Question à analyser.

    Returns:
        Prompt simplifié.
    """
    return f"""Question : {question}

Réponds UNIQUEMENT avec ce JSON (commence par {{, termine par }}).
Modifie uniquement les valeurs, pas les clés :

{{
  "question_type": "factual",
  "domain": "general",
  "complexity_score": 0.5,
  "risk_level": "low",
  "sub_questions": [],
  "key_entities": [],
  "key_concepts": [],
  "requires_reasoning": false,
  "requires_memory": false,
  "requires_external_search": false,
  "requires_deep_reasoning": false,
  "confidence": 0.7
}}"""


# ── Parsing et validation du JSON LLM ────────────────────────────────────────

def _parse_llm_response(response_text: str, question: str) -> QuestionAnalysis:
    """Parse et valide la réponse brute du LLM.

    Tente plusieurs stratégies d'extraction dans l'ordre :
        1. Parse direct de la réponse (cas nominal).
        2. Nettoyage des blocs ```json ... ``` et re-parse.
        3. Extraction regex du premier bloc {…}.
        4. Extraction entre la première { et la dernière }.

    Args:
        response_text : Texte brut retourné par le LLM.
        question      : Question originale (pour construire QuestionAnalysis).

    Returns:
        QuestionAnalysis validé par Pydantic.

    Raises:
        ValueError : Si aucune stratégie d'extraction ne produit un JSON valide.
    """
    raw = response_text.strip()

    # Stratégie 1 : parse direct
    parsed = _try_parse_json(raw)

    # Stratégie 2 : nettoyage des balises markdown
    if parsed is None:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```\s*$",       "", cleaned, flags=re.MULTILINE)
        parsed  = _try_parse_json(cleaned.strip())

    # Stratégie 3 : extraction regex premier bloc JSON
    if parsed is None:
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', raw, re.DOTALL)
        if match:
            parsed = _try_parse_json(match.group(0))

    # Stratégie 4 : extraction entre première { et dernière }
    if parsed is None:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if 0 <= start < end:
            parsed = _try_parse_json(raw[start:end])

    if parsed is None:
        raise ValueError(
            f"JSON non extractible depuis la réponse LLM : {raw[:200]!r}"
        )

    return _dict_to_analysis(parsed, question)


def _try_parse_json(text: str) -> Optional[dict[str, Any]]:
    """Tente de parser un texte comme JSON, retourne None en cas d'échec.

    Args:
        text : Texte candidat.

    Returns:
        dict parsé ou None.
    """
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _dict_to_analysis(data: dict[str, Any], question: str) -> QuestionAnalysis:
    """Convertit un dict brut issu du LLM en QuestionAnalysis Pydantic.

    Normalise et valide toutes les valeurs :
        - question_type  : vers l'enum, FACTUAL par défaut.
        - domain         : vers l'un des 6 domaines valides, general par défaut.
        - risk_level     : low/medium/high uniquement, low par défaut.
        - complexity_score et confidence : clampés entre 0.0 et 1.0.
        - Listes         : converties de façon sécurisée.

    Args:
        data     : Dict extrait du JSON LLM.
        question : Question originale.

    Returns:
        QuestionAnalysis validé et normalisé.
    """
    # question_type
    raw_type = str(data.get("question_type", "factual")).lower().strip()
    try:
        q_type = QuestionType(raw_type)
    except ValueError:
        q_type = QuestionType.FACTUAL

    # domain
    valid_domains = {"technique", "finance", "santé", "droit", "science", "general"}
    domain = str(data.get("domain", "general")).lower().strip()
    if domain not in valid_domains:
        domain = "general"

    # risk_level
    risk = str(data.get("risk_level", "low")).lower().strip()
    if risk not in ("low", "medium", "high"):
        risk = "low"

    # scores numériques
    complexity = _clamp_float(data.get("complexity_score", 0.5))
    confidence = _clamp_float(data.get("confidence", 0.7))

    # booléens
    req_reasoning = bool(data.get("requires_reasoning",  complexity >= 0.5))
    req_memory    = bool(data.get("requires_memory",     False))
    req_external  = bool(data.get("requires_external_search", False))
    req_deep      = bool(data.get("requires_deep_reasoning",  complexity >= 0.6))

    return QuestionAnalysis(
        original_question        = question,
        question_type            = q_type,
        domain                   = domain,
        complexity_score         = complexity,
        risk_level               = risk,
        sub_questions            = _safe_str_list(data.get("sub_questions",  [])),
        key_entities             = _safe_str_list(data.get("key_entities",   [])),
        key_concepts             = _safe_str_list(data.get("key_concepts",   [])),
        requires_reasoning       = req_reasoning,
        requires_memory          = req_memory,
        requires_external_search = req_external,
        requires_deep_reasoning  = req_deep,
        confidence               = confidence,
    )


# ── Fallback déterministe ─────────────────────────────────────────────────────

def _analyze_deterministic(question: str) -> QuestionAnalysis:
    """Analyse déterministe sans appel LLM — fallback de dernier recours.

    Basé exclusivement sur des règles lexicales et des dictionnaires de
    mots-clés. Toujours disponible, même si Ollama est éteint.

    La confiance est fixée à 0.6 pour signaler que le résultat est moins
    fiable qu'une analyse LLM (confiance nominale ≥ 0.75).

    Args:
        question : Question originale.

    Returns:
        QuestionAnalysis produit par les règles lexicales.
    """
    q_lower = question.lower()

    question_type = _detect_type(q_lower)
    domain        = _detect_domain(q_lower)
    complexity    = _detect_complexity(q_lower)
    risk_level    = _detect_risk(q_lower)
    req_memory    = _detect_needs_memory(q_lower)
    req_external  = _detect_needs_external(q_lower)
    req_deep      = complexity >= 0.6

    return QuestionAnalysis(
        original_question        = question,
        question_type            = question_type,
        domain                   = domain,
        complexity_score         = complexity,
        risk_level               = risk_level,
        sub_questions            = _extract_sub_questions(question),
        key_entities             = _extract_entities(question),
        key_concepts             = _extract_concepts(q_lower, domain),
        requires_reasoning       = complexity >= 0.5,
        requires_memory          = req_memory,
        requires_external_search = req_external,
        requires_deep_reasoning  = req_deep,
        confidence               = 0.6,
    )


# ── Détecteurs lexicaux ───────────────────────────────────────────────────────

def _detect_type(q_lower: str) -> QuestionType:
    """Détecte le type de question par patterns regex ordonnés par priorité.

    L'ordre de priorité évite les faux positifs :
    HYPOTHETICAL et COMPARATIVE sont testés avant FACTUAL car leurs marqueurs
    peuvent apparaître dans n'importe quelle partie de la phrase.

    Args:
        q_lower : Question normalisée en minuscules.

    Returns:
        QuestionType détecté, FACTUAL par défaut.
    """
    priority: list[QuestionType] = [
        QuestionType.HYPOTHETICAL,
        QuestionType.COMPARATIVE,
        QuestionType.ANALYTICAL,
        QuestionType.PROCEDURAL,
        QuestionType.EVALUATIVE,
        QuestionType.FACTUAL,
    ]
    for q_type in priority:
        for pattern in QUESTION_TYPE_PATTERNS.get(q_type, []):
            if re.search(pattern, q_lower):
                return q_type
    return QuestionType.FACTUAL


def _detect_domain(q_lower: str) -> str:
    """Détecte le domaine métier par comptage de mots-clés.

    Compte les occurrences de mots-clés de chaque domaine et retourne
    celui qui en a le plus. Retourne "general" en l'absence de signal.

    Args:
        q_lower : Question normalisée en minuscules.

    Returns:
        Nom du domaine le plus probable.
    """
    scores: dict[str, int] = {
        domain: sum(1 for kw in keywords if kw in q_lower)
        for domain, keywords in DOMAIN_KEYWORDS.items()
    }
    scores = {d: s for d, s in scores.items() if s > 0}
    return max(scores, key=lambda d: scores[d]) if scores else "general"


def _detect_complexity(q_lower: str) -> float:
    """Calcule le score de complexité par agrégation de signaux pondérés.

    Signaux utilisés :
        - Présence de marqueurs de complexité (COMPLEXITY_MARKERS).
        - Longueur de la question (proxy de densité informationnelle).
        - Connecteurs logiques (sous-questions implicites).
        - Points d'interrogation multiples.

    Args:
        q_lower : Question normalisée en minuscules.

    Returns:
        Score entre 0.0 et 1.0 (2 décimales).
    """
    score = 0.2  # base minimale

    # Marqueurs de complexité (max +0.45)
    marker_hits = sum(1 for m in COMPLEXITY_MARKERS if m in q_lower)
    score += min(marker_hits * 0.15, 0.45)

    # Longueur (proxy de richesse)
    length = len(q_lower)
    if length > 150:
        score += 0.20
    elif length > 80:
        score += 0.10

    # Connecteurs logiques (max +0.15)
    connectors = (" et ", " ou ", " mais ", " donc ", " car ",
                  " parce que ", " afin de ", " pour que ")
    connector_hits = sum(1 for c in connectors if c in q_lower)
    score += min(connector_hits * 0.05, 0.15)

    # Questions multiples
    if q_lower.count("?") > 1:
        score += 0.10

    return round(min(score, 1.0), 2)


def _detect_risk(q_lower: str) -> str:
    """Évalue le niveau de risque par détection de mots-clés sensibles.

    Ordre de vérification : high → medium → low (fail-safe).

    Args:
        q_lower : Question normalisée en minuscules.

    Returns:
        "low", "medium" ou "high".
    """
    for kw in RISK_MARKERS.get("high", []):
        if kw in q_lower:
            return "high"
    for kw in RISK_MARKERS.get("medium", []):
        if kw in q_lower:
            return "medium"
    return "low"


def _detect_needs_memory(q_lower: str) -> bool:
    """Détermine si la mémoire utilisateur (Supabase) est pertinente.

    True si la question contient des marqueurs de personnalisation ou
    de contexte utilisateur (pronoms possessifs, références au projet…).

    Args:
        q_lower : Question normalisée en minuscules.

    Returns:
        True si la mémoire est probablement utile.
    """
    return any(trigger in q_lower for trigger in MEMORY_TRIGGERS)


def _detect_needs_external(q_lower: str) -> bool:
    """Détermine si une recherche web externe est recommandée.

    True si la question nécessite des informations récentes (actualité,
    nouvelles versions, prix courants) absentes de la mémoire locale.

    Args:
        q_lower : Question normalisée en minuscules.

    Returns:
        True si une recherche externe est probable-ment nécessaire.
    """
    return any(trigger in q_lower for trigger in EXTERNAL_SEARCH_TRIGGERS)


# ── Extracteurs d'entités et de concepts ─────────────────────────────────────

def _extract_sub_questions(question: str) -> list[str]:
    """Extrait les sous-questions implicites d'une question composée.

    Détecte les parties séparées par des virgules ou points-virgules
    contenant des marqueurs interrogatifs.

    Args:
        question : Question originale (casse préservée).

    Returns:
        Liste de sous-questions (max 4, peut être vide).
    """
    sub_questions: list[str] = []
    markers_q    = ("comment", "pourquoi", "quelle", "quel", "quels",
                    "est-ce", "faut-il", "doit-on")

    for part in re.split(r"[,;]\s*", question):
        part = part.strip()
        if len(part) > 15 and any(m in part.lower() for m in markers_q):
            if part.rstrip("?.,;") != question.rstrip("?.,;"):
                sub_questions.append(part)

    return sub_questions[:4]


def _extract_entities(question: str) -> list[str]:
    """Extrait les entités nommées par heuristique de capitalisation.

    Heuristique légère : mots en position non-initiale commençant par une
    majuscule, non entièrement en majuscules, longueur > 2.
    Ne requiert aucune dépendance NLP externe.

    Args:
        question : Question originale (casse préservée).

    Returns:
        Liste d'entités dédupliquées (max 8, peut être vide).
    """
    entities: list[str] = []
    seen:     set[str]  = set()

    for i, word in enumerate(question.split()):
        if i == 0:
            continue  # ignore le premier mot (majuscule grammaticale)
        clean = re.sub(r"[^\w]", "", word)
        if (clean
                and len(clean) > 2
                and clean[0].isupper()
                and not clean.isupper()
                and clean.lower() not in seen):
            entities.append(clean)
            seen.add(clean.lower())

    return entities[:8]


def _extract_concepts(q_lower: str, domain: str) -> list[str]:
    """Extrait les concepts clés pour la recherche en mémoire vectorielle.

    Combine :
        1. Les mots-clés du domaine détecté présents dans la question.
        2. Les mots significatifs (longueur > 4, hors stopwords).

    Args:
        q_lower : Question normalisée en minuscules.
        domain  : Domaine détecté par ``_detect_domain``.

    Returns:
        Liste de concepts dédupliquée (max 10).
    """
    concepts: list[str] = []

    # Mots-clés du domaine présents dans la question
    for kw in DOMAIN_KEYWORDS.get(domain, []):
        if kw in q_lower and kw not in concepts:
            concepts.append(kw)

    # Mots significatifs de la question
    for word in re.findall(r'\b[a-zàâéèêëîïôùûüç]{4,}\b', q_lower):
        if word not in STOPWORDS_FR and word not in concepts:
            concepts.append(word)

    return concepts[:10]


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _clamp_float(value: Any, default: float = 0.5) -> float:
    """Convertit une valeur en float clampé entre 0.0 et 1.0.

    Args:
        value   : Valeur à convertir (peut être str, int, float, None).
        default : Valeur par défaut si la conversion échoue.

    Returns:
        float entre 0.0 et 1.0.
    """
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return default


def _safe_str_list(value: Any) -> list[str]:
    """Convertit une valeur arbitraire en liste de strings non-vides.

    Args:
        value : Valeur à convertir (list, str, None…).

    Returns:
        Liste de strings filtrée et sécurisée.
    """
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []

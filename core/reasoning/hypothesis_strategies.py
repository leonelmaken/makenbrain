"""Stratégies de génération d'hypothèses — Phase 3.2.

Ce module implémente le patron de conception Stratégie (Strategy Pattern) pour
la génération d'hypothèses. Chaque stratégie est indépendante, testable
isolément et extensible sans modifier les autres.

Architecture :
    HypothesisStrategy (Protocol)  ← contrat que chaque stratégie doit respecter
    HeuristicStrategy              ← déterministe, basée sur QuestionType + domaine
    DecompositionStrategy          ← une hypothèse par sous-question détectée
    ConceptualStrategy             ← hypothèses ancrées sur les concepts clés
    LLMStrategy                    ← hypothèses créatives via LLM (optionnel)

Règle d'or :
    Toute erreur dans une stratégie retourne une liste vide, jamais une exception.
    Le pipeline de raisonnement ne peut pas s'arrêter à cause d'une stratégie.

Extensibilité :
    Pour ajouter une stratégie : implémenter HypothesisStrategy et l'enregistrer
    dans DEFAULT_STRATEGIES de hypothesis_engine.py. Aucun autre fichier à toucher.

Dépendances autorisées :
    ✓ models.reasoning
    ✓ core.reasoning.context
    ✗ routers/  (jamais)
    ✗ core.reasoning.reasoning_engine  (jamais — évite le cycle)
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable, Optional, runtime_checkable

from typing import Protocol

from models.reasoning import Hypothesis, QuestionAnalysis, QuestionType
from core.reasoning.context import ReasoningContext

logger = logging.getLogger("makenbrain.reasoning.hypothesis_strategies")

# ── Type de la fonction LLM injectable ───────────────────────────────────────

LLMCallable = Callable[..., Awaitable[str]]
"""Signature minimale attendue : async (prompt: str, **kwargs) -> str."""


# ── Protocole commun ──────────────────────────────────────────────────────────

@runtime_checkable
class HypothesisStrategy(Protocol):
    """Contrat que toute stratégie de génération d'hypothèses doit respecter.

    Le protocole est ``runtime_checkable`` pour permettre les assertions
    ``isinstance(strategy, HypothesisStrategy)`` dans les tests.

    Règles d'implémentation :
        - ``generate()`` ne doit jamais lever d'exception.
        - En cas d'erreur interne, retourner ``[]`` et logger un warning.
        - Les hypothèses retournées ont ``strategy_name`` renseigné.
    """

    @property
    def name(self) -> str:
        """Identifiant unique de la stratégie (snake_case)."""
        ...  # pragma: no cover

    @property
    def weight(self) -> float:
        """Poids de confiance de la stratégie dans le score final [0.0 → 1.0].

        Utilisé par HypothesisRanker pour moduler le score initial.
        """
        ...  # pragma: no cover

    async def generate(self, ctx: ReasoningContext) -> list[Hypothesis]:
        """Génère une liste d'hypothèses à partir du contexte courant.

        Args:
            ctx : Contexte de raisonnement. Contient au minimum
                  ``ctx.analysis`` (peuplé par QuestionAnalyzerAgent).

        Returns:
            Liste d'hypothèses (peut être vide, jamais None).
        """
        ...  # pragma: no cover


# ── Templates heuristiques ────────────────────────────────────────────────────

_HEURISTIC_TEMPLATES: dict[QuestionType, list[tuple[str, str]]] = {
    # (content_template, justification_template)
    QuestionType.FACTUAL: [
        (
            "La réponse directe à cette question concerne {concept} "
            "dans le contexte du domaine {domain}.",
            "Question factuelle centrée sur '{concept}' — "
            "une définition ou un fait précis suffit à répondre.",
        ),
        (
            "'{concept}' peut être défini par ses propriétés fondamentales "
            "et son rôle dans le domaine {domain}.",
            "La question appelle une réponse descriptive sur '{concept}'.",
        ),
    ],
    QuestionType.ANALYTICAL: [
        (
            "La cause principale est liée à '{concept}' "
            "dans le domaine {domain}.",
            "Question analytique : identifier la cause racine liée à '{concept}'.",
        ),
        (
            "Plusieurs facteurs combinés expliquent ce phénomène : {concepts}.",
            "L'analyse révèle une causalité multi-facteurs : {concepts}.",
        ),
        (
            "Il s'agit d'un problème systémique propre au domaine {domain}.",
            "La complexité de la question suggère une dimension systémique.",
        ),
    ],
    QuestionType.COMPARATIVE: [
        (
            "{entity_a} est plus adapté lorsque la priorité est "
            "la performance et la scalabilité.",
            "{entity_a} excelle dans les scénarios à haute contrainte.",
        ),
        (
            "{entity_b} est plus adapté lorsque la priorité est "
            "la simplicité et la maintenabilité.",
            "{entity_b} favorise la productivité des équipes.",
        ),
        (
            "{entity_a} et {entity_b} sont complémentaires selon "
            "le cas d'usage et les contraintes du projet.",
            "La comparaison n'a pas de vainqueur absolu : le contexte décide.",
        ),
    ],
    QuestionType.PROCEDURAL: [
        (
            "La procédure standard dans le domaine {domain} implique "
            "une approche séquentielle étape par étape.",
            "Question procédurale : une démarche structurée est attendue.",
        ),
        (
            "Il existe deux approches : une méthode simplifiée pour les "
            "débutants et une méthode avancée pour les experts.",
            "Le niveau d'expertise guide le choix de la procédure.",
        ),
        (
            "La démarche optimale dépend des contraintes "
            "de l'environnement {domain}.",
            "Les contraintes contextuelles déterminent la meilleure procédure.",
        ),
    ],
    QuestionType.HYPOTHETICAL: [
        (
            "Dans ce scénario, l'impact le plus probable serait positif "
            "à court terme mais nécessiterait des ajustements.",
            "Scénario hypothétique avec des effets prévisibles à court terme.",
        ),
        (
            "Ce changement entraînerait des effets en cascade "
            "sur le système existant dans le domaine {domain}.",
            "Tout changement structurel produit des effets secondaires.",
        ),
        (
            "Le risque principal de ce scénario est une instabilité "
            "temporaire pendant la phase de transition.",
            "La transition est toujours une phase à risque à gérer.",
        ),
    ],
    QuestionType.EVALUATIVE: [
        (
            "La solution optimale dépend des contraintes spécifiques "
            "du projet et du contexte {domain}.",
            "Aucune solution n'est universelle : le contexte prime.",
        ),
        (
            "Il n'existe pas de réponse unique : chaque approche "
            "implique des compromis (trade-offs) différents.",
            "L'évaluation raisonnée expose les trade-offs de chaque option.",
        ),
        (
            "La recommandation basée sur les bonnes pratiques "
            "du domaine {domain} favorise la maintenabilité long terme.",
            "Les bonnes pratiques du domaine guident la recommandation.",
        ),
    ],
}


# ── Stratégie 1 : Heuristique ─────────────────────────────────────────────────

class HeuristicStrategy:
    """Génère des hypothèses déterministes basées sur templates par QuestionType.

    Stratégie de base, toujours disponible, sans dépendance externe.
    Utilise les templates de ``_HEURISTIC_TEMPLATES`` adaptés au type de
    question et au domaine identifiés par QuestionAnalyzerAgent.

    Poids : 0.50 (confiance modérée — générique par nature).
    """

    @property
    def name(self) -> str:
        return "heuristic"

    @property
    def weight(self) -> float:
        return 0.50

    async def generate(self, ctx: ReasoningContext) -> list[Hypothesis]:
        """Génère N hypothèses depuis les templates du QuestionType détecté.

        Args:
            ctx : Contexte de raisonnement avec ``ctx.analysis`` peuplé.

        Returns:
            Liste de 2-3 hypothèses selon le type de question.
        """
        try:
            analysis     = ctx.analysis_or_fallback
            templates    = _HEURISTIC_TEMPLATES.get(
                analysis.question_type,
                _HEURISTIC_TEMPLATES[QuestionType.FACTUAL],
            )
            substitutions = _build_substitutions(analysis)
            hypotheses: list[Hypothesis] = []

            for content_tpl, just_tpl in templates:
                content       = _safe_format(content_tpl, substitutions)
                justification = _safe_format(just_tpl, substitutions)
                hypotheses.append(Hypothesis(
                    content       = content,
                    initial_score = self.weight,
                    origin        = "heuristic",
                    justification = justification,
                    strategy_name = self.name,
                ))

            return hypotheses

        except Exception as exc:
            logger.warning("[HeuristicStrategy] Erreur inattendue : %s", exc)
            return []


# ── Stratégie 2 : Décomposition ───────────────────────────────────────────────

class DecompositionStrategy:
    """Génère une hypothèse par sous-question identifiée par QuestionAnalyzer.

    Si ``ctx.analysis.sub_questions`` est vide, retourne [].
    Chaque sous-question devient une hypothèse indépendante, ce qui
    permet au pipeline d'évaluer des facettes distinctes de la question.

    Poids : 0.60 (confiance élevée — ancré sur la question de l'utilisateur).
    """

    @property
    def name(self) -> str:
        return "decomposition"

    @property
    def weight(self) -> float:
        return 0.60

    async def generate(self, ctx: ReasoningContext) -> list[Hypothesis]:
        """Génère une hypothèse par sous-question détectée.

        Args:
            ctx : Contexte de raisonnement.

        Returns:
            Liste d'hypothèses (vide si aucune sous-question détectée).
        """
        try:
            analysis = ctx.analysis_or_fallback
            if not analysis.sub_questions:
                return []

            hypotheses: list[Hypothesis] = []
            for sub_q in analysis.sub_questions:
                sub_q = sub_q.strip().rstrip("?.,;")
                hypotheses.append(Hypothesis(
                    content       = (
                        f"Pour répondre à la sous-question "
                        f"'{sub_q}' : une réponse ciblée et documentée est possible."
                    ),
                    initial_score = self.weight,
                    origin        = "decomposition",
                    justification = (
                        f"La question principale contient une dimension "
                        f"distincte : '{sub_q}' — traiter chaque aspect séparément."
                    ),
                    strategy_name = self.name,
                ))
            return hypotheses

        except Exception as exc:
            logger.warning("[DecompositionStrategy] Erreur inattendue : %s", exc)
            return []


# ── Stratégie 3 : Conceptuelle ────────────────────────────────────────────────

class ConceptualStrategy:
    """Génère des hypothèses ancrées sur les concepts clés de la question.

    Exploite ``ctx.analysis.key_concepts`` produit par QuestionAnalyzerAgent.
    Génère :
        - Une hypothèse principale centrée sur le concept dominant.
        - Une hypothèse de relation si au moins 2 concepts sont détectés.

    Poids : 0.55 (confiance bonne — ancré sur des termes réels).
    """

    @property
    def name(self) -> str:
        return "conceptual"

    @property
    def weight(self) -> float:
        return 0.55

    async def generate(self, ctx: ReasoningContext) -> list[Hypothesis]:
        """Génère des hypothèses depuis les concepts clés.

        Args:
            ctx : Contexte de raisonnement.

        Returns:
            0, 1 ou 2 hypothèses selon le nombre de concepts disponibles.
        """
        try:
            analysis = ctx.analysis_or_fallback
            if not analysis.key_concepts:
                return []

            hypotheses: list[Hypothesis] = []
            main_concept = analysis.key_concepts[0]

            # Hypothèse principale sur le concept dominant
            hypotheses.append(Hypothesis(
                content       = (
                    f"La réponse s'articule principalement autour "
                    f"du concept '{main_concept}' "
                    f"dans le domaine {analysis.domain}."
                ),
                initial_score = self.weight,
                origin        = "conceptual",
                justification = (
                    f"'{main_concept}' est le terme central identifié "
                    f"dans la question — il structure la réponse."
                ),
                strategy_name = self.name,
            ))

            # Hypothèse de relation si ≥ 2 concepts
            if len(analysis.key_concepts) >= 2:
                concept_b = analysis.key_concepts[1]
                hypotheses.append(Hypothesis(
                    content       = (
                        f"La relation entre '{main_concept}' et '{concept_b}' "
                        f"est au cœur de la réponse à cette question."
                    ),
                    initial_score = self.weight * 0.90,
                    origin        = "conceptual",
                    justification = (
                        f"La question implique une interaction entre "
                        f"'{main_concept}' et '{concept_b}'."
                    ),
                    strategy_name = self.name,
                ))

            return hypotheses

        except Exception as exc:
            logger.warning("[ConceptualStrategy] Erreur inattendue : %s", exc)
            return []


# ── Stratégie 4 : LLM ────────────────────────────────────────────────────────

class LLMStrategy:
    """Génère des hypothèses créatives via le LLM.

    Stratégie optionnelle : si le LLM est indisponible ou retourne un JSON
    invalide, retourne [] sans faire planter le pipeline.

    La fonction LLM est injectable pour les tests (``llm_generate`` parameter).
    Si non fournie, tente un import lazy de ``core.llm.generate`` au runtime.

    Poids : 0.80 (haute confiance — le LLM produit des hypothèses contextuelles).
    """

    def __init__(self, llm_generate: Optional[LLMCallable] = None) -> None:
        """Initialise la stratégie LLM.

        Args:
            llm_generate : Fonction LLM injectable.
                           Si None, ``core.llm.generate`` sera importé lazily.
        """
        self._llm_generate = llm_generate

    @property
    def name(self) -> str:
        return "llm"

    @property
    def weight(self) -> float:
        return 0.80

    async def generate(self, ctx: ReasoningContext) -> list[Hypothesis]:
        """Génère des hypothèses via le LLM avec prompt JSON contraint.

        Args:
            ctx : Contexte de raisonnement.

        Returns:
            Liste d'hypothèses (vide si LLM indisponible ou JSON invalide).
        """
        llm_fn = self._llm_generate

        # Import lazy pour éviter les imports circulaires
        if llm_fn is None:
            try:
                from core.llm import generate as _g  # noqa: PLC0415
                llm_fn = _g
            except ImportError:
                logger.debug("[LLMStrategy] core.llm introuvable — stratégie désactivée")
                return []

        try:
            analysis = ctx.analysis_or_fallback
            n_target = ctx.request.max_hypotheses
            prompt   = self._build_prompt(ctx.request.question, analysis, n_target)
            raw_text = await llm_fn(prompt)
            return self._parse_response(raw_text)

        except Exception as exc:
            logger.warning("[LLMStrategy] Échec LLM — hypothèses LLM ignorées : %s", exc)
            return []

    def _build_prompt(
        self,
        question: str,
        analysis: QuestionAnalysis,
        n: int,
    ) -> str:
        """Construit le prompt JSON contraint pour le LLM.

        Maximise la probabilité d'obtenir un JSON parsable en
        forçant le format de réponse et en interdisant tout texte libre.

        Args:
            question : Question originale de l'utilisateur.
            analysis : Analyse produite par QuestionAnalyzerAgent.
            n        : Nombre d'hypothèses à générer.

        Returns:
            Prompt formaté.
        """
        concepts = ", ".join(analysis.key_concepts[:5]) or "non identifiés"
        entities = ", ".join(analysis.key_entities[:3]) or "non identifiées"

        return f"""Tu es un expert en raisonnement analytique.
Génère exactement {n} hypothèses distinctes pour répondre à cette question.

Question : {question}
Type     : {analysis.question_type.value}
Domaine  : {analysis.domain}
Complexité : {analysis.complexity_score:.2f}
Concepts clés : {concepts}
Entités : {entities}

RÉPONDS UNIQUEMENT avec ce JSON valide. Commence directement par {{.
Aucun texte avant ou après le JSON.

{{
  "hypotheses": [
    {{
      "content": "formulation claire et complète de l'hypothèse",
      "score": 0.75,
      "justification": "explication concise de la plausibilité (1-2 phrases)"
    }}
  ]
}}

Règles :
- Chaque hypothèse apporte un angle différent (ne pas répéter la même idée)
- Le score reflète ta confiance dans l'hypothèse [0.0 → 1.0]
- La justification est concise et factuelle
- Maximum {n} hypothèses dans le tableau
- JSON pur uniquement"""

    def _parse_response(self, raw: str) -> list[Hypothesis]:
        """Parse la réponse brute du LLM en liste de Hypothesis.

        Tente plusieurs stratégies d'extraction JSON avant d'abandonner.

        Args:
            raw : Texte brut retourné par le LLM.

        Returns:
            Liste de Hypothesis parsées et validées.

        Raises:
            ValueError : Si aucune stratégie d'extraction ne fonctionne.
        """
        clean   = raw.strip()
        clean   = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.MULTILINE)
        clean   = re.sub(r"\s*```\s*$",        "", clean, flags=re.MULTILINE)
        clean   = clean.strip()

        parsed: Optional[dict[str, Any]] = None

        # Stratégie 1 : parse direct
        parsed = _try_parse_dict(clean)

        # Stratégie 2 : extraction entre première { et dernière }
        if parsed is None:
            start = clean.find("{")
            end   = clean.rfind("}") + 1
            if 0 <= start < end:
                parsed = _try_parse_dict(clean[start:end])

        if parsed is None or "hypotheses" not in parsed:
            raise ValueError(
                f"JSON sans clé 'hypotheses' dans la réponse LLM : {raw[:200]!r}"
            )

        hypotheses: list[Hypothesis] = []
        for item in parsed["hypotheses"]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            try:
                score = max(0.0, min(1.0, float(item.get("score", self.weight))))
            except (TypeError, ValueError):
                score = self.weight

            hypotheses.append(Hypothesis(
                content       = content,
                initial_score = score,
                origin        = "llm",
                justification = str(item.get("justification", "")).strip(),
                strategy_name = self.name,
            ))

        return hypotheses[:5]  # Sécurité : jamais plus de 5


# ── Utilitaires privés ────────────────────────────────────────────────────────

def _build_substitutions(analysis: QuestionAnalysis) -> dict[str, str]:
    """Prépare les variables de substitution pour les templates heuristiques.

    Args:
        analysis : QuestionAnalysis peuplé par QuestionAnalyzerAgent.

    Returns:
        Dictionnaire de substitution pour str.format().
    """
    concepts  = analysis.key_concepts
    entities  = analysis.key_entities

    return {
        "concept"  : concepts[0]  if concepts  else analysis.domain,
        "concepts" : ", ".join(concepts[:3]) if concepts else analysis.domain,
        "domain"   : analysis.domain,
        "entity_a" : entities[0]  if len(entities) > 0 else "la première option",
        "entity_b" : entities[1]  if len(entities) > 1 else "la seconde option",
    }


def _safe_format(template: str, subs: dict[str, str]) -> str:
    """Applique les substitutions sur un template sans lever d'exception.

    Si une clé est manquante, retourne le template brut.

    Args:
        template : Template avec des placeholders {key}.
        subs     : Dictionnaire de substitution.

    Returns:
        Texte formaté ou template brut en cas d'erreur.
    """
    try:
        return template.format(**subs)
    except (KeyError, IndexError):
        return template


def _try_parse_dict(text: str) -> Optional[dict[str, Any]]:
    """Tente de parser un texte JSON en dict, retourne None en cas d'échec.

    Args:
        text : Texte candidat.

    Returns:
        dict ou None.
    """
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None

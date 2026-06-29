"""Collecte de preuves pour le moteur de raisonnement — Phase 3.3.

Ce module implémente le patron Strategy pour la collecte de preuves.
Chaque stratégie est indépendante, testable isolément et extensible sans
modifier les autres. Les stratégies s'exécutent en parallèle dans
evidence_engine.py via asyncio.gather().

Stratégies disponibles :
    HypothesisEvidenceCollector  ← génère du soutien depuis les hypothèses
    CounterEvidenceCollector     ← génère des contre-preuves systématiques
    ConceptEvidenceCollector     ← ancre les preuves sur les concepts clés
    DomainEvidenceCollector      ← faits généraux du domaine détecté
    LLMEvidenceCollector         ← preuves créatives via LLM (optionnel)

Stratégies futures (Phase 3.4+) :
    ChromaDBEvidenceCollector    ← mémoire vectorielle réelle
    SupabaseEvidenceCollector    ← mémoires utilisateur persistées
    WebEvidenceCollector         ← recherche web via core.web_search

Règle d'or :
    Toute erreur dans une stratégie retourne [], jamais une exception.
    Le pipeline ne peut pas s'arrêter à cause d'une source indisponible.

Dépendances autorisées :
    ✓ models.reasoning
    ✓ core.reasoning.context
    ✗ routers/
    ✗ core.reasoning.reasoning_engine  (évite le cycle)
    ✗ core.reasoning.evidence_engine   (évite le cycle)
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable, Optional, Protocol, runtime_checkable

from models.reasoning import (
    Evidence,
    EvidenceRelation,
    EvidenceSourceType,
    EvidenceType,
    Hypothesis,
    QuestionAnalysis,
)
from core.reasoning.context import ReasoningContext

logger = logging.getLogger("makenbrain.reasoning.evidence_collector")

# ── Type de la fonction LLM injectable ───────────────────────────────────────

LLMCallable = Callable[..., Awaitable[str]]
"""Signature minimale : async (prompt: str, **kwargs) -> str."""

# ── Constantes de contenu ─────────────────────────────────────────────────────

# Faits généraux par domaine métier — utilisés par DomainEvidenceCollector
_DOMAIN_FACTS: dict[str, list[str]] = {
    "technique": [
        "Les bonnes pratiques de développement logiciel incluent les tests "
        "automatisés, la révision de code et le déploiement continu (CI/CD).",
        "La scalabilité et la maintenabilité sont des critères fondamentaux "
        "dans toute décision d'architecture logicielle.",
        "La séparation des responsabilités (SoC) réduit le couplage "
        "et facilite l'évolution du système.",
    ],
    "finance": [
        "Les décisions financières solides reposent sur des données chiffrées "
        "vérifiables et des indicateurs de performance mesurables (KPI).",
        "La gestion du risque est un principe fondamental en finance : "
        "tout investissement comporte une part d'incertitude à quantifier.",
        "La diversification des actifs est une stratégie éprouvée "
        "pour réduire l'exposition au risque.",
    ],
    "santé": [
        "Les recommandations médicales doivent être basées sur des études "
        "cliniques validées et l'avis de professionnels qualifiés.",
        "Le principe de précaution est fondamental en médecine : "
        "primum non nocere (d'abord, ne pas nuire).",
    ],
    "droit": [
        "Tout acte juridique doit respecter le cadre légal en vigueur "
        "dans la juridiction concernée.",
        "L'interprétation des textes juridiques nécessite l'expertise "
        "d'un professionnel du droit qualifié.",
    ],
    "science": [
        "Les conclusions scientifiques doivent être reproductibles "
        "et validées par une méthodologie rigoureuse.",
        "Le consensus scientifique évolue avec les nouvelles découvertes "
        "soumises à la révision par les pairs.",
    ],
    "general": [
        "Une réponse rigoureuse nécessite l'examen de plusieurs perspectives "
        "et sources d'information complémentaires.",
        "Le contexte spécifique de la situation influence fortement "
        "la pertinence de toute solution proposée.",
    ],
}

# Templates de contre-preuves — utilisés par CounterEvidenceCollector
_COUNTER_TEMPLATES: list[str] = [
    "Cependant, dans le domaine {domain}, des cas documentés montrent que "
    "cette conclusion ne s'applique pas universellement.",
    "Des travaux dans le domaine {domain} identifient des facteurs qui "
    "contredisent partiellement cette hypothèse.",
    "Le concept '{concept}' présente des exceptions notables qui "
    "limitent la portée de cette hypothèse.",
    "Bien que plausible, cette hypothèse ne prend pas en compte "
    "certaines contraintes propres au domaine {domain}.",
]

# Marqueurs linguistiques pour la classification du type de preuve
_HYPOTHESIS_MARKERS: frozenset[str] = frozenset({
    "pourrait", "peut-être", "probablement", "semble", "il est possible",
    "on suppose", "hypothèse", "supposons", "potentiellement",
    "vraisemblablement", "il se pourrait", "si l'on considère",
})
_OPINION_MARKERS: frozenset[str] = frozenset({
    "à mon avis", "je pense", "il me semble", "selon moi", "en général",
    "généralement", "certains considèrent", "d'après", "subjectivement",
    "il semblerait", "de façon générale",
})


# ── Protocole commun ──────────────────────────────────────────────────────────

@runtime_checkable
class EvidenceCollectionStrategy(Protocol):
    """Contrat que toute stratégie de collecte de preuves doit respecter.

    ``runtime_checkable`` pour les assertions isinstance() dans les tests.

    Règles d'implémentation :
        - ``collect()`` ne doit jamais lever d'exception.
        - En cas d'erreur interne, retourner [] et logger un warning.
        - Les preuves retournées ont ``strategy_name`` absent (non modélisé)
          mais ``source_ref`` identifie la stratégie source.
    """

    @property
    def name(self) -> str:
        """Identifiant unique de la stratégie (snake_case)."""
        ...  # pragma: no cover

    async def collect(self, ctx: ReasoningContext) -> list[Evidence]:
        """Collecte des preuves depuis la source de cette stratégie.

        Args:
            ctx : Contexte de raisonnement. Contient ``ctx.analysis``
                  (peuplé par Phase 3.1) et ``ctx.hypotheses`` (Phase 3.2).

        Returns:
            Liste de preuves (peut être vide, jamais None).
        """
        ...  # pragma: no cover


# ── Stratégie 1 : Preuves depuis les hypothèses ───────────────────────────────

class HypothesisEvidenceCollector:
    """Génère une preuve de soutien par hypothèse depuis son propre contenu.

    Cette stratégie extrait la substance de chaque hypothèse pour
    produire une preuve qui la soutient formellement. Elle ancre les
    preuves sur les claims déjà formulés par l'HypothesisEngine.

    Confiance : 0.60 (modérée — basée sur les hypothèses, pas sur des faits
    externes vérifiés).
    """

    @property
    def name(self) -> str:
        return "hypothesis_evidence"

    async def collect(self, ctx: ReasoningContext) -> list[Evidence]:
        """Génère une preuve de soutien pour chaque hypothèse du contexte.

        Args:
            ctx : Contexte contenant ``ctx.hypotheses``.

        Returns:
            Liste de preuves (une par hypothèse, vide si pas d'hypothèses).
        """
        try:
            if not ctx.hypotheses:
                return []

            analysis = ctx.analysis_or_fallback
            evidence: list[Evidence] = []

            for h in ctx.hypotheses:
                # Résumer l'hypothèse pour en faire une preuve de soutien
                summary = h.content[:120].rstrip(".")
                ev = Evidence(
                    content=(
                        f"L'analyse de l'hypothèse indique : '{summary}'. "
                        f"Cette position est cohérente avec les connaissances "
                        f"disponibles dans le domaine {analysis.domain}."
                    ),
                    source_type       = EvidenceSourceType.VECTOR_MEMORY,
                    source_ref        = f"hypothesis:{h.hypothesis_id}",
                    relevance_score   = 0.65,
                    credibility_score = 0.60,
                    evidence_type     = EvidenceType.HYPOTHESIS,
                    relations         = {h.hypothesis_id: EvidenceRelation.SUPPORTS},
                )
                evidence.append(ev)

            return evidence

        except Exception as exc:
            logger.warning("[HypothesisEvidenceCollector] Erreur : %s", exc)
            return []


# ── Stratégie 2 : Contre-preuves systématiques ───────────────────────────────

class CounterEvidenceCollector:
    """Génère une contre-preuve par hypothèse pour challenger chaque claim.

    La présence de contre-preuves est essentielle au raisonnement expert :
    elle force le moteur à considérer les limites de chaque hypothèse
    avant d'en sélectionner une.

    Confiance : 0.50 (modérée — contre-preuves génériques, non vérifiées).
    """

    @property
    def name(self) -> str:
        return "counter_evidence"

    async def collect(self, ctx: ReasoningContext) -> list[Evidence]:
        """Génère une contre-preuve pour chaque hypothèse.

        Args:
            ctx : Contexte avec hypothèses et analyse.

        Returns:
            Liste de contre-preuves (une par hypothèse).
        """
        try:
            if not ctx.hypotheses:
                return []

            analysis = ctx.analysis_or_fallback
            concept  = (
                analysis.key_concepts[0]
                if analysis.key_concepts
                else analysis.domain
            )
            evidence: list[Evidence] = []

            for i, h in enumerate(ctx.hypotheses):
                template = _COUNTER_TEMPLATES[i % len(_COUNTER_TEMPLATES)]
                content  = template.format(
                    domain  = analysis.domain,
                    concept = concept,
                )
                ev = Evidence(
                    content           = content,
                    source_type       = EvidenceSourceType.EXTERNAL,
                    source_ref        = f"counter:{h.hypothesis_id}",
                    relevance_score   = 0.50,
                    credibility_score = 0.50,
                    evidence_type     = EvidenceType.HYPOTHESIS,
                    relations         = {h.hypothesis_id: EvidenceRelation.CONTRADICTS},
                )
                evidence.append(ev)

            return evidence

        except Exception as exc:
            logger.warning("[CounterEvidenceCollector] Erreur : %s", exc)
            return []


# ── Stratégie 3 : Preuves conceptuelles ──────────────────────────────────────

class ConceptEvidenceCollector:
    """Génère des preuves ancrées sur les concepts clés identifiés par Phase 3.1.

    Chaque concept clé de ``ctx.analysis.key_concepts`` devient une preuve
    contextuelle neutre qui enrichit le raisonnement sans favoriser
    une hypothèse particulière.

    Confiance : 0.65 (bonne — les concepts sont extraits de la question réelle).
    """

    @property
    def name(self) -> str:
        return "concept_evidence"

    async def collect(self, ctx: ReasoningContext) -> list[Evidence]:
        """Génère une preuve contextuelle par concept clé (max 3).

        Args:
            ctx : Contexte avec analyse et hypothèses.

        Returns:
            Liste de preuves (0 à 3 selon le nombre de concepts).
        """
        try:
            analysis = ctx.analysis_or_fallback
            if not analysis.key_concepts:
                return []

            # Relations neutres vers toutes les hypothèses (contexte général)
            neutral_relations = {
                h.hypothesis_id: EvidenceRelation.NEUTRAL
                for h in ctx.hypotheses
            }

            evidence: list[Evidence] = []
            for concept in analysis.key_concepts[:3]:
                # Si le concept est mentionné dans une hypothèse → SUPPORTS
                relations = {}
                for h in ctx.hypotheses:
                    if concept.lower() in h.content.lower():
                        relations[h.hypothesis_id] = EvidenceRelation.SUPPORTS
                    else:
                        relations[h.hypothesis_id] = EvidenceRelation.NEUTRAL

                ev = Evidence(
                    content=(
                        f"Le concept '{concept}' est un élément central "
                        f"dans le domaine {analysis.domain}. "
                        f"Sa compréhension est déterminante pour "
                        f"répondre à cette question de manière rigoureuse."
                    ),
                    source_type       = EvidenceSourceType.NEURON_GRAPH,
                    source_ref        = f"concept:{concept}",
                    relevance_score   = 0.60,
                    credibility_score = 0.65,
                    evidence_type     = EvidenceType.FACT,
                    relations         = relations if relations else neutral_relations,
                )
                evidence.append(ev)

            return evidence

        except Exception as exc:
            logger.warning("[ConceptEvidenceCollector] Erreur : %s", exc)
            return []


# ── Stratégie 4 : Preuves domaine ────────────────────────────────────────────

class DomainEvidenceCollector:
    """Injecte des faits généraux reconnus du domaine détecté.

    Ces preuves sont des principes établis qui s'appliquent à l'ensemble
    du domaine et servent de contexte factuel pour le raisonnement.

    Confiance : 0.75 (élevée — faits de domaine bien établis).
    """

    @property
    def name(self) -> str:
        return "domain_evidence"

    async def collect(self, ctx: ReasoningContext) -> list[Evidence]:
        """Génère 1-2 preuves factuelles du domaine détecté.

        Args:
            ctx : Contexte avec analyse du domaine.

        Returns:
            Liste de 0 à 2 preuves factuelles (vide si pas d'hypothèses).
        """
        try:
            if not ctx.hypotheses:
                return []

            analysis     = ctx.analysis_or_fallback
            domain_facts = _DOMAIN_FACTS.get(
                analysis.domain,
                _DOMAIN_FACTS["general"],
            )

            # Relations neutres — les faits généraux de domaine ne
            # favorisent aucune hypothèse particulière
            neutral_relations = {
                h.hypothesis_id: EvidenceRelation.NEUTRAL
                for h in ctx.hypotheses
            }

            evidence: list[Evidence] = []
            for fact_content in domain_facts[:2]:
                ev = Evidence(
                    content           = fact_content,
                    source_type       = EvidenceSourceType.VECTOR_MEMORY,
                    source_ref        = f"domain_knowledge:{analysis.domain}",
                    relevance_score   = 0.55,
                    credibility_score = 0.75,
                    evidence_type     = EvidenceType.FACT,
                    relations         = dict(neutral_relations),
                )
                evidence.append(ev)

            return evidence

        except Exception as exc:
            logger.warning("[DomainEvidenceCollector] Erreur : %s", exc)
            return []


# ── Stratégie 5 : Preuves via LLM ────────────────────────────────────────────

class LLMEvidenceCollector:
    """Génère des preuves contextuelles via le LLM.

    Stratégie optionnelle à haute valeur ajoutée. Si le LLM est
    indisponible ou retourne un JSON invalide, retourne [] sans
    interrompre le pipeline.

    Confiance : 0.80 (haute — le LLM produit des preuves contextuelles
    adaptées à la question et aux hypothèses spécifiques).

    Args:
        llm_generate : Fonction LLM injectable (async str → str).
                       Si None, tente un import lazy de core.llm.generate.
        max_evidence : Nombre maximum de preuves à demander au LLM.
    """

    def __init__(
        self,
        llm_generate : Optional[LLMCallable] = None,
        max_evidence : int                   = 4,
    ) -> None:
        self._llm_generate = llm_generate
        self._max_evidence = max_evidence

    @property
    def name(self) -> str:
        return "llm_evidence"

    async def collect(self, ctx: ReasoningContext) -> list[Evidence]:
        """Collecte des preuves en interrogeant le LLM.

        Args:
            ctx : Contexte complet (analyse + hypothèses).

        Returns:
            Liste de preuves (vide si LLM indisponible ou JSON invalide).
        """
        llm_fn = self._llm_generate

        if llm_fn is None:
            try:
                from core.llm import generate as _g  # noqa: PLC0415
                llm_fn = _g
            except ImportError:
                logger.debug("[LLMEvidenceCollector] core.llm introuvable.")
                return []

        try:
            analysis = ctx.analysis_or_fallback
            prompt   = self._build_prompt(ctx, analysis)
            raw_text = await llm_fn(prompt)
            return self._parse_response(raw_text, ctx)

        except Exception as exc:
            logger.warning("[LLMEvidenceCollector] Échec LLM : %s", exc)
            return []

    def _build_prompt(
        self,
        ctx      : ReasoningContext,
        analysis : QuestionAnalysis,
    ) -> str:
        """Construit le prompt JSON contraint pour la collecte de preuves.

        Args:
            ctx      : Contexte de raisonnement.
            analysis : Analyse de la question.

        Returns:
            Prompt formaté.
        """
        hypotheses_summary = "\n".join(
            f"  - [{h.hypothesis_id}] {h.content[:100]}"
            for h in ctx.hypotheses[:3]
        ) or "  (aucune hypothèse générée)"

        return f"""Tu es un expert en recherche de preuves et contre-preuves.
Génère exactement {self._max_evidence} preuves pour évaluer les hypothèses suivantes.

Question : {ctx.request.question}
Domaine   : {analysis.domain}
Concepts  : {", ".join(analysis.key_concepts[:5]) or "non identifiés"}

Hypothèses à évaluer :
{hypotheses_summary}

RÉPONDS UNIQUEMENT avec ce JSON valide. Commence directement par {{.

{{
  "evidence": [
    {{
      "content": "contenu factuel de la preuve",
      "type": "fact|hypothesis|opinion",
      "credibility": 0.75,
      "relevance": 0.80,
      "hypothesis_id": "id_de_lhypothese_ou_null",
      "relation": "supports|contradicts|neutral"
    }}
  ]
}}

Règles :
- Génère un mélange de preuves qui soutiennent ET contredisent les hypothèses
- type=fact pour les informations vérifiables, hypothesis pour les suppositions
- credibility reflète la fiabilité de la source [0.0 → 1.0]
- hypothesis_id doit être l'un des IDs listés ci-dessus, ou null pour une preuve neutre
- JSON pur uniquement, aucun texte avant ou après"""

    def _parse_response(
        self,
        raw : str,
        ctx : ReasoningContext,
    ) -> list[Evidence]:
        """Parse la réponse brute du LLM en liste de Evidence.

        Args:
            raw : Texte JSON brut retourné par le LLM.
            ctx : Contexte (pour valider les hypothesis_id).

        Returns:
            Liste de preuves parsées.

        Raises:
            ValueError : Si aucun JSON valide n'est extractible.
        """
        clean = raw.strip()
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"\s*```\s*$",        "", clean, flags=re.MULTILINE)
        clean = clean.strip()

        parsed: Optional[dict[str, Any]] = _try_parse_dict(clean)
        if parsed is None:
            start = clean.find("{")
            end   = clean.rfind("}") + 1
            if 0 <= start < end:
                parsed = _try_parse_dict(clean[start:end])

        if parsed is None or "evidence" not in parsed:
            raise ValueError(f"JSON sans clé 'evidence' : {raw[:200]!r}")

        valid_ids  = {h.hypothesis_id for h in ctx.hypotheses}
        result     : list[Evidence] = []

        for item in parsed["evidence"]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue

            # Type de preuve
            raw_type = str(item.get("type", "fact")).lower()
            ev_type  = _parse_evidence_type(raw_type)

            # Scores
            credibility = _clamp(item.get("credibility", 0.65))
            relevance   = _clamp(item.get("relevance",   0.60))

            # Relation à une hypothèse
            hyp_id  = str(item.get("hypothesis_id", "") or "").strip()
            rel_raw = str(item.get("relation", "neutral")).lower().strip()
            rel     = _parse_relation(rel_raw)

            relations: dict[str, EvidenceRelation] = {}
            if hyp_id and hyp_id in valid_ids:
                relations[hyp_id] = rel
            else:
                # Preuve neutre pour toutes les hypothèses
                relations = {hid: EvidenceRelation.NEUTRAL for hid in valid_ids}

            result.append(Evidence(
                content           = content,
                source_type       = EvidenceSourceType.EXTERNAL,
                source_ref        = "llm_evidence_collector",
                relevance_score   = relevance,
                credibility_score = credibility,
                evidence_type     = ev_type,
                relations         = relations,
            ))

        return result[:self._max_evidence]


# ── Utilitaires privés ────────────────────────────────────────────────────────

def classify_evidence_type(content: str) -> EvidenceType:
    """Classifie le type d'une preuve depuis son contenu textuel.

    Détecte les marqueurs linguistiques pour distinguer :
        - FACT       : informations objectives, vérifiables.
        - HYPOTHESIS : suppositions, possibilités, incertitudes.
        - OPINION    : points de vue subjectifs, généralisations.

    En l'absence de marqueur → FACT par défaut (fail-safe).

    Args:
        content : Texte de la preuve.

    Returns:
        EvidenceType détecté.
    """
    lower = content.lower()
    if any(m in lower for m in _HYPOTHESIS_MARKERS):
        return EvidenceType.HYPOTHESIS
    if any(m in lower for m in _OPINION_MARKERS):
        return EvidenceType.OPINION
    return EvidenceType.FACT


def _try_parse_dict(text: str) -> Optional[dict[str, Any]]:
    """Tente de parser un texte JSON en dict, retourne None en cas d'échec."""
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _parse_evidence_type(raw: str) -> EvidenceType:
    """Convertit une chaîne brute en EvidenceType (FACT par défaut)."""
    mapping = {
        "fact": EvidenceType.FACT,
        "hypothesis": EvidenceType.HYPOTHESIS,
        "opinion": EvidenceType.OPINION,
    }
    return mapping.get(raw, EvidenceType.FACT)


def _parse_relation(raw: str) -> EvidenceRelation:
    """Convertit une chaîne brute en EvidenceRelation (NEUTRAL par défaut)."""
    mapping = {
        "supports":    EvidenceRelation.SUPPORTS,
        "contradicts": EvidenceRelation.CONTRADICTS,
        "neutral":     EvidenceRelation.NEUTRAL,
    }
    return mapping.get(raw, EvidenceRelation.NEUTRAL)


def _clamp(value: Any, default: float = 0.5) -> float:
    """Convertit une valeur en float clampé [0.0, 1.0]."""
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return default

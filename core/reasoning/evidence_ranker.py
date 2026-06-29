"""Classement et scoring des preuves — Phase 3.3.

Ce module calcule un score composite pour chaque preuve à partir de
deux dimensions orthogonales :

    relevance_score   ← pertinence de la preuve par rapport à la question
    credibility_score ← fiabilité du contenu selon son type et sa source

Score composite (utilisé pour le classement) :
    composite = relevance_score × credibility_score

Cette formule pénalise doublement les preuves peu pertinentes ET peu
crédibles, tout en récompensant celles qui excellent sur les deux axes.

Propriétés garanties :
    - Déterministe (même entrée → même sortie).
    - Aucune dépendance réseau, LLM ou base de données.
    - model_copy() préserve les relations et l'identifiant de la preuve.
    - Les instances Evidence originales ne sont jamais modifiées in-place.

Dépendances autorisées :
    ✓ models.reasoning
    ✓ core.reasoning.context
    ✗ core.reasoning.evidence_collector  (pas de dépendance circulaire)
    ✗ routers/
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from models.reasoning import (
    Evidence,
    EvidenceRelation,
    EvidenceSourceType,
    EvidenceType,
    QuestionAnalysis,
    QuestionType,
)
from core.reasoning.context import ReasoningContext

logger = logging.getLogger("makenbrain.reasoning.evidence_ranker")

# ── Constantes de scoring ─────────────────────────────────────────────────────

# Score de crédibilité de base selon le type de preuve
_CREDIBILITY_BY_TYPE: dict[EvidenceType, float] = {
    EvidenceType.FACT       : 0.75,  # informations vérifiables
    EvidenceType.HYPOTHESIS : 0.55,  # suppositions, moins certaines
    EvidenceType.OPINION    : 0.40,  # subjectif, moins fiable
}

# Bonus de crédibilité selon la source
_SOURCE_CREDIBILITY_BONUS: dict[EvidenceSourceType, float] = {
    EvidenceSourceType.USER_MEMORY   : 0.12,  # connaissance personnelle
    EvidenceSourceType.VECTOR_MEMORY : 0.08,  # base de connaissances indexée
    EvidenceSourceType.NEURON_GRAPH  : 0.08,  # connexions de concepts
    EvidenceSourceType.EXTERNAL      : 0.04,  # sources externes, moins certaines
}

# Mots-clés de type de question qui bonifient le score de pertinence
_TYPE_RELEVANCE_WORDS: dict[QuestionType, list[str]] = {
    QuestionType.COMPARATIVE  : [
        "supérieur", "adapté", "meilleur", "comparaison",
        "avantage", "différence", "versus",
    ],
    QuestionType.PROCEDURAL   : [
        "étape", "procédure", "méthode", "comment", "démarche", "processus",
    ],
    QuestionType.ANALYTICAL   : [
        "cause", "facteur", "raison", "explication", "parce que", "donc",
    ],
    QuestionType.EVALUATIVE   : [
        "recommandation", "optimal", "meilleur", "solution", "approche",
    ],
    QuestionType.HYPOTHETICAL : [
        "scénario", "impact", "risque", "conséquence", "si", "alors",
    ],
}


# ── Classe principale ─────────────────────────────────────────────────────────

@dataclass
class EvidenceRanker:
    """Classe de scoring, classement et filtrage des preuves.

    Usage typique :
        ranker = EvidenceRanker()
        scored = ranker.rank(raw_evidence, ctx, n=10)

    Ou étape par étape :
        scored   = ranker.score(evidence, ctx)
        filtered = ranker.filter_by_threshold(scored, min_score=0.25)
        top10    = ranker.top_n(filtered, n=10)

    Attributs configurables :
        concept_bonus    : Bonus de pertinence par concept clé présent.
        domain_bonus     : Bonus si le domaine de la question est mentionné.
        type_word_bonus  : Bonus par mot-clé de type de question présent.
        relation_bonus   : Bonus si la preuve supporte une hypothèse connue.
    """

    concept_bonus   : float = 0.06
    domain_bonus    : float = 0.08
    type_word_bonus : float = 0.03
    relation_bonus  : float = 0.05

    def rank(
        self,
        evidence  : list[Evidence],
        ctx       : ReasoningContext,
        n         : int = 20,
    ) -> list[Evidence]:
        """Pipeline complet : scoring → tri par composite → top-N.

        Args:
            evidence : Liste brute de preuves collectées.
            ctx      : Contexte de raisonnement (question + analyse + hypothèses).
            n        : Nombre maximum de preuves à retourner.

        Returns:
            Les ``n`` meilleures preuves triées par score composite décroissant.
        """
        if not evidence:
            return []
        scored = self.score(evidence, ctx)
        return self.top_n(scored, n)

    def score(
        self,
        evidence : list[Evidence],
        ctx      : ReasoningContext,
    ) -> list[Evidence]:
        """Recalcule relevance_score et credibility_score en contexte.

        Crée de nouvelles instances Evidence (immutabilité — les originaux
        ne sont jamais modifiés).

        Args:
            evidence : Preuves brutes avec scores initiaux.
            ctx      : Contexte pour le scoring contextuel.

        Returns:
            Nouvelles instances Evidence avec scores mis à jour.
        """
        if not evidence:
            return []

        analysis = ctx.analysis_or_fallback
        scored   : list[Evidence] = []

        for ev in evidence:
            new_relevance   = self._score_relevance(ev, analysis)
            new_credibility = self._score_credibility(ev)
            scored.append(ev.model_copy(update={
                "relevance_score"   : new_relevance,
                "credibility_score" : new_credibility,
            }))

        return scored

    def filter_by_threshold(
        self,
        evidence      : list[Evidence],
        min_composite : float = 0.25,
    ) -> list[Evidence]:
        """Filtre les preuves dont le score composite est trop faible.

        Score composite = relevance × credibility.
        Seuil par défaut très bas (0.25) pour ne pas trop appauvrir
        la base de preuves en phases initiales.

        Args:
            evidence      : Preuves à filtrer.
            min_composite : Score composite minimum [0.0, 1.0].

        Returns:
            Preuves dont le composite ≥ min_composite.
        """
        return [
            ev for ev in evidence
            if ev.relevance_score * ev.credibility_score >= min_composite
        ]

    def top_n(
        self,
        evidence : list[Evidence],
        n        : int,
    ) -> list[Evidence]:
        """Retourne les N meilleures preuves par score composite décroissant.

        Args:
            evidence : Preuves scorées.
            n        : Nombre maximum de résultats.

        Returns:
            Sous-liste triée de longueur min(len(evidence), n).
        """
        if n <= 0:
            return []
        return sorted(
            evidence,
            key     = lambda ev: ev.relevance_score * ev.credibility_score,
            reverse = True,
        )[:n]

    def composite_score(self, evidence: Evidence) -> float:
        """Retourne le score composite d'une preuve.

        Utilitaire public pour les tests et l'introspection.

        Args:
            evidence : Preuve scorée.

        Returns:
            Score composite = relevance × credibility, arrondi à 4 décimales.
        """
        return round(evidence.relevance_score * evidence.credibility_score, 4)

    # ── Méthodes privées ──────────────────────────────────────────────────────

    def _score_relevance(
        self,
        evidence : Evidence,
        analysis : QuestionAnalysis,
    ) -> float:
        """Calcule le score de pertinence d'une preuve.

        Formule :
            score = base_initiale
                  + Σ concept_bonus   (concepts clés présents dans le contenu)
                  + domain_bonus      (domaine mentionné dans le contenu)
                  + Σ type_word_bonus (mots-clés du QuestionType présents)
                  + relation_bonus    (preuve liée à une hypothèse connue)

        Args:
            evidence : Preuve à scorer.
            analysis : Analyse de la question.

        Returns:
            Score de pertinence entre 0.0 et 1.0 (4 décimales).
        """
        content_lower = evidence.content.lower()
        score         = evidence.relevance_score  # valeur initiale de la stratégie

        # Bonus : concepts clés présents dans le contenu
        concept_hits = sum(
            1 for c in analysis.key_concepts
            if c.lower() in content_lower
        )
        score += min(concept_hits * self.concept_bonus, 0.24)

        # Bonus : domaine mentionné
        if analysis.domain and analysis.domain.lower() in content_lower:
            score += self.domain_bonus

        # Bonus : mots-clés du QuestionType présents
        type_words  = _TYPE_RELEVANCE_WORDS.get(analysis.question_type, [])
        type_hits   = sum(1 for w in type_words if w in content_lower)
        score       += min(type_hits * self.type_word_bonus, 0.12)

        # Bonus : preuve liée à une hypothèse connue
        hypothesis_ids = {h.hypothesis_id for h in []}  # pas d'accès aux hypothèses ici
        # On utilise le fait que relations non-vide signifie une relation spécifique
        if evidence.relations and any(
            r in (EvidenceRelation.SUPPORTS, EvidenceRelation.CONTRADICTS)
            for r in evidence.relations.values()
        ):
            score += self.relation_bonus

        return round(min(score, 1.0), 4)

    def _score_credibility(self, evidence: Evidence) -> float:
        """Calcule le score de crédibilité d'une preuve.

        Formule :
            score = base_par_type + bonus_par_source

        La valeur initiale ``credibility_score`` de la stratégie est
        remplacée par ce calcul contextuel plus fin.

        Args:
            evidence : Preuve à scorer.

        Returns:
            Score de crédibilité entre 0.0 et 1.0 (4 décimales).
        """
        base  = _CREDIBILITY_BY_TYPE.get(evidence.evidence_type, 0.55)
        bonus = _SOURCE_CREDIBILITY_BONUS.get(evidence.source_type, 0.04)
        return round(min(base + bonus, 1.0), 4)

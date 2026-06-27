"""Classement, déduplication et filtrage des hypothèses — Phase 3.2.

Ce module est responsable de la qualité de la liste finale d'hypothèses
retournée par HypothesisEngine. Il opère en trois étapes :

    1. Déduplication  → supprime les hypothèses trop similaires (Jaccard ≥ seuil).
    2. Scoring        → recalcule un score final tenant compte de la pertinence
                        au contexte (concepts, domaine, type de question).
    3. Sélection      → trie par score décroissant et retient les N meilleures.

Propriétés garanties :
    - Toutes les opérations sont déterministes (pas de hasard).
    - Aucune dépendance réseau, LLM ou base de données.
    - Fully testable en isolation.
    - Score final toujours dans [0.0, 1.0].

Algorithme de déduplication (Jaccard sur sacs de mots) :
    sim(a, b) = |mots(a) ∩ mots(b)| / |mots(a) ∪ mots(b)|
    Si sim ≥ threshold → on conserve celle avec le score le plus élevé.

Dépendances autorisées :
    ✓ models.reasoning
    ✓ core.reasoning.context
    ✗ core.reasoning.hypothesis_strategies  (évite le cycle)
    ✗ routers/
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from models.reasoning import Hypothesis, QuestionType
from core.reasoning.context import ReasoningContext

logger = logging.getLogger("makenbrain.reasoning.hypothesis_ranker")

# ── Constantes ────────────────────────────────────────────────────────────────

DEFAULT_SIMILARITY_THRESHOLD: float = 0.55
"""Seuil de similarité Jaccard au-dessus duquel deux hypothèses sont
considérées comme doublons. 0.55 = 55% de mots en commun."""

# Mots-clés qui, présents dans le contenu, bonifient le score
# selon le QuestionType de la question.
_TYPE_BONUS_WORDS: dict[QuestionType, list[str]] = {
    QuestionType.COMPARATIVE: [
        "supérieur", "adapté", "complémentaire", "meilleur",
        "avantage", "compromis", "versus", "trade-off",
    ],
    QuestionType.PROCEDURAL: [
        "étape", "procédure", "approche", "méthode", "démarche",
        "séquentiel", "workflow",
    ],
    QuestionType.ANALYTICAL: [
        "cause", "facteur", "raison", "explication", "phénomène",
        "systémique", "impact",
    ],
    QuestionType.EVALUATIVE: [
        "recommandation", "optimal", "solution", "compromis",
        "bonne pratique", "contexte",
    ],
    QuestionType.HYPOTHETICAL: [
        "scénario", "impact", "transition", "risque", "cascade",
        "instabilité", "probable",
    ],
}


# ── Classe principale ─────────────────────────────────────────────────────────

@dataclass
class HypothesisRanker:
    """Classe de classement, déduplication et filtrage des hypothèses.

    Usage typique (par HypothesisEngine) :
        ranker = HypothesisRanker()
        best   = ranker.rank(all_hypotheses, ctx, n=3)

    Ou étape par étape :
        unique  = ranker.deduplicate(hypotheses)
        scored  = ranker.score(unique, ctx)
        top3    = ranker.top_n(scored, n=3)

    Attributs :
        similarity_threshold : Seuil Jaccard pour la déduplication [0.0 → 1.0].
        concept_bonus        : Bonus par concept clé trouvé dans le contenu.
        domain_bonus         : Bonus si le domaine est mentionné dans le contenu.
        type_word_bonus      : Bonus par mot lié au QuestionType trouvé.
    """

    similarity_threshold : float = DEFAULT_SIMILARITY_THRESHOLD
    concept_bonus        : float = 0.04
    domain_bonus         : float = 0.05
    type_word_bonus      : float = 0.03

    def rank(
        self,
        hypotheses  : list[Hypothesis],
        ctx         : ReasoningContext,
        n           : int,
    ) -> list[Hypothesis]:
        """Pipeline complet : déduplication → scoring → top-N.

        Point d'entrée principal utilisé par HypothesisEngine.

        Args:
            hypotheses : Liste brute produite par toutes les stratégies.
            ctx        : Contexte de raisonnement (pour le scoring contextuel).
            n          : Nombre maximum d'hypothèses à retourner.

        Returns:
            Les ``n`` meilleures hypothèses uniques, triées par score décroissant.
        """
        if not hypotheses:
            return []

        unique = self.deduplicate(hypotheses)
        scored = self.score(unique, ctx)
        return self.top_n(scored, n)

    def deduplicate(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Supprime les hypothèses trop similaires.

        Pour chaque paire (a, b) avec similarité ≥ ``similarity_threshold``,
        on conserve celle dont le score ``initial_score`` est le plus élevé.

        L'algorithme est O(n²) mais ``n`` est toujours petit (≤ 15 typiquement).

        Args:
            hypotheses : Liste brute (peut contenir des doublons).

        Returns:
            Liste dédupliquée, ordre de score préservé.
        """
        if len(hypotheses) <= 1:
            return list(hypotheses)

        kept: list[Hypothesis] = []

        for candidate in hypotheses:
            is_duplicate = False
            for i, existing in enumerate(kept):
                sim = self._jaccard(candidate.content, existing.content)
                if sim >= self.similarity_threshold:
                    # Garde le meilleur score entre les deux
                    if candidate.initial_score > existing.initial_score:
                        kept[i] = candidate
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(candidate)

        logger.debug(
            "[Ranker] Déduplication : %d → %d hypothèses",
            len(hypotheses),
            len(kept),
        )
        return kept

    def score(
        self,
        hypotheses : list[Hypothesis],
        ctx        : ReasoningContext,
    ) -> list[Hypothesis]:
        """Recalcule les scores finaux en tenant compte du contexte.

        Le score final est une combinaison pondérée de :
            - Le score initial de la stratégie (base).
            - Un bonus de pertinence conceptuelle (concepts de la question).
            - Un bonus de pertinence du domaine.
            - Un bonus d'alignement avec le type de question.

        Les hypothèses ne sont pas modifiées in-place : de nouvelles
        instances sont créées avec le score recalculé (immutabilité).

        Args:
            hypotheses : Liste dédupliquée.
            ctx        : Contexte de raisonnement.

        Returns:
            Nouvelles instances Hypothesis avec ``initial_score`` recalculé.
        """
        analysis = ctx.analysis_or_fallback
        scored   : list[Hypothesis] = []

        for h in hypotheses:
            final_score = self._compute_score(h, analysis)
            # Créer une nouvelle instance avec le score mis à jour
            scored.append(h.model_copy(update={"initial_score": final_score}))

        return scored

    def top_n(self, hypotheses: list[Hypothesis], n: int) -> list[Hypothesis]:
        """Retourne les N meilleures hypothèses triées par score décroissant.

        Si ``len(hypotheses) <= n``, toutes sont retournées.

        Args:
            hypotheses : Liste scorée.
            n          : Nombre maximum de résultats souhaités.

        Returns:
            Sous-liste triée, de longueur min(len(hypotheses), n).
        """
        if n <= 0:
            return []
        sorted_h = sorted(
            hypotheses,
            key     = lambda h: h.initial_score,
            reverse = True,
        )
        return sorted_h[:n]

    # ── Méthodes privées ──────────────────────────────────────────────────────

    def _jaccard(self, a: str, b: str) -> float:
        """Calcule la similarité de Jaccard entre deux textes.

        Définition : |A ∩ B| / |A ∪ B| sur les ensembles de mots.

        - Retourne 1.0 si les deux chaînes sont identiques ou vides.
        - Retourne 0.0 si l'une est vide et l'autre non.
        - Opère sur les formes minuscules pour être insensible à la casse.

        Args:
            a : Premier texte.
            b : Second texte.

        Returns:
            Score de similarité entre 0.0 et 1.0.
        """
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())

        if not words_a and not words_b:
            return 1.0
        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union        = words_a | words_b
        return len(intersection) / len(union)

    def _compute_score(
        self,
        hypothesis : Hypothesis,
        analysis   : "QuestionAnalysis",  # noqa: F821 — forward ref ok
    ) -> float:
        """Calcule le score final d'une hypothèse en contexte.

        Formule :
            score = base
                  + Σ concept_bonus pour chaque concept dans le contenu
                  + domain_bonus si domaine mentionné
                  + Σ type_word_bonus pour chaque mot-clé de type présent

        Le résultat est clampé entre 0.0 et 1.0.

        Args:
            hypothesis : Hypothèse à scorer.
            analysis   : Analyse de la question (contexte).

        Returns:
            Score final entre 0.0 et 1.0 (4 décimales).
        """
        content_lower = hypothesis.content.lower()
        score         = hypothesis.initial_score  # base fournie par la stratégie

        # Bonus de pertinence conceptuelle
        concept_hits = sum(
            1 for c in analysis.key_concepts
            if c.lower() in content_lower
        )
        score += min(concept_hits * self.concept_bonus, 0.20)

        # Bonus de pertinence du domaine
        if analysis.domain and analysis.domain in content_lower:
            score += self.domain_bonus

        # Bonus d'alignement avec le QuestionType
        type_words  = _TYPE_BONUS_WORDS.get(analysis.question_type, [])
        type_hits   = sum(1 for w in type_words if w in content_lower)
        score       += min(type_hits * self.type_word_bonus, 0.12)

        return round(min(score, 1.0), 4)

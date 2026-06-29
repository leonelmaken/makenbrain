"""Validation et évaluation des preuves — Phase 3.3.

Ce module analyse la cohérence de l'ensemble des preuves collectées
et produit un ``EvidenceEvaluation`` structuré qui servira à
l'EvidenceEngine (et au futur Synthesizer en Phase 3.4) pour
sélectionner la meilleure hypothèse.

Responsabilités :
    - Détecter les contradictions entre preuves.
    - Identifier les lacunes de connaissance (hypothèses sans soutien).
    - Calculer un score de soutien net par hypothèse.
    - Évaluer la qualité globale de la base de preuves.
    - Produire le ``EvidenceEvaluation`` final.

Propriétés garanties :
    - Toutes les opérations sont déterministes.
    - Aucune dépendance LLM, réseau ou base de données.
    - Scores toujours dans [0.0, 1.0].

Algorithme de score de soutien (support_score) :
    soutien_brut = Σ (relevance × credibility) pour les preuves SUPPORTS
    contra_brut  = Σ (relevance × credibility) pour les preuves CONTRADICTS
    net          = soutien_brut - contra_brut
    score        = clamp((net + 1.0) / 2.0, 0.0, 1.0)

    Cette formule normalise le score net dans [0.0, 1.0] :
        net = +1 → score = 1.0  (soutien maximal)
        net =  0 → score = 0.5  (équilibre, incertitude)
        net = -1 → score = 0.0  (contradiction totale)

Dépendances autorisées :
    ✓ models.reasoning
    ✓ core.reasoning.context
    ✗ core.reasoning.evidence_collector  (pas de cycle)
    ✗ core.reasoning.evidence_ranker     (pas de cycle)
    ✗ routers/
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from models.reasoning import (
    Evidence,
    EvidenceEvaluation,
    EvidenceRelation,
    Hypothesis,
)

logger = logging.getLogger("makenbrain.reasoning.evidence_validator")


@dataclass
class EvidenceValidator:
    """Valide et évalue la cohérence d'un ensemble de preuves.

    Usage typique (par EvidenceEngine) :
        validator  = EvidenceValidator()
        evaluation = validator.build_evaluation(ctx.hypotheses, ctx.evidence)
        ctx.evaluation = evaluation

    Ou étape par étape :
        contradictions = validator.detect_contradictions(evidence, hypotheses)
        gaps           = validator.identify_knowledge_gaps(hypotheses, evidence)
        support_scores = validator.compute_support_scores(hypotheses, evidence)
        quality        = validator.compute_overall_quality(evidence)
    """

    def build_evaluation(
        self,
        hypotheses : list[Hypothesis],
        evidence   : list[Evidence],
    ) -> EvidenceEvaluation:
        """Produit l'évaluation complète de la base de preuves.

        Point d'entrée principal. Agrège toutes les analyses en un
        ``EvidenceEvaluation`` structuré.

        Args:
            hypotheses : Liste des hypothèses (de HypothesisEngine).
            evidence   : Liste des preuves scorées (de EvidenceRanker).

        Returns:
            EvidenceEvaluation avec tous les champs renseignés.
        """
        if not hypotheses and not evidence:
            return EvidenceEvaluation()

        contradictions       = self.detect_contradictions(evidence, hypotheses)
        gaps                 = self.identify_knowledge_gaps(hypotheses, evidence)
        support_scores       = self.compute_support_scores(hypotheses, evidence)
        quality              = self.compute_overall_quality(evidence)
        evidence_per_hyp     = self._count_evidence_per_hypothesis(
            hypotheses, evidence
        )

        # Meilleure hypothèse = score de soutien le plus élevé
        best_hypothesis_id: str | None = None
        if support_scores:
            best_hypothesis_id = max(
                support_scores,
                key=lambda k: support_scores[k],
            )

        logger.debug(
            "[EvidenceValidator] %d preuves | %d contradictions | "
            "%d lacunes | qualité=%.2f | meilleure hyp=%s",
            len(evidence),
            len(contradictions),
            len(gaps),
            quality,
            best_hypothesis_id,
        )

        return EvidenceEvaluation(
            total_evidence_count    = len(evidence),
            contradictions          = contradictions,
            knowledge_gaps          = gaps,
            overall_quality_score   = quality,
            best_hypothesis_id      = best_hypothesis_id,
            support_scores          = support_scores,
            evidence_per_hypothesis = evidence_per_hyp,
        )

    def detect_contradictions(
        self,
        evidence   : list[Evidence],
        hypotheses : list[Hypothesis],
    ) -> list[tuple[str, str]]:
        """Détecte les paires de preuves contradictoires pour la même hypothèse.

        Définition de contradiction :
            Deux preuves E1 et E2 sont contradictoires vis-à-vis d'une
            hypothèse H si E1 SUPPORTS H et E2 CONTRADICTS H.

        L'ordre dans la paire est normalisé (min, max) pour éviter les
        doublons (E1, E2) et (E2, E1).

        Args:
            evidence   : Toutes les preuves collectées.
            hypotheses : Toutes les hypothèses générées.

        Returns:
            Liste de paires (evidence_id_a, evidence_id_b) ordonnées.
        """
        contradictions : set[tuple[str, str]] = set()

        for h in hypotheses:
            hid = h.hypothesis_id

            supporting   = [
                ev for ev in evidence
                if ev.relations.get(hid) == EvidenceRelation.SUPPORTS
            ]
            contradicting = [
                ev for ev in evidence
                if ev.relations.get(hid) == EvidenceRelation.CONTRADICTS
            ]

            for s in supporting:
                for c in contradicting:
                    # Normalisation de l'ordre pour éviter les doublons
                    pair = tuple(sorted([s.evidence_id, c.evidence_id]))
                    contradictions.add(pair)  # type: ignore[arg-type]

        return sorted(contradictions)  # type: ignore[return-value]

    def identify_knowledge_gaps(
        self,
        hypotheses : list[Hypothesis],
        evidence   : list[Evidence],
    ) -> list[str]:
        """Identifie les hypothèses sans aucune preuve de soutien.

        Une lacune de connaissance signifie que le moteur ne dispose
        pas de preuves pour défendre une hypothèse — elle ne peut
        pas être recommandée en confiance.

        Args:
            hypotheses : Hypothèses générées par HypothesisEngine.
            evidence   : Preuves collectées par EvidenceEngine.

        Returns:
            Liste de descriptions textuelles des lacunes.
        """
        gaps: list[str] = []

        for h in hypotheses:
            hid         = h.hypothesis_id
            has_support = any(
                ev.relations.get(hid) == EvidenceRelation.SUPPORTS
                for ev in evidence
            )
            if not has_support:
                summary = h.content[:80].rstrip(".")
                gaps.append(
                    f"Aucune preuve disponible pour soutenir : '{summary}'"
                )

        return gaps

    def compute_support_scores(
        self,
        hypotheses : list[Hypothesis],
        evidence   : list[Evidence],
    ) -> dict[str, float]:
        """Calcule le score de soutien net de chaque hypothèse.

        Algorithme :
            soutien  = Σ (relevance × credibility) pour les preuves SUPPORTS
            contra   = Σ (relevance × credibility) pour les preuves CONTRADICTS
            net      = soutien - contra
            score    = clamp((net + 1.0) / 2.0, 0.0, 1.0)

        Interprétation :
            1.0 → soutenu sans contradiction
            0.5 → équilibre (incertain)
            0.0 → uniquement contredit

        Args:
            hypotheses : Hypothèses à évaluer.
            evidence   : Preuves disponibles.

        Returns:
            Dict {hypothesis_id: support_score} avec scores dans [0.0, 1.0].
        """
        scores: dict[str, float] = {}

        for h in hypotheses:
            hid = h.hypothesis_id

            support_sum = sum(
                ev.relevance_score * ev.credibility_score
                for ev in evidence
                if ev.relations.get(hid) == EvidenceRelation.SUPPORTS
            )
            contra_sum = sum(
                ev.relevance_score * ev.credibility_score
                for ev in evidence
                if ev.relations.get(hid) == EvidenceRelation.CONTRADICTS
            )

            net   = support_sum - contra_sum
            score = max(0.0, min(1.0, (net + 1.0) / 2.0))
            scores[hid] = round(score, 4)

        return scores

    def compute_overall_quality(
        self,
        evidence : list[Evidence],
    ) -> float:
        """Calcule la qualité globale de la base de preuves.

        Qualité = moyenne des scores composites (relevance × credibility)
        de toutes les preuves disponibles.

        Une qualité élevée (> 0.6) indique une base de preuves fiable
        pour la prise de décision finale du Synthesizer.

        Args:
            evidence : Toutes les preuves disponibles.

        Returns:
            Score de qualité entre 0.0 et 1.0 (4 décimales).
            Retourne 0.0 si la liste est vide.
        """
        if not evidence:
            return 0.0
        composites = [ev.relevance_score * ev.credibility_score for ev in evidence]
        return round(sum(composites) / len(composites), 4)

    # ── Méthodes privées ──────────────────────────────────────────────────────

    def _count_evidence_per_hypothesis(
        self,
        hypotheses : list[Hypothesis],
        evidence   : list[Evidence],
    ) -> dict[str, int]:
        """Compte les preuves (soutien + contradiction) par hypothèse.

        Les preuves neutres ne sont pas comptabilisées : elles n'apportent
        pas d'argument direct pour ou contre une hypothèse.

        Args:
            hypotheses : Liste des hypothèses.
            evidence   : Toutes les preuves.

        Returns:
            Dict {hypothesis_id: count_evidence}.
        """
        counts: dict[str, int] = {}

        for h in hypotheses:
            hid = h.hypothesis_id
            counts[hid] = sum(
                1 for ev in evidence
                if ev.relations.get(hid) in (
                    EvidenceRelation.SUPPORTS,
                    EvidenceRelation.CONTRADICTS,
                )
            )

        return counts

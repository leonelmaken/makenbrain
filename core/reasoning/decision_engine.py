"""Moteur de décision finale — Phase 3.4.

Quatrième étape du pipeline de raisonnement. Reçoit les hypothèses
(Phase 3.2) et les preuves évaluées (Phase 3.3) pour produire une
décision finale argumentée : quelle hypothèse retenir, pourquoi, et
quels risques subsistent.

Le DecisionEngine ne génère ni hypothèses ni preuves nouvelles — il
arbitre entre celles déjà collectées par les étapes précédentes.
Cette séparation des responsabilités garde chaque étape testable
isolément et garantit qu'aucune décision n'est prise sur des données
non vérifiées.

Algorithme de décision (par hypothèse) :
    1. evidence_score               ← repris de EvidenceEvaluation.support_scores
    2. contradiction_penalty        ← proportionnel aux contradictions ciblant l'hypothèse
    3. missing_information_penalty  ← proportionnel aux lacunes de connaissance
    4. memory_bonus                 ← bonus si soutenue par EvidenceSourceType.USER_MEMORY
    5. graph_bonus                  ← bonus si soutenue par EvidenceSourceType.NEURON_GRAPH
    6. confidence                   ← evidence_score pondéré par le volume de preuves
    7. global_score                 ← agrégation finale (voir _compute_global_score)

L'hypothèse au global_score le plus élevé est sélectionnée. En cas
d'égalité stricte, la première hypothèse de la liste d'origine est
retenue (déterminisme garanti, pas de hasard).

Dépendances autorisées :
    ✓ core.reasoning.context
    ✓ models.reasoning
    ✗ routers/  (jamais)
    ✗ core.reasoning.reasoning_engine  (évite le cycle)
    ✗ core.reasoning.evidence_engine, hypothesis_engine  (pas de couplage direct —
      le DecisionEngine consomme uniquement ctx.hypotheses, ctx.evidence, ctx.evaluation)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from models.reasoning import (
    DecisionCandidate,
    DecisionReason,
    DecisionResult,
    DecisionScore,
    Evidence,
    EvidenceEvaluation,
    EvidenceRelation,
    EvidenceSourceType,
    Hypothesis,
    ReasoningStepRecord,
)
from core.reasoning.context import ReasoningContext

logger = logging.getLogger("makenbrain.reasoning.decision_engine")

# ── Constantes de pondération ─────────────────────────────────────────────────

# Poids de chaque composante dans le score global.
# La somme des poids positifs (hors pénalités) vaut 1.0 pour garder
# global_score interprétable dans [0.0, 1.0] avant clamp final.
_WEIGHT_EVIDENCE     : float = 0.55
_WEIGHT_CONFIDENCE   : float = 0.20
_WEIGHT_MEMORY_BONUS : float = 0.10
_WEIGHT_GRAPH_BONUS  : float = 0.15

# Pénalité maximale appliquée par contradiction (clampée au global)
_CONTRADICTION_PENALTY_STEP : float = 0.15
_MAX_CONTRADICTION_PENALTY  : float = 0.60

# Pénalité fixe si l'hypothèse n'a aucune preuve de soutien
_MISSING_INFO_PENALTY_NO_SUPPORT : float = 0.30

# Bonus fixe accordé par preuve de soutien d'une source donnée (plafonné)
_MEMORY_BONUS_PER_EVIDENCE : float = 0.20
_GRAPH_BONUS_PER_EVIDENCE  : float = 0.15
_MAX_SOURCE_BONUS          : float = 0.40

# Seuils de qualité de décision (alignés sur les seuils de risk_level existants)
_QUALITY_HIGH_THRESHOLD   : float = 0.75
_QUALITY_MEDIUM_THRESHOLD : float = 0.55


# ── Agent (protocole ReasoningAgent) ─────────────────────────────────────────

class DecisionEngineAgent:
    """Agent de décision finale — étape 4 du pipeline de raisonnement.

    Implémente le protocole ``ReasoningAgent`` défini dans
    ``core.reasoning.__init__``.

    Responsabilités :
        - Calculer un score détaillé (DecisionScore) pour chaque hypothèse.
        - Sélectionner la meilleure hypothèse selon le score global.
        - Construire une justification structurée (DecisionReason) :
          raisons de sélection, raisons de rejet par hypothèse rejetée,
          informations manquantes, risques résiduels.
        - Calculer la confiance globale de la décision (intègre analyse,
          hypothèses, preuves, contradictions et score de décision).
        - Peupler ``ctx.decision`` et mettre à jour ``ctx.confidence``.
        - Alimenter ``ctx.trace`` avec un ``ReasoningStepRecord``.

    Le moteur fonctionne même sans hypothèses (retourne une décision vide
    avec confiance nulle) et même sans preuves (toutes les hypothèses
    reçoivent alors la pénalité missing_information maximale).

    Usage dans le pipeline :
        Instancié une fois dans ``DEFAULT_PIPELINE`` de reasoning_engine.py.
    """

    name: str = "decision_engine"

    async def run(self, ctx: ReasoningContext) -> ReasoningContext:
        """Exécute la prise de décision et enrichit le contexte.

        Délègue à ``make_decision()`` pour rester testable indépendamment
        du protocole agent.

        Args:
            ctx : Contexte de raisonnement courant.

        Returns:
            ctx enrichi avec ctx.decision peuplé et ctx.confidence mis à jour.
        """
        return await make_decision(ctx)


# ── Point d'entrée public ─────────────────────────────────────────────────────

async def make_decision(ctx: ReasoningContext) -> ReasoningContext:
    """Calcule la décision finale et peuple ctx.decision.

    Fonction publique du module, testable sans instancier l'agent.

    Étapes :
        1. Si aucune hypothèse → décision vide, confiance nulle, retour anticipé.
        2. Calculer un DecisionScore pour chaque hypothèse (DecisionScorer).
        3. Construire la liste de DecisionCandidate triée par score décroissant.
        4. Sélectionner le meilleur candidat (premier de la liste triée).
        5. Construire la justification (DecisionReasoner).
        6. Calculer la confiance globale (intègre tout le pipeline).
        7. Peupler ctx.decision et ctx.confidence.
        8. Enregistrer dans ctx.trace.

    Args:
        ctx : Contexte de raisonnement courant (analysis, hypotheses,
              evidence, evaluation déjà peuplés par les étapes précédentes).

    Returns:
        ctx enrichi.
    """
    step_start = time.monotonic()
    scorer     = DecisionScorer()
    reasoner   = DecisionReasoner()

    logger.info(
        "[DecisionEngine] Décision | %d hypothèse(s) | %d preuve(s)",
        len(ctx.hypotheses),
        len(ctx.evidence),
    )

    # ── Cas limite : aucune hypothèse disponible ──────────────────────────────
    if not ctx.hypotheses:
        decision = DecisionResult(
            candidates              = [],
            selected_hypothesis_id  = None,
            reason                  = DecisionReason(
                missing_information=["Aucune hypothèse n'a été générée."],
                residual_risks=["Décision impossible sans hypothèse candidate."],
            ),
            overall_confidence      = 0.0,
            decision_quality        = "low",
        )
        ctx.decision   = decision
        ctx.confidence = 0.0

        duration_ms = (time.monotonic() - step_start) * 1000
        ctx.trace.steps.append(ReasoningStepRecord(
            step_name      = "decision_engine",
            duration_ms    = round(duration_ms, 2),
            success        = True,
            input_summary  = "0 hypothèse disponible",
            output_summary = "Décision vide — aucune hypothèse à évaluer.",
            degraded       = True,
            error_message  = "Pipeline en amont n'a produit aucune hypothèse.",
        ))
        ctx.mark_degraded("decision_engine", "Aucune hypothèse disponible pour décision.")
        return ctx

    # ── Calcul du score de chaque hypothèse ───────────────────────────────────
    evaluation = ctx.evaluation or EvidenceEvaluation()
    candidates : list[DecisionCandidate] = []

    for h in ctx.hypotheses:
        score = scorer.score_hypothesis(h, ctx.evidence, evaluation)
        candidates.append(DecisionCandidate(
            hypothesis_id = h.hypothesis_id,
            content       = h.content,
            score         = score,
            selected      = False,
        ))

    # ── Tri par score global décroissant (stable → déterministe) ─────────────
    candidates.sort(key=lambda c: c.score.global_score, reverse=True)

    # ── Sélection du meilleur candidat ────────────────────────────────────────
    best = candidates[0]
    best_index = 0
    # Marquer le candidat sélectionné via model_copy (immutabilité Pydantic)
    candidates[best_index] = best.model_copy(update={"selected": True})
    selected_id = candidates[best_index].hypothesis_id

    # ── Construction de la justification ──────────────────────────────────────
    reason = reasoner.build_reason(
        candidates = candidates,
        selected_id = selected_id,
        evaluation  = evaluation,
        ctx         = ctx,
    )

    # ── Confiance globale et qualité de décision ──────────────────────────────
    overall_confidence = _compute_overall_confidence(ctx, candidates[best_index].score)
    quality            = _compute_decision_quality(overall_confidence)

    decision = DecisionResult(
        candidates             = candidates,
        selected_hypothesis_id = selected_id,
        reason                 = reason,
        overall_confidence     = overall_confidence,
        decision_quality       = quality,
    )

    ctx.decision   = decision
    ctx.confidence = overall_confidence

    duration_ms = (time.monotonic() - step_start) * 1000

    ctx.trace.steps.append(ReasoningStepRecord(
        step_name      = "decision_engine",
        duration_ms    = round(duration_ms, 2),
        success        = True,
        input_summary  = (
            f"{len(ctx.hypotheses)} hypothèse(s) | {len(ctx.evidence)} preuve(s) | "
            f"qualité_preuves={evaluation.overall_quality_score:.2f}"
        ),
        output_summary = (
            f"sélectionnée={selected_id} | "
            f"score={candidates[best_index].score.global_score:.2f} | "
            f"confiance={overall_confidence:.2f} | qualité={quality} | "
            f"rejetées={len(candidates) - 1}"
        ),
        degraded       = quality == "low",
        error_message  = (
            "Confiance de décision faible." if quality == "low" else None
        ),
    ))

    logger.info(
        "[DecisionEngine] Terminé %.0fms | sélectionnée=%s | score=%.2f | "
        "confiance=%.2f | qualité=%s",
        duration_ms, selected_id,
        candidates[best_index].score.global_score,
        overall_confidence, quality,
    )

    return ctx


# ── DecisionScorer ────────────────────────────────────────────────────────────

@dataclass
class DecisionScorer:
    """Calcule le DecisionScore détaillé d'une hypothèse donnée.

    Isolé de l'orchestration (make_decision) pour rester testable
    indépendamment, sans avoir à construire un ReasoningContext complet.

    Tous les calculs sont déterministes et n'effectuent aucun appel
    réseau, LLM ou base de données.
    """

    def score_hypothesis(
        self,
        hypothesis : Hypothesis,
        evidence   : list[Evidence],
        evaluation : EvidenceEvaluation,
    ) -> DecisionScore:
        """Calcule le score complet d'une hypothèse.

        Args:
            hypothesis : Hypothèse à scorer.
            evidence   : Toutes les preuves disponibles (Phase 3.3).
            evaluation : Évaluation globale des preuves (Phase 3.3).

        Returns:
            DecisionScore avec toutes les composantes renseignées.
        """
        hid = hypothesis.hypothesis_id

        evidence_score = evaluation.support_scores.get(hid, 0.5)

        related_evidence = [
            ev for ev in evidence
            if ev.relations.get(hid) in (
                EvidenceRelation.SUPPORTS, EvidenceRelation.CONTRADICTS
            )
        ]
        supporting_evidence = [
            ev for ev in evidence
            if ev.relations.get(hid) == EvidenceRelation.SUPPORTS
        ]

        confidence = self._compute_confidence(evidence_score, related_evidence)
        contradiction_penalty = self._compute_contradiction_penalty(
            hid, evaluation.contradictions, evidence
        )
        missing_info_penalty = self._compute_missing_info_penalty(
            hid, supporting_evidence, evaluation.knowledge_gaps, hypothesis
        )
        memory_bonus = self._compute_source_bonus(
            supporting_evidence,
            EvidenceSourceType.USER_MEMORY,
            _MEMORY_BONUS_PER_EVIDENCE,
        )
        graph_bonus = self._compute_source_bonus(
            supporting_evidence,
            EvidenceSourceType.NEURON_GRAPH,
            _GRAPH_BONUS_PER_EVIDENCE,
        )

        global_score = self._compute_global_score(
            evidence_score        = evidence_score,
            confidence             = confidence,
            contradiction_penalty  = contradiction_penalty,
            missing_info_penalty   = missing_info_penalty,
            memory_bonus           = memory_bonus,
            graph_bonus            = graph_bonus,
        )

        return DecisionScore(
            evidence_score               = round(evidence_score, 4),
            confidence                   = round(confidence, 4),
            contradiction_penalty        = round(contradiction_penalty, 4),
            missing_information_penalty  = round(missing_info_penalty, 4),
            memory_bonus                  = round(memory_bonus, 4),
            graph_bonus                   = round(graph_bonus, 4),
            global_score                  = round(global_score, 4),
        )

    # ── Composantes individuelles ─────────────────────────────────────────────

    def _compute_confidence(
        self,
        evidence_score   : float,
        related_evidence : list[Evidence],
    ) -> float:
        """Calcule la confiance composite d'une hypothèse.

        La confiance pondère evidence_score par le volume de preuves
        disponibles : un score élevé basé sur une seule preuve est moins
        fiable que le même score basé sur plusieurs preuves convergentes.

        Args:
            evidence_score   : Score de soutien net (EvidenceEvaluation).
            related_evidence : Preuves liées à cette hypothèse (support+contra).

        Returns:
            Score de confiance entre 0.0 et 1.0.
        """
        if not related_evidence:
            # Aucune preuve → confiance réduite même si evidence_score=0.5 (neutre)
            return round(evidence_score * 0.5, 4)

        # Volume bonus : jusqu'à 3 preuves apportent un bonus de confiance croissant
        volume_factor = min(len(related_evidence) / 3.0, 1.0)
        confidence    = evidence_score * (0.7 + 0.3 * volume_factor)
        return round(max(0.0, min(1.0, confidence)), 4)

    def _compute_contradiction_penalty(
        self,
        hypothesis_id  : str,
        contradictions : list[tuple[str, str]],
        evidence       : list[Evidence],
    ) -> float:
        """Calcule la pénalité liée aux contradictions ciblant l'hypothèse.

        Compte les paires de contradictions (evidence_id_a, evidence_id_b)
        de EvidenceEvaluation où au moins une des deux preuves est liée
        à cette hypothèse.

        Args:
            hypothesis_id  : ID de l'hypothèse évaluée.
            contradictions : Paires contradictoires (EvidenceEvaluation).
            evidence       : Toutes les preuves (pour résoudre les IDs).

        Returns:
            Pénalité entre 0.0 et _MAX_CONTRADICTION_PENALTY.
        """
        if not contradictions:
            return 0.0

        evidence_by_id = {ev.evidence_id: ev for ev in evidence}
        count = 0

        for id_a, id_b in contradictions:
            ev_a = evidence_by_id.get(id_a)
            ev_b = evidence_by_id.get(id_b)
            relevant = False
            if ev_a and hypothesis_id in ev_a.relations:
                relevant = True
            if ev_b and hypothesis_id in ev_b.relations:
                relevant = True
            if relevant:
                count += 1

        penalty = min(count * _CONTRADICTION_PENALTY_STEP, _MAX_CONTRADICTION_PENALTY)
        return round(penalty, 4)

    def _compute_missing_info_penalty(
        self,
        hypothesis_id      : str,
        supporting_evidence : list[Evidence],
        knowledge_gaps      : list[str],
        hypothesis          : Hypothesis,
    ) -> float:
        """Calcule la pénalité liée aux informations manquantes.

        Deux sources de pénalité :
            1. Absence totale de preuve de soutien → pénalité fixe.
            2. Lacune de connaissance (knowledge_gaps) référençant le
               contenu de l'hypothèse → pénalité additionnelle légère.

        Args:
            hypothesis_id       : ID de l'hypothèse évaluée.
            supporting_evidence : Preuves SUPPORTS pour cette hypothèse.
            knowledge_gaps      : Lacunes identifiées par EvidenceValidator.
            hypothesis           : Hypothèse complète (pour matcher le contenu).

        Returns:
            Pénalité entre 0.0 et 1.0.
        """
        penalty = 0.0

        if not supporting_evidence:
            penalty += _MISSING_INFO_PENALTY_NO_SUPPORT

        # Recherche d'une lacune mentionnant un extrait du contenu de l'hypothèse
        content_snippet = hypothesis.content[:40].lower()
        for gap in knowledge_gaps:
            if content_snippet and content_snippet[:20] in gap.lower():
                penalty += 0.10
                break

        return round(min(penalty, 1.0), 4)

    def _compute_source_bonus(
        self,
        supporting_evidence : list[Evidence],
        source_type         : EvidenceSourceType,
        bonus_per_evidence  : float,
    ) -> float:
        """Calcule un bonus pour les preuves provenant d'une source donnée.

        Args:
            supporting_evidence : Preuves SUPPORTS pour cette hypothèse.
            source_type         : Type de source à bonifier (USER_MEMORY, NEURON_GRAPH).
            bonus_per_evidence  : Bonus accordé par preuve de cette source.

        Returns:
            Bonus cumulé, plafonné à _MAX_SOURCE_BONUS.
        """
        count = sum(1 for ev in supporting_evidence if ev.source_type == source_type)
        return round(min(count * bonus_per_evidence, _MAX_SOURCE_BONUS), 4)

    def _compute_global_score(
        self,
        evidence_score        : float,
        confidence              : float,
        contradiction_penalty   : float,
        missing_info_penalty    : float,
        memory_bonus             : float,
        graph_bonus              : float,
    ) -> float:
        """Agrège toutes les composantes en un score global unique.

        Formule :
            base   = evidence_score × _WEIGHT_EVIDENCE
                   + confidence × _WEIGHT_CONFIDENCE
                   + memory_bonus × _WEIGHT_MEMORY_BONUS
                   + graph_bonus × _WEIGHT_GRAPH_BONUS
            global = base - contradiction_penalty - missing_info_penalty

        Le résultat est clampé dans [0.0, 1.0] : les pénalités peuvent
        ramener un score élevé proche de 0, mais jamais en négatif.

        Args:
            evidence_score, confidence, contradiction_penalty,
            missing_info_penalty, memory_bonus, graph_bonus : composantes
            calculées par les méthodes ci-dessus.

        Returns:
            Score global entre 0.0 et 1.0.
        """
        base = (
            evidence_score * _WEIGHT_EVIDENCE
            + confidence    * _WEIGHT_CONFIDENCE
            + memory_bonus  * _WEIGHT_MEMORY_BONUS
            + graph_bonus   * _WEIGHT_GRAPH_BONUS
        )
        global_score = base - contradiction_penalty - missing_info_penalty
        return round(max(0.0, min(1.0, global_score)), 4)


# ── DecisionReasoner ───────────────────────────────────────────────────────────

@dataclass
class DecisionReasoner:
    """Construit la justification structurée (DecisionReason) d'une décision.

    Sépare la logique d'explication du calcul de score (DecisionScorer)
    pour que chacune reste testable et modifiable indépendamment.
    """

    def build_reason(
        self,
        candidates  : list[DecisionCandidate],
        selected_id : str,
        evaluation  : EvidenceEvaluation,
        ctx         : ReasoningContext,
    ) -> DecisionReason:
        """Construit la justification complète de la décision.

        Args:
            candidates  : Tous les candidats triés par score décroissant.
            selected_id : ID de l'hypothèse sélectionnée.
            evaluation  : Évaluation des preuves (Phase 3.3).
            ctx         : Contexte complet (pour accéder à l'analyse).

        Returns:
            DecisionReason avec toutes les sections renseignées.
        """
        selected = next(c for c in candidates if c.hypothesis_id == selected_id)
        rejected = [c for c in candidates if c.hypothesis_id != selected_id]

        selection_reasons = self._build_selection_reasons(selected, candidates)
        rejection_reasons = {
            c.hypothesis_id: self._build_rejection_reasons(c, selected)
            for c in rejected
        }
        missing_information = list(evaluation.knowledge_gaps)
        residual_risks       = self._build_residual_risks(selected, evaluation, ctx)

        return DecisionReason(
            selection_reasons    = selection_reasons,
            rejection_reasons    = rejection_reasons,
            missing_information  = missing_information,
            residual_risks       = residual_risks,
        )

    def _build_selection_reasons(
        self,
        selected   : DecisionCandidate,
        candidates : list[DecisionCandidate],
    ) -> list[str]:
        """Construit les raisons de sélection de l'hypothèse gagnante.

        Args:
            selected   : Candidat sélectionné.
            candidates : Tous les candidats (pour comparaison relative).

        Returns:
            Liste de phrases justificatives.
        """
        reasons: list[str] = []
        s = selected.score

        reasons.append(
            f"Score global le plus élevé parmi {len(candidates)} hypothèse(s) "
            f"évaluée(s) ({s.global_score:.2f})."
        )

        if s.evidence_score >= 0.65:
            reasons.append(
                f"Soutenue par un score de preuves favorable ({s.evidence_score:.2f})."
            )
        if s.contradiction_penalty == 0.0:
            reasons.append("Aucune contradiction détectée pour cette hypothèse.")
        if s.memory_bonus > 0.0:
            reasons.append(
                "Renforcée par des éléments de mémoire utilisateur pertinents."
            )
        if s.graph_bonus > 0.0:
            reasons.append(
                "Cohérente avec le graphe de connaissances existant."
            )
        if s.missing_information_penalty == 0.0:
            reasons.append("Aucune lacune de connaissance majeure identifiée.")

        return reasons

    def _build_rejection_reasons(
        self,
        candidate : DecisionCandidate,
        selected  : DecisionCandidate,
    ) -> list[str]:
        """Construit les raisons de rejet d'une hypothèse écartée.

        Args:
            candidate : Hypothèse rejetée.
            selected  : Hypothèse retenue (pour comparaison).

        Returns:
            Liste de phrases justificatives du rejet.
        """
        reasons: list[str] = []
        s = candidate.score

        gap = selected.score.global_score - s.global_score
        reasons.append(
            f"Score global inférieur à l'hypothèse retenue "
            f"({s.global_score:.2f} contre {selected.score.global_score:.2f})."
        )

        if s.contradiction_penalty > 0.0:
            reasons.append(
                f"Pénalisée par des contradictions détectées "
                f"(pénalité={s.contradiction_penalty:.2f})."
            )
        if s.missing_information_penalty > 0.0:
            reasons.append(
                f"Manque de preuves de soutien suffisantes "
                f"(pénalité={s.missing_information_penalty:.2f})."
            )
        if s.evidence_score < 0.45:
            reasons.append(
                f"Score de preuves insuffisant ({s.evidence_score:.2f})."
            )

        return reasons

    def _build_residual_risks(
        self,
        selected   : DecisionCandidate,
        evaluation : EvidenceEvaluation,
        ctx        : ReasoningContext,
    ) -> list[str]:
        """Identifie les risques résiduels malgré la décision prise.

        Args:
            selected   : Candidat sélectionné.
            evaluation : Évaluation des preuves.
            ctx        : Contexte complet (pour le niveau de risque de la question).

        Returns:
            Liste de phrases décrivant les risques résiduels.
        """
        risks: list[str] = []
        s = selected.score

        if s.global_score < _QUALITY_MEDIUM_THRESHOLD:
            risks.append(
                "Le score de la décision retenue reste modéré : "
                "une vérification humaine est recommandée."
            )
        if s.contradiction_penalty > 0.0:
            risks.append(
                "Des contradictions persistent sur l'hypothèse retenue malgré sa sélection."
            )
        if evaluation.overall_quality_score < 0.5:
            risks.append(
                f"La qualité globale des preuves collectées est limitée "
                f"({evaluation.overall_quality_score:.2f})."
            )

        analysis = ctx.analysis
        if analysis and analysis.risk_level in ("medium", "high"):
            risks.append(
                f"Le domaine de la question présente un niveau de risque "
                f"'{analysis.risk_level}' — la prudence reste recommandée."
            )

        return risks


# ── Confiance globale et qualité ─────────────────────────────────────────────

def _compute_overall_confidence(
    ctx                   : ReasoningContext,
    selected_score        : DecisionScore,
) -> float:
    """Calcule la confiance globale du pipeline de raisonnement complet.

    Intègre, comme demandé en Phase 3.4, l'ensemble des étapes :
        - analyse de la question (QuestionAnalysis.confidence)
        - qualité des hypothèses (volume disponible)
        - qualité des preuves (EvidenceEvaluation.overall_quality_score)
        - absence de contradictions (1 - contradiction_penalty normalisée)
        - score de la décision finale (DecisionScore.global_score)

    Args:
        ctx            : Contexte complet du pipeline.
        selected_score : Score détaillé de l'hypothèse retenue.

    Returns:
        Confiance globale entre 0.0 et 1.0.
    """
    analysis_confidence = ctx.analysis.confidence if ctx.analysis else 0.5
    hypotheses_factor    = min(len(ctx.hypotheses) / 3.0, 1.0)
    evidence_quality      = ctx.evidence_quality_score
    contradiction_factor  = max(
        0.0, 1.0 - (selected_score.contradiction_penalty / _MAX_CONTRADICTION_PENALTY)
    ) if _MAX_CONTRADICTION_PENALTY > 0 else 1.0
    decision_score         = selected_score.global_score

    overall = (
        analysis_confidence   * 0.15
        + hypotheses_factor    * 0.10
        + evidence_quality     * 0.20
        + contradiction_factor * 0.15
        + decision_score        * 0.40
    )
    return round(max(0.0, min(1.0, overall)), 4)


def _compute_decision_quality(overall_confidence: float) -> str:
    """Convertit la confiance globale en niveau qualitatif de décision.

    Seuils alignés sur les seuils de risk_level utilisés ailleurs dans
    le pipeline (response_confidence.py, reasoning_engine.py) pour
    rester cohérent à travers tout le système.

    Args:
        overall_confidence : Score de confiance global [0.0, 1.0].

    Returns:
        "high", "medium" ou "low".
    """
    if overall_confidence >= _QUALITY_HIGH_THRESHOLD:
        return "high"
    if overall_confidence >= _QUALITY_MEDIUM_THRESHOLD:
        return "medium"
    return "low"

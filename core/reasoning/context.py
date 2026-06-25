"""Shared state traversant le pipeline de raisonnement de MakenBrain.

ReasoningContext est un dataclass Python pur (non-Pydantic) pour éviter
la validation à chaque mutation intermédiaire. La validation Pydantic
n'intervient qu'aux frontières API (requête entrante, réponse sortante).

Règles :
- Chaque agent lit ce dont il a besoin et écrit son résultat.
- Aucun agent ne modifie les champs écrits par une étape précédente.
- context_version est en lecture seule après initialisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from models.reasoning import (
    Evidence,
    EvidenceEvaluation,
    Hypothesis,
    QuestionAnalysis,
    QuestionType,
    ReasoningRequest,
    ReasoningTrace,
)


@dataclass
class ReasoningContext:
    """Objet de shared state traversant le pipeline de raisonnement complet.

    Attributs:
        request         : Requête originale validée par Pydantic.
        user_id         : Identifiant Supabase de l'utilisateur authentifié.
        context_version : Version du contrat (ne jamais modifier en cours de pipeline).
        analysis        : Résultat de QuestionAnalyzer — étape 1.
        hypotheses      : Résultat de HypothesisEngine — étape 2.
        evidence        : Résultat de EvidenceCollector — étape 3.
        evaluation      : Résultat de EvidenceEvaluator — étape 4.
        final_answer    : Réponse synthétisée par le Synthesizer — étape 5.
        confidence      : Score de confiance calculé par le Synthesizer.
        evidence_quality_score : Score qualité des preuves (0.0 → 1.0).
        trace           : Méta-données d'exécution de chaque étape.
        pipeline_degraded : True si au moins une étape a échoué partiellement.
        errors          : Erreurs partielles collectées pendant le pipeline.
    """

    # ── Entrée (fournie à l'initialisation, immuable) ─────────────────────────
    request : ReasoningRequest
    user_id : str

    # ── Versioning du contrat — ne jamais modifier en cours de pipeline ───────
    context_version : str = "3.0"

    # ── Résultats des étapes (None jusqu'à exécution de l'étape) ─────────────
    analysis              : QuestionAnalysis | None = None
    hypotheses            : list[Hypothesis]        = field(default_factory=list)
    evidence              : list[Evidence]           = field(default_factory=list)
    evaluation            : EvidenceEvaluation | None = None
    final_answer          : str | None              = None
    confidence            : float                   = 0.0
    evidence_quality_score: float                   = 0.0

    # ── Trace d'exécution (initialisée par initialize_trace) ─────────────────
    trace : ReasoningTrace = field(
        default_factory=lambda: ReasoningTrace(question="")
    )

    # ── État de dégradation ───────────────────────────────────────────────────
    pipeline_degraded : bool                  = False
    errors            : list[dict[str, Any]]  = field(default_factory=list)

    # ── Méthodes d'initialisation et de gestion d'état ───────────────────────

    def initialize_trace(self) -> None:
        """Initialise la trace avec la question et les métadonnées de départ.

        Doit être appelé par reasoning_engine avant l'exécution du premier agent.
        Réinitialiser la trace en cours de pipeline est une erreur de logique.
        """
        self.trace = ReasoningTrace(
            trace_id   = uuid4().hex,
            created_at = datetime.now(),
            question   = self.request.question,
        )

    def mark_degraded(self, step_name: str, error: str) -> None:
        """Enregistre une dégradation partielle du pipeline.

        Le pipeline continue après cet appel avec les données partielles
        disponibles. La confiance finale reflète les lacunes.

        Args:
            step_name : Nom de l'étape qui a échoué.
            error     : Message d'erreur lisible.
        """
        self.pipeline_degraded = True
        self.errors.append({"step": step_name, "error": error})

    # ── Propriétés de commodité ───────────────────────────────────────────────

    @property
    def analysis_or_fallback(self) -> QuestionAnalysis:
        """Retourne l'analyse ou un fallback minimal si l'étape 1 a échoué.

        Garantit que les étapes suivantes peuvent toujours lire une analyse,
        même en mode dégradé.
        """
        if self.analysis is not None:
            return self.analysis
        return QuestionAnalysis(
            original_question  = self.request.question,
            question_type      = QuestionType.FACTUAL,
            complexity_score   = 0.5,
            sub_questions      = [self.request.question],
            key_entities       = [],
            key_concepts       = [],
            requires_reasoning = False,
        )

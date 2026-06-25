"""DTOs Pydantic pour le moteur de raisonnement expert de MakenBrain.

Phase 3.0 — Expert Reasoning Engine.

Règles strictes :
- Zéro logique métier dans ce module.
- Zéro import depuis core/ ou routers/.
- Ce module est la source de vérité des contrats de données Phase 3.0.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


# ── Énumérations ──────────────────────────────────────────────────────────────

class QuestionType(str, Enum):
    """Type sémantique de la question analysée."""

    FACTUAL      = "factual"       # "Qu'est-ce que X ?"
    ANALYTICAL   = "analytical"    # "Pourquoi X ?" / "Qu'est-ce qui cause Y ?"
    COMPARATIVE  = "comparative"   # "X vs Y ?" / "Différence entre X et Y ?"
    PROCEDURAL   = "procedural"    # "Comment faire X ?" / "Étapes pour Y ?"
    HYPOTHETICAL = "hypothetical"  # "Que se passerait-il si X ?"
    EVALUATIVE   = "evaluative"    # "Quelle est la meilleure approche pour X ?"


class EvidenceSourceType(str, Enum):
    """Origine d'une preuve collectée par l'EvidenceCollector."""

    VECTOR_MEMORY = "vector_memory"   # ChromaDB
    NEURON_GRAPH  = "neuron_graph"    # NetworkX
    USER_MEMORY   = "user_memory"     # Supabase mémoires utilisateur
    EXTERNAL      = "external"        # Futur : web, API tierces


class EvidenceRelation(str, Enum):
    """Relation d'une preuve vis-à-vis d'une hypothèse donnée."""

    SUPPORTS    = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL     = "neutral"


# ── Modèles intermédiaires (résultats d'étapes du pipeline) ───────────────────

class QuestionAnalysis(BaseModel):
    """Résultat de l'étape 1 — QuestionAnalyzer."""

    original_question  : str
    question_type      : QuestionType
    complexity_score   : float              = Field(ge=0.0, le=1.0)
    sub_questions      : list[str]          = Field(default_factory=list)
    key_entities       : list[str]          = Field(default_factory=list)
    key_concepts       : list[str]          = Field(default_factory=list)
    requires_reasoning : bool               = False


class Hypothesis(BaseModel):
    """Une hypothèse candidate générée par HypothesisEngine (étape 2)."""

    hypothesis_id : str   = Field(default_factory=lambda: uuid4().hex[:8])
    content       : str
    initial_score : float = Field(default=0.5, ge=0.0, le=1.0)
    origin        : str   = "prior_knowledge"


class Evidence(BaseModel):
    """Une preuve collectée depuis une source par EvidenceCollector (étape 3)."""

    evidence_id     : str                         = Field(default_factory=lambda: uuid4().hex[:8])
    content         : str
    source_type     : EvidenceSourceType
    source_ref      : str
    relevance_score : float                        = Field(default=0.5, ge=0.0, le=1.0)
    relations       : dict[str, EvidenceRelation]  = Field(default_factory=dict)


class EvidenceEvaluation(BaseModel):
    """Résultat de l'étape 4 — EvidenceEvaluator."""

    total_evidence_count  : int               = 0
    contradictions        : list[tuple[str, str]] = Field(default_factory=list)
    knowledge_gaps        : list[str]         = Field(default_factory=list)
    overall_quality_score : float             = Field(default=0.0, ge=0.0, le=1.0)
    best_hypothesis_id    : str | None        = None


class ReasoningStepRecord(BaseModel):
    """Enregistrement d'exécution d'une étape dans la trace de raisonnement."""

    step_name      : str
    duration_ms    : float
    success        : bool
    input_summary  : str
    output_summary : str
    degraded       : bool       = False
    error_message  : str | None = None


class ReasoningTrace(BaseModel):
    """Trace complète d'exécution du pipeline de raisonnement.

    Peuplée en continu par chaque étape. Exposée dans la réponse API
    uniquement si include_trace=True.
    """

    trace_id               : str      = Field(default_factory=lambda: uuid4().hex)
    created_at             : datetime = Field(default_factory=datetime.now)
    question               : str
    steps                  : list[ReasoningStepRecord] = Field(default_factory=list)
    total_duration_ms      : float    = 0.0
    evidence_count         : int      = 0
    hypotheses_count       : int      = 0
    evidence_quality_score : float    = 0.0


# ── Modèles de requête / réponse API ─────────────────────────────────────────

class ReasoningRequest(BaseModel):
    """Corps de la requête POST /reasoning/analyze."""

    question                : str   = Field(min_length=1)
    session_id              : str | None = None
    provider                : str   = "auto"
    max_hypotheses          : int   = Field(default=3, ge=1, le=5)
    max_evidence_per_source : int   = Field(default=5, ge=1, le=20)
    relevance_threshold     : float = Field(default=0.70, ge=0.0, le=1.0)
    include_trace           : bool  = False

    model_config = ConfigDict(json_schema_extra={"example": {
        "question"        : "Quelle est la meilleure architecture pour MakenBrain ?",
        "provider"        : "auto",
        "max_hypotheses"  : 3,
        "include_trace"   : False,
    }})


class ReasoningResponse(BaseModel):
    """Corps de la réponse de POST /reasoning/analyze."""

    answer                 : str
    confidence             : float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level             : str   = "high"
    suggested_sources      : list[dict[str, str]] = Field(default_factory=list)
    reasoning_summary      : str   = ""
    question_type          : QuestionType = QuestionType.FACTUAL
    complexity_score       : float = Field(default=0.0, ge=0.0, le=1.0)
    hypotheses_considered  : list[str]            = Field(default_factory=list)
    best_hypothesis        : str   = ""
    evidence_used          : int   = 0
    evidence_quality_score : float = Field(default=0.0, ge=0.0, le=1.0)
    knowledge_gaps         : list[str]            = Field(default_factory=list)
    provider               : str   = "unknown"
    model                  : str   = "unknown"
    trace                  : ReasoningTrace | None = None

"""Objets de communication inter-agents — Phase 5.

Toute communication entre l'Orchestrator et les agents passe par ces
dataclasses. Aucun agent ne doit jamais appeler directement un autre agent.

Hiérarchie :
    AgentTask       : La tâche à accomplir (input de l'Orchestrator).
    AgentResult     : Le résultat produit par un agent (output vers Orchestrator).
    ExecutionContext : Contexte partagé pendant l'exécution d'une tâche.
    ExecutionReport : Rapport final agrégé (output vers le client HTTP).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


class TaskType(str):
    """Types de tâches reconnus par le TaskRouter.

    Définis comme simples constantes string pour rester extensibles
    sans re-déploiement (les futurs agents peuvent déclarer leurs
    propres types via les capacités du BaseAgent).
    """
    REASONING   = "reasoning"
    MEMORY      = "memory"
    PLANNING    = "planning"
    CODING      = "coding"
    RESEARCH    = "research"
    GENERAL     = "general"


class TaskPriority:
    """Niveaux de priorité numériques (1=basse … 10=critique)."""
    LOW      = 1
    NORMAL   = 5
    HIGH     = 8
    CRITICAL = 10


@dataclass
class AgentTask:
    """Tâche à transmettre à un ou plusieurs agents via l'Orchestrator.

    Attributes:
        task_id    : Identifiant unique de la tâche (UUID généré automatiquement).
        type       : Type de tâche (TaskType.*). Utilisé par le TaskRouter.
        input      : Contenu principal de la tâche (question, code, texte…).
        user_id    : ID Supabase de l'utilisateur à l'origine de la tâche.
        session_id : ID de la session de conversation.
        priority   : Niveau de priorité (1–10, défaut = NORMAL).
        context    : Données contextuelles libres (mémoire RAG, historique…).
        metadata   : Métadonnées arbitraires (endpoint source, timing…).
    """
    type      : str
    input     : str
    task_id   : str                  = field(default_factory=lambda: str(uuid.uuid4()))
    user_id   : str | None           = None
    session_id: str | None           = None
    priority  : int                  = TaskPriority.NORMAL
    context   : dict[str, Any]       = field(default_factory=dict)
    metadata  : dict[str, Any]       = field(default_factory=dict)


@dataclass
class AgentResult:
    """Résultat produit par un agent et retourné à l'Orchestrator.

    Attributes:
        task_id     : Référence à l'AgentTask traité.
        agent_name  : Nom de l'agent qui a produit ce résultat.
        success     : True si l'agent a accompli la tâche sans erreur.
        output      : Contenu généré (texte, JSON…). None si success=False.
        error       : Message d'erreur. None si success=True.
        duration_ms : Durée d'exécution de l'agent en millisecondes.
        confidence  : Score de confiance 0.0–1.0. None si non applicable.
        metadata    : Métadonnées additionnelles (tokens, stratégie…).
    """
    task_id    : str
    agent_name : str
    success    : bool
    output     : str | None          = None
    error      : str | None          = None
    duration_ms: float               = 0.0
    confidence : float | None        = None
    metadata   : dict[str, Any]      = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "task_id"    : self.task_id,
            "agent_name" : self.agent_name,
            "success"    : self.success,
            "duration_ms": self.duration_ms,
        }
        if self.output     is not None: d["output"]     = self.output
        if self.error      is not None: d["error"]      = self.error
        if self.confidence is not None: d["confidence"] = self.confidence
        if self.metadata:               d["metadata"]   = self.metadata
        return d


@dataclass
class ExecutionContext:
    """Contexte mutable partagé pendant l'exécution d'une tâche.

    L'Orchestrator crée ce contexte, le passe à chaque agent sélectionné
    et y accumule les résultats au fur et à mesure.

    Attributes:
        request_id : ID de la requête HTTP parente (pour les logs).
        task       : La tâche en cours d'exécution.
        user_id    : ID utilisateur (copié depuis task pour commodité).
        session_id : ID de session (copié depuis task pour commodité).
        results    : Résultats accumulés par les agents déjà exécutés.
        shared     : Dict libre pour que les agents partagent des données
                     intermédiaires (ex. : mémoire récupérée par MemoryAgent
                     réutilisée par ReasoningAgent). Jamais de communication
                     directe entre agents — passe toujours ici.
    """
    request_id : str
    task       : AgentTask
    user_id    : str | None          = None
    session_id : str | None          = None
    results    : list[AgentResult]   = field(default_factory=list)
    shared     : dict[str, Any]      = field(default_factory=dict)

    def add_result(self, result: AgentResult) -> None:
        """Enregistre un résultat et propage les métadonnées si nécessaire."""
        self.results.append(result)

    @property
    def last_successful_output(self) -> str | None:
        """Retourne le dernier output d'un agent ayant réussi."""
        for r in reversed(self.results):
            if r.success and r.output:
                return r.output
        return None


@dataclass
class ExecutionReport:
    """Rapport final après exécution complète de la tâche.

    Produit par l'Orchestrator une fois tous les agents exécutés.
    C'est cet objet qui est sérialisé et renvoyé au client HTTP.

    Attributes:
        request_id      : ID de la requête HTTP parente.
        task_id         : ID de la tâche AgentTask.
        status          : "success" | "partial" | "failed".
        final_output    : Réponse finale agrégée. None si status="failed".
        agents_used     : Noms des agents qui ont participé.
        total_duration_ms: Durée totale de l'orchestration.
        results         : Tous les AgentResult individuels.
        metadata        : Métadonnées additionnelles de l'exécution.
    """
    request_id       : str
    task_id          : str
    status           : str                  # "success" | "partial" | "failed"
    final_output     : str | None
    agents_used      : list[str]
    total_duration_ms: float
    results          : list[AgentResult]    = field(default_factory=list)
    metadata         : dict[str, Any]       = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id"       : self.request_id,
            "task_id"          : self.task_id,
            "status"           : self.status,
            "final_output"     : self.final_output,
            "agents_used"      : self.agents_used,
            "total_duration_ms": self.total_duration_ms,
            "results"          : [r.to_dict() for r in self.results],
            **({"metadata": self.metadata} if self.metadata else {}),
        }

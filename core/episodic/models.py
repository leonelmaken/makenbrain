"""Modèles de la mémoire épisodique — Phase 6.

La mémoire épisodique enregistre ce que MakenBrain a fait, décidé et appris.
Contrairement à la mémoire vectorielle (ChromaDB), elle est chronologique
et structurée — chaque entrée représente un événement discret.

Types d'entrées :
    ACTION   : Une action exécutée par un agent (lecture/écriture mémoire, etc.)
    DECISION : Une décision prise lors d'un raisonnement (sélection hypothèse…)
    LEARNING : Un apprentissage ou une découverte nouvelle.
    PROJECT  : Un événement lié à un projet spécifique.
    ERROR    : Une erreur ou un échec notable.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class EpisodicType(StrEnum):
    ACTION   = "action"
    DECISION = "decision"
    LEARNING = "learning"
    PROJECT  = "project"
    ERROR    = "error"


@dataclass
class EpisodicEntry:
    """Une entrée dans la mémoire épisodique.

    Attributes:
        entry_id   : Identifiant unique (UUID4, généré automatiquement).
        timestamp  : Horodatage ISO de l'événement.
        type       : Catégorie de l'événement (EpisodicType).
        agent_name : Nom de l'agent qui a déclenché l'événement.
        content    : Description lisible de l'événement.
        task_id    : ID de l'AgentTask associé (None si événement hors-tâche).
        user_id    : ID Supabase de l'utilisateur concerné.
        metadata   : Données structurées additionnelles (libre).
    """
    type      : EpisodicType
    agent_name: str
    content   : str
    entry_id  : str                 = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp : str                 = field(default_factory=lambda: datetime.now().isoformat())
    task_id   : str | None          = None
    user_id   : str | None          = None
    metadata  : dict[str, Any]      = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "entry_id"  : self.entry_id,
            "timestamp" : self.timestamp,
            "type"      : self.type,
            "agent_name": self.agent_name,
            "content"   : self.content,
        }
        if self.task_id : d["task_id"]  = self.task_id
        if self.user_id : d["user_id"]  = self.user_id
        if self.metadata: d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodicEntry":
        return cls(
            entry_id  = data["entry_id"],
            timestamp = data["timestamp"],
            type      = EpisodicType(data["type"]),
            agent_name= data["agent_name"],
            content   = data["content"],
            task_id   = data.get("task_id"),
            user_id   = data.get("user_id"),
            metadata  = data.get("metadata", {}),
        )

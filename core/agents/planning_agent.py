"""PlanningAgent — décomposition de tâches complexes — Phase 6.

Reçoit une demande complexe et la découpe en sous-tâches ordonnées
avec dépendances. Ne les exécute PAS — retourne uniquement le plan.

Le plan est généré par LLM (via LLMRouter) et parsé en JSON structuré.
Si le LLM échoue ou retourne du JSON invalide, un plan minimal basé
sur des heuristiques est retourné (fallback déterministe).

Autonomie : SUGGEST — l'agent propose un plan, l'humain ou l'Orchestrator
décide d'activer son exécution.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from core.agents.base import AgentAutonomy, BaseAgent
from core.agents.models import AgentResult, AgentTask, ExecutionContext, TaskPriority, TaskType
from core.provider_layer.router import get_router


# ── Modèles du plan ───────────────────────────────────────────────────────────

@dataclass
class SubTask:
    """Sous-tâche issue de la décomposition d'une tâche complexe.

    Attributes:
        subtask_id  : Identifiant unique dans le plan (ex. "step_1").
        title       : Titre court de la sous-tâche.
        type        : Type de tâche (TaskType.*) pour le futur routage.
        description : Description détaillée de ce qui doit être fait.
        priority    : Priorité 1–10.
        dependencies: IDs des sous-tâches qui doivent être terminées avant.
    """
    subtask_id  : str
    title       : str
    type        : str               = TaskType.GENERAL
    description : str               = ""
    priority    : int               = TaskPriority.NORMAL
    dependencies: list[str]         = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id"  : self.subtask_id,
            "title"       : self.title,
            "type"        : self.type,
            "description" : self.description,
            "priority"    : self.priority,
            "dependencies": self.dependencies,
        }


@dataclass
class ExecutionPlan:
    """Plan d'exécution produit par PlanningAgent.

    Attributes:
        plan_id  : Identifiant unique du plan.
        goal     : Objectif principal reformulé par l'agent.
        subtasks : Liste ordonnée des sous-tâches.
        metadata : Métadonnées du plan (modèle LLM utilisé, stratégie…).
    """
    goal    : str
    subtasks: list[SubTask]         = field(default_factory=list)
    plan_id : str                   = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any]        = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id" : self.plan_id,
            "goal"    : self.goal,
            "subtasks": [s.to_dict() for s in self.subtasks],
            **({"metadata": self.metadata} if self.metadata else {}),
        }


# ── Agent ─────────────────────────────────────────────────────────────────────

_PLANNING_SYSTEM_PROMPT = """Tu es un planificateur expert. Tu reçois une demande complexe et tu la décomposes en sous-tâches claires, ordonnées et sans ambiguïté.

IMPORTANT : tu dois toujours répondre UNIQUEMENT avec un objet JSON valide, sans texte avant ni après.

Format attendu :
{
  "goal": "Objectif reformulé en une phrase claire",
  "subtasks": [
    {
      "subtask_id": "step_1",
      "title": "Titre court",
      "type": "reasoning|memory|research|planning|general",
      "description": "Description détaillée de ce qui doit être fait",
      "priority": 5,
      "dependencies": []
    }
  ]
}

Règles :
- Entre 2 et 8 sous-tâches.
- Les dépendances référencent des subtask_id définis dans le même plan.
- Les types valides sont : reasoning, memory, research, planning, general.
- La priorité est un entier entre 1 (basse) et 10 (critique).
- Réponds uniquement en JSON, sans markdown ni explication."""


class PlanningAgent(BaseAgent):
    """Agent de planification — décompose une demande en sous-tâches.

    Ne lance aucune exécution. Retourne un ExecutionPlan sérialisé en JSON
    dans AgentResult.output.

    task.context optionnel :
        max_subtasks (int) : Nombre maximum de sous-tâches (défaut 6).
        language (str)     : Langue des sous-tâches ("fr" ou "en").
    """

    name                : str           = "planning_agent"
    description         : str           = (
        "Agent de planification qui décompose une demande complexe en "
        "sous-tâches ordonnées avec dépendances. Ne les exécute pas."
    )
    capabilities        : list[str]     = ["planning", "decompose", "strategy", "general"]
    autonomy            : AgentAutonomy = AgentAutonomy.SUGGEST
    version             : str           = "1.0.0"

    cost_per_call       : float         = 0.5
    confidence_threshold: float         = 0.70

    async def run(self, task: AgentTask, ctx: ExecutionContext) -> AgentResult:
        """Génère un plan de décomposition via LLM.

        Tente d'abord une génération LLM. En cas d'échec, retourne
        un plan minimal déterministe (fallback).
        """
        t0 = time.monotonic()

        try:
            plan = await self._generate_plan(task)
        except Exception as exc:
            plan = self._fallback_plan(task, error=str(exc))

        ctx.shared["execution_plan"] = plan.to_dict()

        return self._timed_result(
            task, t0,
            success    = True,
            output     = json.dumps(plan.to_dict(), ensure_ascii=False),
            confidence = 0.85 if plan.metadata.get("strategy") == "llm" else 0.60,
            metadata   = {
                "plan_id"      : plan.plan_id,
                "subtask_count": len(plan.subtasks),
                "strategy"     : plan.metadata.get("strategy", "fallback"),
            },
        )

    async def _generate_plan(self, task: AgentTask) -> ExecutionPlan:
        """Appelle le LLM Router pour générer le plan, puis parse le JSON."""
        prompt = (
            f"Décompose cette demande en sous-tâches :\n\n{task.input}\n\n"
            f"Nombre maximum de sous-tâches : {task.context.get('max_subtasks', 6)}."
        )

        raw = await get_router().generate(
            prompt       = prompt,
            system_prompt= _PLANNING_SYSTEM_PROMPT,
        )

        plan = self._parse_llm_response(raw, task)
        plan.metadata["strategy"] = "llm"
        return plan

    def _parse_llm_response(self, raw: str, task: AgentTask) -> ExecutionPlan:
        """Parse la réponse JSON du LLM. Lève ValueError si invalide."""
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError("Aucun JSON trouvé dans la réponse LLM.")

        data = json.loads(json_match.group())

        subtasks = [
            SubTask(
                subtask_id  = s.get("subtask_id", f"step_{i+1}"),
                title       = s.get("title", f"Étape {i+1}"),
                type        = s.get("type", TaskType.GENERAL),
                description = s.get("description", ""),
                priority    = int(s.get("priority", TaskPriority.NORMAL)),
                dependencies= s.get("dependencies", []),
            )
            for i, s in enumerate(data.get("subtasks", []))
        ]

        return ExecutionPlan(
            goal    = data.get("goal", task.input[:100]),
            subtasks= subtasks,
        )

    def _fallback_plan(self, task: AgentTask, error: str = "") -> ExecutionPlan:
        """Plan minimal déterministe si le LLM est indisponible."""
        return ExecutionPlan(
            goal    = task.input[:200],
            subtasks= [
                SubTask(
                    subtask_id  = "step_1",
                    title       = "Analyser la demande",
                    type        = TaskType.REASONING,
                    description = f"Analyser et comprendre : {task.input[:100]}",
                    priority    = TaskPriority.HIGH,
                ),
                SubTask(
                    subtask_id  = "step_2",
                    title       = "Rechercher les informations nécessaires",
                    type        = TaskType.RESEARCH,
                    description = "Collecter les informations pertinentes.",
                    priority    = TaskPriority.NORMAL,
                    dependencies= ["step_1"],
                ),
                SubTask(
                    subtask_id  = "step_3",
                    title       = "Synthétiser et répondre",
                    type        = TaskType.REASONING,
                    description = "Produire une réponse complète.",
                    priority    = TaskPriority.NORMAL,
                    dependencies= ["step_1", "step_2"],
                ),
            ],
            metadata= {"strategy": "fallback", "fallback_reason": error},
        )

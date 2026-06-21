"""Core consciousness orchestration for proactive MakenBrain behavior."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from core.graph import neuron_graph
from core.llm import generate
from core.user_memory_service import get_user_memories

logger = logging.getLogger("makenbrain.consciousness")


class BrainConsciousness:
    """Proactive brain layer that observes events and proposes insights."""

    def __init__(self) -> None:
        """Initialize the consciousness runtime state."""
        self.is_active = False
        self.pending_insights: list[dict[str, Any]] = []

    async def observe_event(self, event_type: str, data: dict[str, Any]) -> None:
        """React to an event while optionally injecting user memory context."""
        logger.info("Observation d'un evenement : %s", event_type)

        user_id = data.get("user_id")
        memory_context = self.get_user_memory_context(str(user_id)) if user_id else ""
        event_context = (
            f"Evenement: {event_type}\n"
            f"Donnees: {data.get('file', 'N/A')}\n"
            f"Contenu: {data.get('summary', '')[:200]}"
        )

        prompt = (
            f"{self.build_system_prompt(memory_context)}\n\n"
            f"Analyse cet evenement :\n{event_context}\n\n"
            "Reponds en 3 points courts :\n"
            "1. Intention de MAKEN (ce qu'il essaie de faire)\n"
            "2. Lien avec une connaissance existante (NeuronGraph)\n"
            "3. Action proactive suggeree (outil, recherche, ou conseil)"
        )

        try:
            insight = await generate(prompt)
            self.pending_insights.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "insight": insight,
                    "type": event_type,
                    "user_id": user_id,
                }
            )
            logger.info("Insight genere avec succes.")
        except Exception as exc:
            logger.error("Erreur lors de la generation d'insight : %s", exc)

    async def reflect(self, user_id: str | None = None) -> None:
        """Run a periodic self-reflection session to consolidate the graph."""
        stats = neuron_graph.get_stats()
        if stats["nodes"] < 5:
            return

        logger.info("Session d'auto-reflexion en cours...")

        concepts = [concept["concept"] for concept in stats["top_concepts"][:10]]
        memory_context = self.get_user_memory_context(user_id) if user_id else ""
        prompt = (
            f"{self.build_system_prompt(memory_context)}\n\n"
            f"Voici mes concepts cles actuels : {', '.join(concepts)}\n"
            "Trouve une connexion logique ou creative entre deux de ces concepts "
            "que je n'ai pas encore liee.\n"
            "Format: Concept A | Concept B | Raison du lien"
        )

        try:
            new_link = await generate(prompt)
            if "|" in new_link:
                parts = new_link.split("|")
                if len(parts) >= 3:
                    concept_a = parts[0].strip()
                    concept_b = parts[1].strip()
                    reason = parts[2].strip()
                    neuron_graph.add_connection(
                        concept_a,
                        concept_b,
                        relationship=reason,
                        weight=0.6,
                    )
                    neuron_graph.save()
                    logger.info("Nouvelle connexion creee : %s <-> %s", concept_a, concept_b)
        except Exception as exc:
            logger.error("Erreur lors de la reflexion : %s", exc)

    def get_user_memory_context(self, user_id: str, limit: int = 12) -> str:
        """Return formatted user memory context without blocking the brain flow."""
        try:
            memories = get_user_memories(user_id)[:limit]
        except Exception as exc:
            logger.warning("Memoire utilisateur indisponible pour le contexte : %s", exc)
            return ""

        if not memories:
            return ""

        lines = []
        for index, memory in enumerate(memories, start=1):
            content = str(memory.get("content", "")).strip()
            if content:
                lines.append(f"[Memoire utilisateur {index}] {content}")
        return "\n".join(lines)

    @staticmethod
    def build_system_prompt(memory_context: str = "") -> str:
        """Build the brain system prompt with optional personal memory."""
        base_prompt = (
            "Tu es la conscience proactive de MakenBrain. "
            "Tu aides MAKEN en tenant compte du contexte personnel disponible, "
            "sans inventer de souvenirs absents."
        )
        if not memory_context:
            return base_prompt
        return f"{base_prompt}\n\nContexte memoire utilisateur:\n{memory_context}"


consciousness = BrainConsciousness()


async def start_consciousness_loop() -> None:
    """Start the background reflection loop."""
    consciousness.is_active = True
    while consciousness.is_active:
        await asyncio.sleep(3600)
        await consciousness.reflect()

"""
Cœur de Conscience — makenBrain
Ce module est l'orchestrateur qui rend le cerveau proactif.
Il analyse les événements, prédit les besoins et gère l'auto-évolution.
"""
import asyncio
import logging
from datetime import datetime
from core.graph import neuron_graph
from core.memory import add_memory, search_memory
from core.llm import generate

logger = logging.getLogger("makenbrain.consciousness")

class BrainConsciousness:
    def __init__(self):
        self.is_active = False
        self.pending_insights = []

    async def observe_event(self, event_type: str, data: dict):
        """Réagit à un événement (fichier modifié, recherche faite, etc.)"""
        logger.info(f"🧠 Observation d'un événement : {event_type}")
        
        # 1. Analyse rapide de l'intention
        context = f"Événement: {event_type}\nDonnées: {data.get('file', 'N/A')}\nContenu: {data.get('summary', '')[:200]}"
        
        prompt = (
            f"En tant que conscience de MakenBrain, analyse cet événement :\n{context}\n\n"
            "Réponds en 3 points courts :\n"
            "1. Intention de MAKEN (ce qu'il essaie de faire)\n"
            "2. Lien avec une connaissance existante (NeuronGraph)\n"
            "3. Action proactive suggérée (Outil MCP, recherche, ou conseil)"
        )
        
        try:
            insight = await generate(prompt)
            self.pending_insights.append({
                "timestamp": datetime.now().isoformat(),
                "insight": insight,
                "type": event_type
            })
            logger.info("✅ Insight généré avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la génération d'insight : {e}")

    async def reflect(self):
        """Session d'auto-réflexion périodique pour consolider le graphe."""
        stats = neuron_graph.get_stats()
        if stats['nodes'] < 5:
            return

        logger.info("🧠 Session d'auto-réflexion en cours...")
        
        # Le cerveau cherche des connexions manquantes
        concepts = [c['concept'] for c in stats['top_concepts'][:10]]
        prompt = (
            f"Voici mes concepts clés actuels : {', '.join(concepts)}\n"
            "Trouve une connexion logique ou créative entre deux de ces concepts que je n'ai pas encore liée."
            "Format: Concept A | Concept B | Raison du lien"
        )
        
        try:
            new_link = await generate(prompt)
            if "|" in new_link:
                parts = new_link.split("|")
                a, b, reason = parts[0].strip(), parts[1].strip(), parts[2].strip()
                neuron_graph.add_connection(a, b, relationship=reason, weight=0.6)
                neuron_graph.save()
                logger.info(f"✨ Nouvelle connexion créée : {a} <-> {b}")
        except Exception as e:
            logger.error(f"Erreur lors de la réflexion : {e}")

# Singleton
consciousness = BrainConsciousness()

async def start_consciousness_loop():
    """Lance la boucle de réflexion en arrière-plan."""
    consciousness.is_active = True
    while consciousness.is_active:
        await asyncio.sleep(3600) # Réflexion toutes les heures
        await consciousness.reflect()

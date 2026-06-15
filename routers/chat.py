"""
Chat avec raisonnement hybride :
  1. Mémoire vectorielle (ChromaDB) → souvenirs pertinents
  2. Graphe de neurones (NetworkX)  → concepts connectés
  3. LLM (local ou Groq 70B)        → réponse enrichie

C'est le vrai cerveau en action : mémoire + graphe + langage.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from core.llm import generate, check_ollama_status
from core.personality import personality_engine
from core.memory import search_memory
from core.providers import groq_generate, detect_provider
from core.config import settings

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    use_memory: bool = True
    use_graph: bool = True          # Nouveau : raisonnement graphe activé
    n_context: int = 5
    relevance_threshold: float = 0.75
    provider: str = "auto"

    model_config = {"json_schema_extra": {
        "example": {
            "message": "Comment fonctionne l'authentification JWT dans SmartBudget?",
            "use_memory": True,
            "use_graph": True,
            "n_context": 5,
            "provider": "auto"
        }
    }}


class ChatResponse(BaseModel):
    response: str
    memories_used: int
    graph_concepts: list[str]
    model: str
    provider: str
    context_preview: Optional[list[str]] = None


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat hybride Mémoire + Graphe + LLM.

    Fonctionnement :
      1. Cherche les souvenirs proches dans ChromaDB (similarité vectorielle)
      2. Extrait les concepts de la question et explore le graphe
      3. Fusionne les deux contextes
      4. Envoie au LLM choisi (local ou Groq selon provider)
    """
    context_parts = []
    memories_used = 0
    graph_concepts: list[str] = []
    context_preview = []

    # ── 1. Mémoire vectorielle ────────────────────────────────────────────────
    if request.use_memory:
        memories = await search_memory(request.message, n_results=request.n_context)
        relevant = [m for m in memories if m["distance"] < request.relevance_threshold]
        if relevant:
            for i, m in enumerate(relevant):
                context_parts.append(f"[Souvenir {i+1}] {m['content']}")
                preview = m["content"][:100] + "..." if len(m["content"]) > 100 else m["content"]
                context_preview.append(preview)
            memories_used = len(relevant)

    # ── 2. Graphe de neurones ─────────────────────────────────────────────────
    if request.use_graph:
        try:
            from core.graph import neuron_graph
            from core.extractor import extract_fast

            # Extraire les concepts clés de la question
            concepts_in_question = extract_fast(request.message)
            graph_insights = []

            for c in concepts_in_question[:3]:    # Top 3 concepts de la question
                name = c.get("name", "").lower()
                if not name or len(name) < 3:
                    continue

                result = neuron_graph.explore(name, depth=2)
                if not result.get("found"):
                    continue

                # Récupérer les voisins directs les plus forts
                neighbors = result.get("neighbors", {})
                direct = [
                    n for n, info in neighbors.items()
                    if info["distance"] == 1 and info["strength"] > 0.5
                ][:5]

                if direct:
                    graph_insights.append(
                        f"[Graphe] '{name}' est connecté à : {', '.join(direct)}"
                    )
                    graph_concepts.extend(direct)

            if graph_insights:
                context_parts.append("\n".join(graph_insights))

        except Exception:
            pass    # Graphe non disponible → continue sans erreur

    # ── 3. Assemblage du contexte ─────────────────────────────────────────────
    context = "\n\n".join(context_parts)

    # ── 4. Choix du provider et génération ────────────────────────────────────
    chosen = detect_provider(request.message, request.provider)

    # --- ADAPTATION PERSONNALITE ---
    style = personality_engine.detect_style(request.message)
    dynamic_prompt = personality_engine.get_system_prompt(style)
    # -------------------------------

    if chosen == "groq":
        response_text = await groq_generate(request.message, context, system_prompt=dynamic_prompt)
        model_name = "llama-3.3-70b-versatile (Groq)"
    else:
        response_text = await generate(request.message, context, system_prompt=dynamic_prompt)
        model_name = settings.OLLAMA_MODEL + " (local)"

    return ChatResponse(
        response=response_text,
        memories_used=memories_used,
        graph_concepts=list(set(graph_concepts))[:10],
        model=model_name,
        provider=chosen,
        context_preview=context_preview if context_preview else None,
    )


@router.get("/status")
async def status():
    """Statut du LLM local Ollama."""
    return await check_ollama_status()


@router.get("/welcome")
async def welcome():
    """Message d'accueil personnalisé de MakenBrain."""
    return {"message": personality_engine.get_greeting()}
"""
Router Graphe de Neurones — Phase 3
Prefix /graph ajouté dans main.py.
"""
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from core.graph import neuron_graph
from core.extractor import extract_and_graph

router = APIRouter()


class ExploreRequest(BaseModel):
    concept: str
    depth:   int = 2

class PathRequest(BaseModel):
    from_concept: str
    to_concept:   str

class ConnectRequest(BaseModel):
    concept_a:    str
    concept_b:    str
    relationship: str   = "related"
    weight:       float = 0.8

class EnrichRequest(BaseModel):
    text:      str
    memory_id: str  = ""
    fast:      bool = True

class BuildRequest(BaseModel):
    max_memories: int  = 500
    fast:         bool = True

class SearchRequest(BaseModel):
    query: str
    limit: int = 10


@router.get("/stats")
async def graph_stats():
    """Statistiques globales du graphe."""
    return neuron_graph.get_stats()


@router.post("/explore")
async def explore(req: ExploreRequest):
    """Explore le voisinage d'un concept (depth=1 direct, depth=2 indirect)."""
    return neuron_graph.explore(req.concept, req.depth)


@router.post("/path")
async def find_path(req: PathRequest):
    """Chemin le plus court entre deux concepts — raisonnement multi-sauts."""
    return neuron_graph.find_path(req.from_concept, req.to_concept)


@router.post("/search")
async def search(req: SearchRequest):
    """Cherche des concepts dans le graphe par mot-clé."""
    return {"query": req.query, "results": neuron_graph.search_concepts(req.query, req.limit)}


@router.get("/map")
async def get_map(max_nodes: int = 80):
    """Données JSON pour la visualisation D3.js → /static/brain_map.html"""
    return neuron_graph.get_map(max_nodes)


@router.post("/connect")
async def connect(req: ConnectRequest):
    """Ajoute manuellement une connexion entre deux concepts."""
    neuron_graph.add_concept(req.concept_a)
    neuron_graph.add_concept(req.concept_b)
    neuron_graph.add_connection(req.concept_a, req.concept_b, req.relationship, req.weight)
    neuron_graph.save()
    return {"connected": True,
            "edge": f"{req.concept_a.lower()} ←→ {req.concept_b.lower()}",
            "graph_edges": neuron_graph.G.number_of_edges()}


@router.post("/enrich")
async def enrich(req: EnrichRequest):
    """Extrait les concepts d'un texte et enrichit le graphe."""
    return await extract_and_graph(req.text, req.memory_id, req.fast)


@router.post("/build")
async def build(req: BuildRequest, background_tasks: BackgroundTasks):
    """⚡ Construit le graphe depuis toutes les mémoires ChromaDB (arrière-plan)."""
    background_tasks.add_task(_build_task, req.max_memories, req.fast)
    return {"status": "démarré en arrière-plan",
            "message": f"Analyse de {req.max_memories} mémoires.",
            "next_step": "Attends 30s puis GET /graph/stats"}


@router.delete("/reset")
async def reset():
    """Remet le graphe à zéro."""
    import networkx as nx
    neuron_graph.G.clear()
    neuron_graph.save()
    return {"reset": True, "message": "Lance POST /graph/build pour reconstruire."}


async def _build_task(max_memories: int, fast: bool):
    try:
        import chromadb
        from core.config import settings
        client     = chromadb.PersistentClient(path=str(settings.CHROMA_PATH))
        collection = client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        all_data  = collection.get(include=["documents", "metadatas"], limit=max_memories)
        ids, docs = all_data.get("ids", []), all_data.get("documents", [])
        processed = 0
        for i, doc in enumerate(docs):
            if doc and len(doc) > 15:
                await extract_and_graph(doc, ids[i] if i < len(ids) else "", fast=fast)
                processed += 1
        neuron_graph.save()
        print(f"✅ Graphe : {processed} mémoires → "
              f"{neuron_graph.G.number_of_nodes()} nœuds, "
              f"{neuron_graph.G.number_of_edges()} arêtes")
    except Exception as e:
        import traceback
        print(f"❌ Erreur build graphe : {e}")
        traceback.print_exc()

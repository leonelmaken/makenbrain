"""
Router Graphe de Neurones — Phase 3
─────────────────────────────────────
IMPORTANT : ce router n'a PAS de prefix ici.
Le prefix "/graph" est ajouté dans main.py via :
    app.include_router(graph.router, prefix="/graph")

Règle de bonne pratique FastAPI :
  → le router définit SES PROPRES endpoints (les chemins relatifs)
  → main.py décide du chemin absolu (le prefix)
  Exemple : cet endpoint est "/explore" → accessible en "/graph/explore"

Les endpoints disponibles :
  GET  /graph/stats        → statistiques globales du graphe
  POST /graph/explore      → explore les connexions d'un concept
  POST /graph/path         → chemin le plus court entre 2 concepts
  POST /graph/search       → recherche de concepts par mot-clé
  GET  /graph/map          → données pour la visualisation D3.js
  POST /graph/connect      → ajoute manuellement une connexion
  POST /graph/enrich       → extrait et ingère les concepts d'un texte
  POST /graph/build        → ⚡ construit le graphe depuis toutes les mémoires
  DELETE /graph/reset      → remet le graphe à zéro
"""

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from core.graph     import neuron_graph
from core.extractor import extract_and_graph

# ── Pas de prefix ici → c'est main.py qui le gère ───────────────────────────
router = APIRouter()


# ── Schémas Pydantic ──────────────────────────────────────────────────────────
# Pydantic valide automatiquement les données entrantes.
# Si un champ obligatoire manque ou a le mauvais type → 422 Unprocessable Entity.

class ExploreRequest(BaseModel):
    concept: str                    # ex: "spring boot"
    depth: int = 2                  # 1 = voisins directs, 2 = voisins de voisins


class PathRequest(BaseModel):
    from_concept: str               # ex: "jwt"
    to_concept:   str               # ex: "mobile money"


class ConnectRequest(BaseModel):
    concept_a:    str               # premier concept
    concept_b:    str               # deuxième concept
    relationship: str = "related"   # type de lien
    weight:       float = 0.8       # force du lien (0.0 → 1.0)


class EnrichRequest(BaseModel):
    text:      str                  # texte à analyser
    memory_id: str = ""             # ID mémoire liée (optionnel)
    fast:      bool = True          # True = patterns, False = LLM


class BuildRequest(BaseModel):
    max_memories: int = 500         # nombre max de mémoires à lire
    fast:         bool = True       # True recommandé pour le premier build


class SearchRequest(BaseModel):
    query: str
    limit: int = 10


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/stats")
async def graph_stats():
    """
    Statistiques globales du graphe.
    Lance ça après /graph/build pour voir l'état du cerveau.

    Retourne :
      nodes       → nombre de concepts connus
      edges       → nombre de connexions
      density     → ratio connexions / connexions possibles (0 = isolé, 1 = tout connecté)
      components  → îles séparées dans le graphe (idéalement 1 = tout est relié)
      top_concepts → concepts les plus centraux (les plus connectés)
    """
    return neuron_graph.get_stats()


@router.post("/explore")
async def explore_concept(req: ExploreRequest):
    """
    Explore le voisinage d'un concept.

    depth=1 → voisins directs (connexion directe)
    depth=2 → voisins de voisins (connexion indirecte via 1 intermédiaire)

    Exemple : explore("spring boot", depth=2)
    → trouve "java", "postgresql", "jwt", "smartbudget africa", "backend", etc.
    """
    return neuron_graph.explore(req.concept, req.depth)


@router.post("/path")
async def find_path(req: PathRequest):
    """
    Raisonnement multi-sauts : quel est le chemin entre deux concepts ?

    Exemple : path("jwt", "mobile money")
    → jwt → authentification → smartbudget africa → mobile money
    → révèle les concepts intermédiaires que le cerveau a appris
    """
    return neuron_graph.find_path(req.from_concept, req.to_concept)


@router.post("/search")
async def search_concepts(req: SearchRequest):
    """Cherche des concepts dans le graphe par mot-clé partiel."""
    return {
        "query":   req.query,
        "results": neuron_graph.search_concepts(req.query, req.limit),
    }


@router.get("/map")
async def get_map(max_nodes: int = 80):
    """
    Données JSON pour la visualisation D3.js.

    Après le /graph/build, ouvre static/brain_map.html dans ton navigateur.
    La page se connecte à cet endpoint et affiche le graphe de neurones animé.
    """
    return neuron_graph.get_map(max_nodes)


@router.post("/connect")
async def add_connection(req: ConnectRequest):
    """
    Ajoute manuellement une connexion entre deux concepts.
    Utile pour enseigner au cerveau des relations qu'il n'a pas découvertes seul.

    Exemple : connect("mtn momo", "mobile money", relationship="est-un-type-de")
    """
    neuron_graph.add_concept(req.concept_a)
    neuron_graph.add_concept(req.concept_b)
    neuron_graph.add_connection(
        req.concept_a, req.concept_b, req.relationship, req.weight
    )
    neuron_graph.save()
    return {
        "connected":   True,
        "edge":        f"{req.concept_a.lower()} ←→ {req.concept_b.lower()}",
        "relationship": req.relationship,
        "graph_edges": neuron_graph.G.number_of_edges(),
    }


@router.post("/enrich")
async def enrich_from_text(req: EnrichRequest):
    """
    Extrait les concepts d'un texte libre et enrichit le graphe.
    Utile pour nourrir le cerveau avec des notes, recherches, articles.

    fast=True  → extraction par patterns (rapide, pas de LLM)
    fast=False → extraction par LLM Groq (précise, 2-3 secondes)
    """
    return await extract_and_graph(req.text, req.memory_id, req.fast)


@router.post("/build")
async def build_from_memory(req: BuildRequest, background_tasks: BackgroundTasks):
    """
    ⚡ COMMANDE PRINCIPALE — Lance cette commande une seule fois
    après avoir ingéré tes fichiers (Phase 2).

    Elle lit toutes les mémoires ChromaDB, extrait les concepts de chacune,
    et construit le graphe complet en arrière-plan (non-bloquant).

    Après 30 secondes → GET /graph/stats pour voir le résultat.
    """
    # BackgroundTasks = FastAPI exécute _build_task sans bloquer la réponse HTTP
    background_tasks.add_task(_build_task, req.max_memories, req.fast)
    return {
        "status":    "démarré en arrière-plan ⚙️",
        "message":   f"Analyse de {req.max_memories} mémoires → construction du graphe.",
        "next_step": "Attends 30 secondes puis GET /graph/stats",
        "fast_mode": req.fast,
    }


@router.delete("/reset")
async def reset_graph():
    """
    Remet le graphe à zéro.
    À utiliser si tu veux reconstruire proprement depuis 0.
    Suivi obligatoire d'un /graph/build pour repeupler.
    """
    import networkx as nx
    neuron_graph.G.clear()
    neuron_graph.save()
    return {
        "reset":    True,
        "message":  "Graphe vidé. Lance POST /graph/build pour reconstruire.",
    }


# ── Tâche de fond ─────────────────────────────────────────────────────────────

async def _build_task(max_memories: int, fast: bool):
    """
    Construit le graphe en lisant directement ChromaDB.
    S'exécute en arrière-plan via BackgroundTasks FastAPI.

    Note : on utilise .get() de ChromaDB (pas .query()) car on veut
    TOUTES les mémoires, pas celles qui ressemblent à une requête.
    """
    try:
        import chromadb
        from core.config import settings

        # Se connecter à ChromaDB (même chemin que core/memory.py)
        client     = chromadb.PersistentClient(path=str(settings.CHROMA_PATH))
        collection = client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # .get() sans filtre = récupère TOUS les documents stockés
        all_data = collection.get(
            include=["documents", "metadatas"],
            limit=max_memories,
        )

        ids  = all_data.get("ids", [])
        docs = all_data.get("documents", [])

        processed = 0
        for i, doc in enumerate(docs):
            if doc and len(doc) > 15:  # ignorer les fragments trop courts
                mem_id = ids[i] if i < len(ids) else ""
                await extract_and_graph(doc, mem_id, fast=fast)
                processed += 1

        # Sauvegarde finale dans brain_data/neuron_graph.json
        neuron_graph.save()
        print(
            f"✅ Graphe construit : {processed} mémoires traitées → "
            f"{neuron_graph.G.number_of_nodes()} nœuds, "
            f"{neuron_graph.G.number_of_edges()} arêtes"
        )

    except Exception as e:
        print(f"❌ Erreur construction graphe : {e}")
        import traceback
        traceback.print_exc()
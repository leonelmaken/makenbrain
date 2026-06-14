"""
MakenBrain — v0.4.0 — Phase 3 : Graphe de Neurones
─────────────────────────────────────────────────────
Ce fichier est le point d'entrée de l'application FastAPI.
Il orchestre :
  - le démarrage / arrêt du cerveau (lifespan)
  - l'assemblage de tous les routers (= modules d'API)
  - les middlewares globaux (CORS, etc.)

Architecture FastAPI :
  main.py           → chef d'orchestre
  routers/xyz.py    → un fichier par domaine fonctionnel
  core/xyz.py       → logique métier (pas d'HTTP ici)
  models/xyz.py     → schémas Pydantic (validation des données)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── Imports du cerveau ────────────────────────────────────────────────────────
from core.memory import init_memory   # initialise ChromaDB
from core.graph  import neuron_graph  # graphe de neurones (Phase 3)


# ── Cycle de vie de l'application ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan = code qui tourne au démarrage ET à l'arrêt.
    Le 'yield' sépare les deux :
      - Avant yield → démarrage (init des ressources)
      - Après yield  → arrêt (libération des ressources)
    """
    # ─ DÉMARRAGE ─
    print("🧠 MakenBrain s'éveille...")
    await init_memory()
    print("✅ Mémoire vectorielle initialisée.")
    print(
        f"✅ Graphe de neurones : "
        f"{neuron_graph.G.number_of_nodes()} nœuds, "
        f"{neuron_graph.G.number_of_edges()} arêtes"
    )
    print("🌐 Swagger : http://localhost:8000/docs\n")

    yield  # ← l'app tourne ici

    # ─ ARRÊT ─
    print("💤 MakenBrain s'endort.")
    # Arrêter le watcher de fichiers s'il tourne
    try:
        from core.watcher import stop_watcher
        stop_watcher()
    except Exception:
        pass  # pas grave si le watcher n'était pas actif


# ── Application FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title="MakenBrain",
    description="🧠 Cerveau numérique personnel de MAKEN — v0.4.0",
    version="0.4.0",
    lifespan=lifespan,
)

# CORS = Cross-Origin Resource Sharing
# Permet à n'importe quelle origine (ton navigateur, Postman, etc.)
# d'appeler l'API. En production on restreindrait les origines.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # toutes les origines autorisées
    allow_credentials=True,
    allow_methods=["*"],       # GET, POST, DELETE, etc.
    allow_headers=["*"],
)

# Servir les fichiers statiques (visualisation du graphe)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Routers ───────────────────────────────────────────────────────────────────
# Règle : le PREFIX est toujours défini ICI (dans main.py), pas dans le fichier
# router. Cela donne une vue centrale de toutes les routes du cerveau.
#
# Convention : les tags définissent les sections dans Swagger /docs
from routers import chat, memory, ingest, files, search, providers, graph

app.include_router(chat.router,      prefix="/chat",      tags=["💬 Chat"])
app.include_router(memory.router,    prefix="/memory",    tags=["💾 Mémoire"])
app.include_router(ingest.router,    prefix="/ingest",    tags=["📥 Ingestion"])
app.include_router(files.router,     prefix="/files",     tags=["📁 Fichiers & Watcher"])
app.include_router(search.router,    prefix="/search",    tags=["🌐 Recherche Web"])
app.include_router(providers.router, prefix="/providers", tags=["🔌 Connecteurs IA"])
app.include_router(graph.router,     prefix="/graph",     tags=["🧠 Graphe de Neurones"])  # ← NOUVEAU


# ── Route racine ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Status"])
def root():
    """
    Endpoint de santé : retourne l'état du cerveau.
    GET http://localhost:8000/ → JSON de statut.
    """
    stats = neuron_graph.get_stats()
    return {
        "status": "alive",
        "brain": "MakenBrain",
        "version": "0.4.0",
        "owner": "MAKEN",
        "graph": stats,          # ← état du graphe de neurones
        "intelligence": {
            "local":     "llama3.2:3b (toujours disponible, hors-ligne)",
            "cloud":     "llama-3.3-70b via Groq (questions complexes)",
            "knowledge": "DuckDuckGo + tes fichiers locaux + graphe de concepts",
        },
        "capabilities": [
            "Chat RAG avec mémoire vectorielle",
            "Routage intelligent auto local ↔ Groq 70B",
            "Ingestion PDF + dossiers + HTML",
            "Déduplication SHA256",
            "Résumé automatique",
            "Surveillance dossiers en temps réel (Watcher)",
            "Recherche web DuckDuckGo autonome",
            "Graphe de neurones — connexions automatiques entre concepts",
            "Raisonnement multi-sauts A → B → C",
        ],
        # Pour voir la visualisation graphe :
        "brain_map": "http://localhost:8000/static/brain_map.html",
    }
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.memory import init_memory
from routers import chat, memory, ingest, files, search, providers


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🧠 MakenBrain s'éveille...")
    await init_memory()
    print("✅ Mémoire vectorielle initialisée.")
    yield
    print("💤 MakenBrain s'endort.")
    from core.watcher import stop_watcher
    stop_watcher()


app = FastAPI(
    title="MakenBrain",
    description="🧠 Cerveau numérique personnel de MAKEN — v0.3.0",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router,      prefix="/chat",      tags=["💬 Chat"])
app.include_router(memory.router,    prefix="/memory",    tags=["💾 Mémoire"])
app.include_router(ingest.router,    prefix="/ingest",    tags=["📥 Ingestion"])
app.include_router(files.router,     prefix="/files",     tags=["📁 Fichiers & Watcher"])
app.include_router(search.router,    prefix="/search",    tags=["🌐 Recherche Web"])
app.include_router(providers.router, prefix="/providers", tags=["🔌 Connecteurs IA"])


@app.get("/", tags=["Status"])
def root():
    return {
        "status": "alive",
        "brain": "MakenBrain",
        "version": "0.3.0",
        "owner": "MAKEN",
        "intelligence": {
            "local":     "llama3.2:3b (toujours disponible)",
            "cloud":     "llama-3.3-70b via Groq (questions complexes)",
            "knowledge": "Wikipedia + DuckDuckGo + tes fichiers locaux",
        },
        "capabilities": [
            "Chat RAG avec mémoire locale",
            "Routage intelligent auto local ↔ Groq 70B",
            "Ingestion PDF + dossiers + HTML",
            "Déduplication SHA256",
            "Résumé automatique",
            "Surveillance temps réel (Watcher)",
            "Recherche web DuckDuckGo",
            "Recherche multi-angles autonome",
            "Wikipedia FR/EN auto-ingéré",
            "Groq Llama 3.3 70B connecté",
        ],
    }

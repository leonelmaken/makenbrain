from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.memory import init_memory
from routers import chat, memory, ingest


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise les ressources au démarrage du cerveau."""
    print("🧠 MakenBrain s'éveille...")
    await init_memory()
    print("✅ Mémoire vectorielle initialisée.")
    yield
    print("💤 MakenBrain s'endort.")


app = FastAPI(
    title="MakenBrain",
    description="🧠 Cerveau numérique personnel de MAKEN — Phase 1: Foundation",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/chat", tags=["💬 Chat"])
app.include_router(memory.router, prefix="/memory", tags=["💾 Mémoire"])
app.include_router(ingest.router, prefix="/ingest", tags=["📥 Ingestion"])


@app.get("/", tags=["Status"])
def root():
    return {
        "status": "alive",
        "brain": "MakenBrain",
        "version": "0.1.0",
        "phase": 1,
        "owner": "MAKEN",
    }

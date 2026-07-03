import logging
import sys
import time
from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from routers import chat_history
from core.auth import SECURE
from core.config import settings
from core.memory import init_memory
from core.middleware import RequestContextMiddleware
from core.observability import configure_json_logging
from core.version import APP_NAME, APP_RELEASE_NAME, APP_VERSION
from routers import chat, memory, ingest, files, search, providers, agent, analysis, brain, video, identity, audit, users
from routers import reasoning
from routers import health as health_router
from routers import agents as agents_router
from routers import config as config_router
# Force l'encodage UTF-8 sur stdout/stderr, quel que soit le code page actif
# de la console Windows (cp1252 par défaut en environnement francophone).
# Sans ça, le moindre print() contenant un emoji (utilisés dans tout le
# projet pour les logs) lève UnicodeEncodeError et fait planter le
# lifespan avant même que le serveur ait pu démarrer.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Activer les logs JSON structurés si configuré dans .env (JSON_LOGS=true)
if settings.JSON_LOGS:
    configure_json_logging(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

try:
    from routers import graph
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie de l'application : initialisation au boot, nettoyage à l'arrêt.

    Logs préfixés [STARTUP]/[SHUTDOWN] avec timing par étape -- volontairement
    sans emoji sur ce chemin précis (la protection UTF-8 ci-dessus couvre déjà
    tout le reste du projet, mais le boot est l'endroit où une régression
    d'encodage coûte le plus cher : elle empêche le serveur de démarrer).
    """
    boot_start = time.monotonic()
    print("[STARTUP] MakenBrain s'éveille...")

    step_start = time.monotonic()
    await init_memory()
    print(f"[STARTUP] Mémoire vectorielle prête ({time.monotonic() - step_start:.1f}s).")

    # Phase 5 — Enregistrement des agents dans le registry
    from core.agents.registry import register_defaults
    register_defaults()
    print("[STARTUP] Agents Phase 5 enregistrés.")

    # Démarrer le scheduler si activé
    from core.scheduler import load_config, start_scheduler
    cfg = load_config()
    if cfg.get("enabled"):
        loop = asyncio.get_event_loop()
        start_scheduler(loop)
        print("[STARTUP] Scheduler autonome démarré.")
    else:
        print("[STARTUP] Scheduler désactivé (rien à démarrer).")

    print(f"[STARTUP] MakenBrain opérationnel en {time.monotonic() - boot_start:.1f}s.")

    yield

    print("[SHUTDOWN] MakenBrain s'endort.")
    from core.watcher import stop_watcher
    from core.scheduler import stop_scheduler
    stop_watcher()
    stop_scheduler()


app = FastAPI(
    title=APP_NAME,
    description=f"🧠 Cerveau numérique personnel de MAKEN — v{APP_VERSION} ({APP_RELEASE_NAME})",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestContextMiddleware)

static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Monter les outputs
outputs_dir = Path("brain_data/outputs")
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/outputs", StaticFiles(directory="brain_data/outputs"), name="outputs")

app.include_router(chat_history.router, prefix="/history", tags=["📜 Historique"])
app.include_router(identity.router, prefix="/identity", tags=["👤 Identité & Projets"])
app.include_router(chat.router,     prefix="/chat",     tags=["💬 Chat"])
app.include_router(memory.router,   prefix="/memory",   tags=["💾 Mémoire"])
app.include_router(ingest.router,   prefix="/ingest",   tags=["📥 Ingestion"])
app.include_router(files.router,    prefix="/files",    tags=["📁 Fichiers & Watcher"])
app.include_router(search.router,   prefix="/search",   tags=["🌐 Recherche Web"])
app.include_router(providers.router,prefix="/providers",tags=["🔌 Connecteurs IA"])
app.include_router(agent.router,    prefix="/agent",    tags=["🤖 Agent Fichiers"], dependencies=SECURE)
app.include_router(analysis.router, prefix="/analysis", tags=["🔍 Analyse & Validation"])
app.include_router(brain.router,    prefix="/brain",    tags=["🧠 Expertise & Autonomie"])
app.include_router(video.router,    prefix="/video",    tags=["🎬 Génération Vidéo"])
app.include_router(reasoning.router, prefix="/reasoning", tags=["🧠 Raisonnement Expert"])

app.include_router(users.router, tags=["Users"])
app.include_router(audit.router,    prefix="/audit",    tags=["Audit"], dependencies=SECURE)
app.include_router(health_router.router)
app.include_router(agents_router.router, prefix="/agents", tags=["🤖 Agents"])
app.include_router(config_router.router, prefix="/config",  tags=["⚙️ Config"])

if GRAPH_AVAILABLE:
    app.include_router(graph.router, prefix="/graph", tags=["🕸️ Graphe de Neurones"])


@app.get("/", tags=["Status"])
def root():
    return {
        "status":  "alive",
        "brain":   APP_NAME,
        "version": APP_VERSION,
        "release": APP_RELEASE_NAME,
        "owner":   "MAKEN",
        "phases_complete": ["1-Foundation", "2-Ingestion+Web", "3-Graph",
                            "4-FileAgent+Analysis", "5-Autonomy+Expertise"],
        "quick_start": {
            "1_domains":     "GET  /brain/list            - voir les domaines disponibles",
            "2_explore":     "POST /brain/explore         - devenir expert dans un domaine",
            "3_expert_chat": "POST /brain/expert-chat     - réponse d'expert comme Perplexity",
            "4_scheduler":   "POST /brain/scheduler/configure - activer l'autonomie",
            "5_report":      "POST /brain/scheduler/report-now - rapport immédiat",
            "6_visualize":   "GET  /static/brain_map.html - visualiser le graphe",
        },
    }

"""
Scheduler autonome — Phase 5
Le cerveau fait des recherches, s'améliore et génère des rapports
pendant que tu travailles ou dors. Sans que tu aies à lui demander.
"""
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

SCHEDULE_FILE = Path("brain_data/scheduler_config.json")
REPORTS_DIR   = Path("brain_data/reports")
_tasks: dict  = {}          # tâches planifiées actives
_running       = False
_loop_task     = None


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "enabled":              False,
    "research_interval_h":  6,      # Recherche web toutes les 6h
    "report_interval_h":    24,     # Rapport quotidien
    "domains_to_watch":     [],     # Domaines surveillés automatiquement
    "queries_auto":         [],     # Requêtes de veille personnalisées
    "last_research":        None,
    "last_report":          None,
    "created_at":           datetime.now().isoformat(),
}


def load_config() -> dict:
    if SCHEDULE_FILE.exists():
        try:
            return json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict) -> None:
    SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULE_FILE.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── Tâches autonomes ──────────────────────────────────────────────────────────

async def run_auto_research(cfg: dict) -> dict:
    """
    Recherche web autonome sur les domaines et requêtes configurés.
    S'exécute périodiquement sans intervention.
    """
    results = []
    tags    = "auto,scheduler"

    # 1. Domaines configurés
    from core.domain_explorer import explore_domain
    for domain in cfg.get("domains_to_watch", [])[:3]:
        try:
            r = await explore_domain(domain, depth="rapide")
            results.append({"type": "domain", "domain": domain,
                             "ingested": r.get("ingested", 0)})
        except Exception as e:
            results.append({"type": "domain", "domain": domain, "error": str(e)})

    # 2. Requêtes de veille personnalisées
    from duckduckgo_search import DDGS
    from routers.files import chunk_text
    from core.memory import add_memory
    from core.dedup import is_duplicate, register

    for query in cfg.get("queries_auto", [])[:5]:
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=3))
            for r in raw:
                content = f"[AUTO] {r.get('title','')}\n{r.get('body','')}"
                if not is_duplicate(content):
                    register(content)
                    for chunk in chunk_text(content):
                        await add_memory(chunk, metadata={
                            "source": r.get("href", ""),
                            "title":  r.get("title", ""),
                            "tags":   tags,
                            "type":   "auto_research",
                        })
            results.append({"type": "query", "query": query, "found": len(raw)})
        except Exception as e:
            results.append({"type": "query", "query": query, "error": str(e)})
        await asyncio.sleep(1)

    cfg["last_research"] = datetime.now().isoformat()
    save_config(cfg)
    print(f"✅ Recherche autonome terminée — {len(results)} tâches")
    return {"tasks": results, "timestamp": cfg["last_research"]}


async def generate_daily_report(cfg: dict) -> dict:
    """
    Génère un rapport quotidien de ce que le cerveau a appris.
    Sauvegardé dans brain_data/reports/
    """
    from core.memory import get_memory_stats, search_memory
    from core.graph import neuron_graph
    from core.providers import groq_generate

    stats = await get_memory_stats()

    # Récupérer les dernières connaissances acquises
    recent_topics = ["nouveautés", "apprentissage", "recherche", "amélioration"]
    recent_memories = []
    for topic in recent_topics[:2]:
        mems = await search_memory(topic, n_results=3)
        recent_memories.extend([m["content"][:200] for m in mems
                                 if m.get("distance", 1) < 0.7])

    graph_stats = neuron_graph.get_stats()

    # Générer le rapport avec Groq
    prompt = (
        f"Tu es MakenBrain, le cerveau numérique de MAKEN. "
        f"Génère un rapport quotidien concis (5-8 lignes) sur ton évolution.\n\n"
        f"Statistiques actuelles :\n"
        f"- Mémoires totales : {stats.get('total_memories', 0)}\n"
        f"- Nœuds dans le graphe : {graph_stats.get('nodes', 0)}\n"
        f"- Connexions : {graph_stats.get('edges', 0)}\n"
        f"- Concepts clés : {', '.join([c['concept'] for c in graph_stats.get('top_concepts', [])[:5]])}\n\n"
        f"Récentes acquisitions :\n" + "\n".join(recent_memories[:3]) + "\n\n"
        f"Format du rapport :\n"
        f"📅 Date\n"
        f"🧠 Ce que j'ai appris aujourd'hui\n"
        f"📈 Croissance (mémoires, connexions)\n"
        f"💡 Ce que je suggère à MAKEN de travailler\n"
        f"⚠️ Lacunes détectées dans mes connaissances"
    )

    report_text = await groq_generate(prompt)

    # Sauvegarder
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str    = datetime.now().strftime("%Y-%m-%d")
    report_file = REPORTS_DIR / f"rapport_{date_str}.txt"
    report_file.write_text(
        f"# Rapport MakenBrain — {date_str}\n\n{report_text}\n\n"
        f"## Stats\n"
        f"- Mémoires : {stats.get('total_memories', 0)}\n"
        f"- Nœuds graphe : {graph_stats.get('nodes', 0)}\n"
        f"- Arêtes graphe : {graph_stats.get('edges', 0)}\n",
        encoding="utf-8",
    )

    cfg["last_report"] = datetime.now().isoformat()
    save_config(cfg)
    print(f"📄 Rapport généré : {report_file}")

    return {
        "report":     report_text,
        "file":       str(report_file),
        "stats":      stats,
        "graph":      graph_stats,
        "timestamp":  cfg["last_report"],
    }


# ── Boucle principale ─────────────────────────────────────────────────────────

async def _scheduler_loop():
    """Boucle infinie qui vérifie les tâches à exécuter."""
    global _running
    print("⏰ Scheduler MakenBrain démarré")
    while _running:
        try:
            cfg = load_config()
            if not cfg.get("enabled"):
                await asyncio.sleep(60)
                continue

            now = datetime.now()

            # Vérifier recherche autonome
            last_r = cfg.get("last_research")
            interval_r = cfg.get("research_interval_h", 6)
            if last_r is None or (now - datetime.fromisoformat(last_r)) >= timedelta(hours=interval_r):
                print("🔍 Lancement recherche autonome...")
                await run_auto_research(cfg)

            # Vérifier rapport quotidien
            last_rep = cfg.get("last_report")
            interval_rep = cfg.get("report_interval_h", 24)
            if last_rep is None or (now - datetime.fromisoformat(last_rep)) >= timedelta(hours=interval_rep):
                print("📄 Génération rapport quotidien...")
                await generate_daily_report(cfg)

        except Exception as e:
            print(f"❌ Erreur scheduler : {e}")

        await asyncio.sleep(300)   # Vérifie toutes les 5 minutes


def start_scheduler(loop: asyncio.AbstractEventLoop = None) -> bool:
    global _running, _loop_task
    if _running:
        return False
    _running = True
    if loop:
        _loop_task = loop.create_task(_scheduler_loop())
    return True


def stop_scheduler():
    global _running, _loop_task
    _running = False
    if _loop_task:
        _loop_task.cancel()
        _loop_task = None


def get_status() -> dict:
    cfg = load_config()
    return {
        "running":             _running,
        "enabled":             cfg.get("enabled", False),
        "research_interval_h": cfg.get("research_interval_h", 6),
        "report_interval_h":   cfg.get("report_interval_h", 24),
        "domains_watched":     cfg.get("domains_to_watch", []),
        "queries_auto":        cfg.get("queries_auto", []),
        "last_research":       cfg.get("last_research"),
        "last_report":         cfg.get("last_report"),
    }

"""
Router Phase 5 — Autonomie & Expertise
- Sélecteur de domaines avec auto-recherche
- Scheduler autonome (travaille pendant que tu dors)
- Rapports quotidiens
- Mode expert par domaine
"""
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from core.audit import audit_event
from core.auth import SECURE
from core.domain_explorer import explore_domain, list_domains, DOMAIN_CATALOG
from core.scheduler import (
    start_scheduler, stop_scheduler, get_status as scheduler_status,
    load_config, save_config, run_auto_research, generate_daily_report,
)

router = APIRouter()

REPORTS_DIR = Path("brain_data/reports")


# ── Schemas ───────────────────────────────────────────────────────────────────

class DomainRequest(BaseModel):
    domain:   str
    depth:    str = "standard"   # rapide | standard | expert
    language: str = "fr"

    model_config = {"json_schema_extra": {"example": {
        "domain": "robotique",
        "depth":  "expert",
        "language": "fr"
    }}}


class ExpertChatRequest(BaseModel):
    question:  str
    domain:    Optional[str] = None
    provider:  str = "groq"

    model_config = {"json_schema_extra": {"example": {
        "question": "Explique-moi le principe d'un contrôleur PID en robotique",
        "domain":   "robotique",
        "provider": "groq"
    }}}


class SchedulerConfigRequest(BaseModel):
    enabled:              bool       = True
    research_interval_h:  int        = 6
    report_interval_h:    int        = 24
    domains_to_watch:     list[str]  = []
    queries_auto:         list[str]  = []

    model_config = {"json_schema_extra": {"example": {
        "enabled": True,
        "research_interval_h": 6,
        "report_interval_h": 24,
        "domains_to_watch": ["intelligence artificielle", "fintech afrique"],
        "queries_auto": [
            "latest AI research papers 2025",
            "mobile money Africa news",
            "spring boot security vulnerabilities"
        ]
    }}}


# ── Endpoints Domaines ────────────────────────────────────────────────────────

@router.get("/list")
async def list_all_domains():
    """
    Liste tous les domaines disponibles avec leurs sous-topics.
    Point de départ pour choisir son domaine d'expertise.
    """
    return {
        "total":   len(DOMAIN_CATALOG),
        "domains": list_domains(),
        "tip":     "Lance POST /brain/explore pour qu'il devienne expert dans un domaine.",
    }


@router.post("/explore")
async def explore(req: DomainRequest, background_tasks: BackgroundTasks):
    """
    Le cerveau explore un domaine en profondeur de façon autonome.

    depth='rapide'   → 2 min, connaissances de base
    depth='standard' → 5 min, bonne maîtrise
    depth='expert'   → 10 min, connaissances pointues + Wikipedia

    Exécuté en arrière-plan — répond immédiatement puis travaille en fond.
    """
    background_tasks.add_task(explore_domain, req.domain, req.depth, req.language)
    return {
        "status":  "en cours d'exploration",
        "domain":  req.domain,
        "depth":   req.depth,
        "message": f"Le cerveau explore '{req.domain}' en mode {req.depth}. Reviens dans quelques minutes.",
        "check":   "GET /memory/stats pour voir la croissance",
    }


@router.post("/explore-sync")
async def explore_sync(req: DomainRequest):
    """
    Même chose que /explore mais attend la fin (bloquant).
    Utilise pour les domaines rapides ou les tests.
    """
    result = await explore_domain(req.domain, req.depth, req.language)
    return result


@router.post("/expert-chat")
async def expert_chat(req: ExpertChatRequest):
    """
    Pose une question à MakenBrain en mode expert.
    Il cherche d'abord dans sa mémoire (domaine spécifique),
    puis répond avec Groq 70B comme un expert humain.
    Réaliste, précis, factuel — comme Perplexity mais avec ta mémoire personnelle.
    """
    from core.memory import search_memory
    from core.providers import groq_generate
    from core.graph import neuron_graph
    from core.extractor import extract_fast

    # 1. Recherche mémoire avec filtre domaine
    memories = await search_memory(req.question, n_results=8)
    if req.domain:
        domain_memories = [m for m in memories
                           if req.domain.lower() in m.get("metadata", {}).get("tags", "").lower()]
        relevant = domain_memories or [m for m in memories if m.get("distance", 1) < 0.7]
    else:
        relevant = [m for m in memories if m.get("distance", 1) < 0.75]

    context_parts = []
    if relevant:
        context_parts.append("=== Mémoire ===")
        for m in relevant[:5]:
            context_parts.append(m["content"][:300])

    # 2. Contexte graphe
    concepts = extract_fast(req.question)
    for c in concepts[:2]:
        r = neuron_graph.explore(c.get("name", ""), depth=2)
        if r.get("found"):
            neighbors = list(r.get("neighbors", {}).keys())[:5]
            if neighbors:
                context_parts.append(f"=== Graphe: {c['name']} → {', '.join(neighbors)} ===")

    context = "\n".join(context_parts)

    # 3. Prompt expert
    domain_str = f" expert en {req.domain}" if req.domain else " expert"
    system = (
        f"Tu es MakenBrain, un assistant{domain_str} au service de MAKEN. "
        f"Tu réponds comme un expert humain : précis, factuel, pédagogue. "
        f"Tu cites des exemples concrets. Tu signales clairement ce que tu ne sais pas. "
        f"Tu es comme Perplexity : tu bases tes réponses sur des faits, pas des suppositions. "
        f"Réponds en français sauf si la question est en anglais."
    )

    response = await groq_generate(req.question, context, system_prompt=system)
    from core.consciousness import consciousness

    response_evaluation = consciousness.evaluate_response(
        question=req.question,
        answer=response,
        user_memory_count=0,
        supabase_data_count=0,
        vector_memory_count=len(relevant),
        graph_context_count=len(context_parts),
    )

    return {
        "question":      req.question,
        "domain":        req.domain or "général",
        "response":      response_evaluation["answer"],
        "confidence":    response_evaluation["confidence"],
        "risk_level":    response_evaluation["risk_level"],
        "suggested_sources": response_evaluation["suggested_sources"],
        "sources_used":  len(relevant),
        "model":         "llama-3.3-70b-versatile (Groq)",
        "context_used":  len(context) > 0,
    }


# ── Endpoints Scheduler ───────────────────────────────────────────────────────

@router.get("/scheduler/status", dependencies=SECURE)
async def get_scheduler_status():
    """Statut du scheduler autonome."""
    return scheduler_status()


@router.post("/scheduler/configure", dependencies=SECURE)
async def configure_scheduler(req: SchedulerConfigRequest):
    """
    Configure et active le scheduler autonome.
    Le cerveau fera des recherches et générera des rapports automatiquement.
    """
    cfg = load_config()
    cfg.update({
        "enabled":              req.enabled,
        "research_interval_h":  req.research_interval_h,
        "report_interval_h":    req.report_interval_h,
        "domains_to_watch":     req.domains_to_watch,
        "queries_auto":         req.queries_auto,
    })
    save_config(cfg)

    if req.enabled:
        loop = asyncio.get_event_loop()
        start_scheduler(loop)
        msg = "Scheduler activé — le cerveau travaille de façon autonome."
    else:
        stop_scheduler()
        msg = "Scheduler désactivé."

    audit_event(
        action="scheduler.configure",
        tool="brain",
        endpoint="/brain/scheduler/configure",
        result=msg,
        success=True,
        details=cfg,
    )
    return {"message": msg, "config": cfg}


@router.post("/scheduler/research-now", dependencies=SECURE)
async def research_now(background_tasks: BackgroundTasks):
    """Lance une recherche autonome immédiatement (sans attendre le scheduler)."""
    cfg = load_config()
    background_tasks.add_task(run_auto_research, cfg)
    audit_event(
        action="scheduler.research_now",
        tool="brain",
        endpoint="/brain/scheduler/research-now",
        result="started",
        success=True,
        details={"domains": cfg.get("domains_to_watch", []), "queries": cfg.get("queries_auto", [])},
    )
    return {
        "status":  "lancé",
        "message": "Recherche autonome démarrée en arrière-plan.",
        "domains": cfg.get("domains_to_watch", []),
        "queries": cfg.get("queries_auto", []),
    }


@router.post("/scheduler/report-now", dependencies=SECURE)
async def report_now():
    """Génère le rapport quotidien immédiatement."""
    cfg    = load_config()
    report = await generate_daily_report(cfg)
    audit_event(action="scheduler.report_now", tool="brain", endpoint="/brain/scheduler/report-now", result="generated", success=True)
    return report


@router.get("/reports")
async def list_reports():
    """Liste tous les rapports générés."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reports = sorted(REPORTS_DIR.glob("rapport_*.txt"), reverse=True)
    return {
        "total": len(reports),
        "reports": [
            {
                "file": f.name,
                "date": f.stem.replace("rapport_", ""),
                "size": f.stat().st_size,
            }
            for f in reports[:20]
        ],
    }


@router.get("/reports/{date}")
async def get_report(date: str):
    """Récupère un rapport par date (format YYYY-MM-DD)."""
    report_file = REPORTS_DIR / f"rapport_{date}.txt"
    if not report_file.exists():
        from fastapi import HTTPException
        raise HTTPException(404, f"Rapport {date} introuvable.")
    return {
        "date":    date,
        "content": report_file.read_text(encoding="utf-8"),
    }

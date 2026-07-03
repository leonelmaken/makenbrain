"""
Recherche web autonome — le cerveau se nourrit d'internet.
Utilise DuckDuckGo (gratuit, sans clé API).
Ce router délègue toute la logique métier à core.web_search.
"""
import asyncio
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from core.web_search import fetch_page_text, ingest_web_result, search_and_ingest

router = APIRouter()

# ── Schemas ───────────────────────────────────────────────────────────────────

class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5
    fetch_pages: bool = True
    auto_summarize: bool = False
    tags: Optional[str] = ""

    model_config = {"json_schema_extra": {
        "example": {
            "query": "best practices tontine fintech Africa 2025",
            "max_results": 5,
            "fetch_pages": True,
            "auto_summarize": False,
            "tags": "web,fintech,tontine"
        }
    }}


class AutoResearchRequest(BaseModel):
    topic: str
    depth: int = 3
    max_results_per_query: int = 3
    tags: Optional[str] = ""

    model_config = {"json_schema_extra": {
        "example": {
            "topic": "Mobile Money integration MTN MoMo API Africa",
            "depth": 3,
            "tags": "web,momo,integration"
        }
    }}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/web")
async def web_search(request: WebSearchRequest):
    """
    Recherche sur le web et ingère les résultats en mémoire.
    Le cerveau apprend depuis internet de façon autonome.
    """
    return await search_and_ingest(
        query          = request.query,
        max_results    = request.max_results,
        fetch_pages    = request.fetch_pages,
        auto_summarize = request.auto_summarize,
        tags           = request.tags or "",
    )


@router.post("/auto-research")
async def auto_research(request: AutoResearchRequest):
    """
    Recherche autonome multi-angles.
    Le cerveau génère plusieurs requêtes autour d'un sujet,
    cherche sur le web, et ingère tout automatiquement.
    """
    from core.llm import _client
    from core.config import settings

    prompt = f"""Tu es un assistant de recherche. Génère {request.depth} requêtes de recherche Google différentes et complémentaires sur ce sujet : "{request.topic}"

Règles :
- Chaque requête sur une nouvelle ligne
- En anglais (meilleurs résultats)
- Angles différents : définition, cas pratiques, comparaisons, actualités, tutoriels
- Maximum 10 mots par requête

Réponds UNIQUEMENT avec les requêtes, une par ligne, sans numérotation ni explication."""

    try:
        response = _client.chat(
            model=settings.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7, "num_predict": 200},
        )
        queries = [q.strip() for q in response.message.content.strip().split("\n") if q.strip()][:request.depth]
    except Exception:
        queries = [request.topic]

    all_results = []
    for query in queries:
        try:
            # `ddgs` = successeur officiel de `duckduckgo_search` (renommé)
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=request.max_results_per_query))
            for r in raw:
                full_text = await fetch_page_text(r.get("href", ""))
                result = await ingest_web_result(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                    full_text=full_text,
                    tags=f"auto-research,{request.tags}".strip(","),
                    query=query,
                )
                all_results.append({**result, "query": query})
        except Exception as e:
            all_results.append({"query": query, "error": str(e)})
        await asyncio.sleep(1)

    ingested_count = len([r for r in all_results if r.get("status") == "ingéré"])

    return {
        "topic":             request.topic,
        "queries_generated": queries,
        "total_results":     len(all_results),
        "ingested":          ingested_count,
        "message":           f"Recherche autonome sur '{request.topic}' — {ingested_count} nouveaux souvenirs web.",
        "details":           all_results,
    }


@router.get("/status")
async def search_status():
    """Vérifie que DuckDuckGo est accessible."""
    try:
        # `ddgs` = successeur officiel de `duckduckgo_search` (renommé)
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            test = list(ddgs.text("test", max_results=1))
        return {"status": "online", "engine": "DuckDuckGo", "api_key_required": False}
    except Exception as e:
        return {"status": "offline", "error": str(e)}

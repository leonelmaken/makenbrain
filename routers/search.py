"""
Recherche web autonome — le cerveau se nourrit d'internet.
Utilise DuckDuckGo (gratuit, sans clé API).
Cherche, lit les pages, ingère les résultats en mémoire.
"""
import asyncio
from typing import Optional
import httpx
from bs4 import BeautifulSoup
from fastapi import APIRouter
from pydantic import BaseModel
from duckduckgo_search import DDGS

from core.memory import add_memory
from core.dedup import is_duplicate, register
from core.summarizer import summarize

router = APIRouter()

# ── Schemas ───────────────────────────────────────────────────────────────────

class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5
    fetch_pages: bool = True        # Lire le contenu complet des pages
    auto_summarize: bool = False    # Résumer chaque page (lent sur CPU)
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
    depth: int = 3              # Nombre de requêtes générées automatiquement
    max_results_per_query: int = 3
    tags: Optional[str] = ""

    model_config = {"json_schema_extra": {
        "example": {
            "topic": "Mobile Money integration MTN MoMo API Africa",
            "depth": 3,
            "tags": "web,momo,integration"
        }
    }}


# ── Utilitaires ───────────────────────────────────────────────────────────────

async def fetch_page_text(url: str, timeout: float = 10.0) -> str:
    """Récupère et nettoie le texte d'une page web."""
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "MakenBrain/0.2 (Personal AI Research Tool)"}
            )
            resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "ads"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        # Nettoyer les espaces multiples
        import re
        return re.sub(r'\s{2,}', ' ', text)[:8000]  # Max 8000 chars
    except Exception:
        return ""


async def ingest_web_result(
    title: str,
    url: str,
    snippet: str,
    full_text: str = "",
    tags: str = "",
    query: str = "",
) -> dict:
    """Ingère un résultat web dans la mémoire du cerveau."""
    # Contenu à stocker : snippet minimum, page complète si disponible
    content = full_text if len(full_text) > len(snippet) else snippet
    content = f"[WEB] {title}\nSource: {url}\n\n{content}"

    if is_duplicate(content):
        return {"title": title, "url": url, "status": "déjà connu"}

    register(content)

    # Découper si contenu long
    from routers.files import chunk_text
    chunks = chunk_text(content, chunk_size=500)

    for i, chunk in enumerate(chunks):
        await add_memory(
            content=chunk,
            metadata={
                "source": url,
                "title": title,
                "tags": f"web,{tags}".strip(","),
                "query": query,
                "chunk_index": str(i),
                "total_chunks": str(len(chunks)),
                "type": "web_search",
            }
        )

    return {"title": title, "url": url, "chunks": len(chunks), "status": "ingéré"}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/web")
async def web_search(request: WebSearchRequest):
    """
    Recherche sur le web et ingère les résultats en mémoire.
    Le cerveau apprend depuis internet de façon autonome.
    """
    results = []

    # Recherche DuckDuckGo
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(request.query, max_results=request.max_results))
    except Exception as e:
        return {"error": f"Recherche DuckDuckGo échouée : {e}", "results": []}

    ingested = []
    for r in raw_results:
        title   = r.get("title", "")
        url     = r.get("href", "")
        snippet = r.get("body", "")

        # Récupérer le contenu complet de la page si demandé
        full_text = ""
        if request.fetch_pages and url:
            full_text = await fetch_page_text(url)

        result = await ingest_web_result(
            title=title,
            url=url,
            snippet=snippet,
            full_text=full_text,
            tags=request.tags or "",
            query=request.query,
        )
        ingested.append(result)

    return {
        "query": request.query,
        "results_found": len(raw_results),
        "ingested": len([r for r in ingested if r["status"] == "ingéré"]),
        "already_known": len([r for r in ingested if r["status"] == "déjà connu"]),
        "details": ingested,
        "message": f"Recherche '{request.query}' — {len(ingested)} résultat(s) traité(s).",
    }


@router.post("/auto-research")
async def auto_research(request: AutoResearchRequest):
    """
    Recherche autonome multi-angles.
    Le cerveau génère plusieurs requêtes autour d'un sujet,
    cherche sur le web, et ingère tout automatiquement.
    Simule un chercheur autonome.
    """
    from core.llm import _client
    from core.config import settings

    # Le LLM génère des requêtes de recherche pertinentes
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
    except Exception as e:
        queries = [request.topic]

    # Rechercher et ingérer pour chaque requête
    all_results = []
    for query in queries:
        try:
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
        # Petite pause entre requêtes pour éviter le rate limiting
        await asyncio.sleep(1)

    ingested_count = len([r for r in all_results if r.get("status") == "ingéré"])

    return {
        "topic": request.topic,
        "queries_generated": queries,
        "total_results": len(all_results),
        "ingested": ingested_count,
        "message": f"Recherche autonome sur '{request.topic}' — {ingested_count} nouveaux souvenirs web.",
        "details": all_results,
    }


@router.get("/status")
async def search_status():
    """Vérifie que DuckDuckGo est accessible."""
    try:
        with DDGS() as ddgs:
            test = list(ddgs.text("test", max_results=1))
        return {"status": "online", "engine": "DuckDuckGo", "api_key_required": False}
    except Exception as e:
        return {"status": "offline", "error": str(e)}

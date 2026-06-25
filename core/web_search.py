"""
Recherche web et ingestion — couche métier pure.

Ce module contient toute la logique de recherche web et d'ingestion
en mémoire, extraite de routers/search.py pour respecter l'architecture :

    routers/  →  core/  →  models/

Règle : ce module n'importe jamais depuis routers/.

Utilisateurs :
- routers/search.py  : délègue les endpoints HTTP à ce module.
- core/domain_explorer.py : appelle directement search_and_ingest().
"""
from __future__ import annotations

import re

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from core.dedup import is_duplicate, register
from core.ingestion import chunk_text
from core.memory import add_memory


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
        return re.sub(r'\s{2,}', ' ', text)[:8000]
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
    content = full_text if len(full_text) > len(snippet) else snippet
    content = f"[WEB] {title}\nSource: {url}\n\n{content}"

    if is_duplicate(content):
        return {"title": title, "url": url, "status": "déjà connu"}

    register(content)

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


async def search_and_ingest(
    query: str,
    max_results: int = 5,
    fetch_pages: bool = True,
    auto_summarize: bool = False,
    tags: str = "",
) -> dict:
    """Recherche sur DuckDuckGo et ingère les résultats en mémoire.

    Args:
        query         : Requête de recherche.
        max_results   : Nombre maximum de résultats DuckDuckGo.
        fetch_pages   : Si True, lit le contenu complet de chaque page.
        auto_summarize: Paramètre réservé (non utilisé, compatibilité future).
        tags          : Tags à associer aux chunks ingérés.

    Returns:
        Dict avec clés : query, results_found, ingested, already_known,
                         details, message.
    """
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return {"error": f"Recherche DuckDuckGo échouée : {e}", "results": [],
                "query": query, "results_found": 0, "ingested": 0,
                "already_known": 0, "details": [], "message": str(e)}

    ingested: list[dict] = []
    for r in raw_results:
        title   = r.get("title", "")
        url     = r.get("href", "")
        snippet = r.get("body", "")

        full_text = ""
        if fetch_pages and url:
            full_text = await fetch_page_text(url)

        result = await ingest_web_result(
            title=title,
            url=url,
            snippet=snippet,
            full_text=full_text,
            tags=tags,
            query=query,
        )
        ingested.append(result)

    return {
        "query":         query,
        "results_found": len(raw_results),
        "ingested":      len([r for r in ingested if r["status"] == "ingéré"]),
        "already_known": len([r for r in ingested if r["status"] == "déjà connu"]),
        "details":       ingested,
        "message":       f"Recherche '{query}' — {len(ingested)} résultat(s) traité(s).",
    }

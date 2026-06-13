import re
import httpx
from fastapi import APIRouter, HTTPException
from bs4 import BeautifulSoup
from models.schemas import IngestTextRequest, IngestUrlRequest, IngestResponse
from core.memory import add_memory

router = APIRouter()


def chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    """Découpe le texte en fragments cohérents (sur fins de phrase)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= chunk_size:
            current = current + " " + s if current else s
        else:
            if current:
                chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 20]


@router.post("/text", response_model=IngestResponse)
async def ingest_text(request: IngestTextRequest):
    """Ingère un texte brut, le fragmente et le stocke en mémoire."""
    chunks = chunk_text(request.text, chunk_size=request.chunk_size)
    if not chunks:
        raise HTTPException(status_code=400, detail="Texte trop court ou vide.")
    ids = []
    for i, chunk in enumerate(chunks):
        mid = await add_memory(chunk, metadata={
            "source": request.source, "title": request.title or "Sans titre",
            "tags": request.tags, "chunk_index": str(i), "total_chunks": str(len(chunks)),
        })
        ids.append(mid)
    return IngestResponse(
        message=f"{len(chunks)} fragment(s) ingéré(s) dans le cerveau.",
        chunks_created=len(chunks), ids=ids, title=request.title or "Sans titre",
    )


@router.post("/url", response_model=IngestResponse)
async def ingest_url(request: IngestUrlRequest):
    """Récupère et ingère le contenu textuel d'une URL."""
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(
                request.url,
                headers={"User-Agent": "MakenBrain/0.1 (Personal AI Research Tool)"},
            )
            resp.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"URL inaccessible : {e}")

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title else request.url
    text = re.sub(r'\s{2,}', ' ', soup.get_text(separator=" ", strip=True))

    if len(text) < 50:
        raise HTTPException(status_code=400, detail="Contenu trop court ou inaccessible.")

    chunks = chunk_text(text)
    ids = []
    for i, chunk in enumerate(chunks):
        mid = await add_memory(chunk, metadata={
            "source": request.url, "title": title, "tags": request.tags,
            "chunk_index": str(i), "total_chunks": str(len(chunks)), "ingestion_type": "url",
        })
        ids.append(mid)

    return IngestResponse(
        message=f"'{title}' — {len(chunks)} fragment(s) ingéré(s).",
        chunks_created=len(chunks), ids=ids, title=title,
    )

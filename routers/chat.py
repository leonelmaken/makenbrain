from fastapi import APIRouter
from models.schemas import ChatRequest, ChatResponse
from core.llm import generate, check_ollama_status
from core.memory import search_memory
from core.config import settings

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Discuter avec MakenBrain.
    Si use_memory=True, le cerveau cherche ses souvenirs (RAG) avant de répondre.
    """
    context = ""
    memories_used = 0
    context_preview = []

    if request.use_memory:
        memories = await search_memory(request.message, n_results=request.n_context)
        relevant = [m for m in memories if m["distance"] < request.relevance_threshold]
        if relevant:
            parts = []
            for i, m in enumerate(relevant):
                parts.append(f"[Souvenir {i+1}] {m['content']}")
                preview = m["content"][:100] + "..." if len(m["content"]) > 100 else m["content"]
                context_preview.append(preview)
            context = "\n\n".join(parts)
            memories_used = len(relevant)

    response_text = await generate(request.message, context)

    return ChatResponse(
        response=response_text,
        memories_used=memories_used,
        model=settings.OLLAMA_MODEL,
        context_preview=context_preview if context_preview else None,
    )


@router.get("/status")
async def status():
    """Vérifie la disponibilité du LLM Ollama."""
    return await check_ollama_status()

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from core.llm import generate, check_ollama_status
from core.memory import search_memory
from core.providers import groq_generate, detect_provider
from core.config import settings

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    use_memory: bool = True
    n_context: int = 5
    relevance_threshold: float = 0.75
    provider: str = "auto"  # "auto" | "local" | "groq"

    model_config = {"json_schema_extra": {
        "example": {
            "message": "Analyse l'architecture de SmartBudget Africa et propose des améliorations.",
            "use_memory": True,
            "n_context": 5,
            "provider": "auto"
        }
    }}


class ChatResponse(BaseModel):
    response: str
    memories_used: int
    model: str
    provider: str
    context_preview: Optional[list[str]] = None


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat intelligent avec MakenBrain.
    provider='auto' → le cerveau choisit seul local ou Groq 70B selon la complexité.
    provider='groq'  → force Groq (questions complexes, analyses, architecture).
    provider='local' → force le LLM local (questions simples, rapide).
    """
    # 1. Chercher dans la mémoire
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

    # 2. Choisir le provider
    chosen = detect_provider(request.message, request.provider)

    # 3. Générer la réponse
    if chosen == "groq":
        response_text = await groq_generate(request.message, context)
        model_name = "llama-3.3-70b-versatile (Groq)"
    else:
        response_text = await generate(request.message, context)
        model_name = settings.OLLAMA_MODEL + " (local)"

    return ChatResponse(
        response=response_text,
        memories_used=memories_used,
        model=model_name,
        provider=chosen,
        context_preview=context_preview if context_preview else None,
    )


@router.get("/status")
async def status():
    """Statut du LLM local Ollama."""
    return await check_ollama_status()

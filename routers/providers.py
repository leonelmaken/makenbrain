"""
Endpoints des connecteurs externes :
- Statut de tous les providers
- Chat direct Groq 70B
- Recherche et ingestion Wikipedia
"""
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from core.providers import (
    groq_generate, check_groq,
    wikipedia_search, wikipedia_ingest,
    GROQ_MODEL,
)
from core.config import settings, reload_settings
from core.llm import reload_client as reload_llm_client
from core.summarizer import reload_client as reload_summarizer_client
from core.memory import reload_memory
from core.scheduler import load_config as load_scheduler_config

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class GroqChatRequest(BaseModel):
    message: str
    context: str = ""

    model_config = {"json_schema_extra": {
        "example": {
            "message": "Analyse l'architecture de mon backend Spring Boot SmartBudget et dis-moi les points à améliorer.",
            "context": ""
        }
    }}


class WikiRequest(BaseModel):
    query: str
    lang: str = "fr"
    ingest: bool = True

    model_config = {"json_schema_extra": {
        "example": {
            "query": "tontine Afrique subsaharienne",
            "lang": "fr",
            "ingest": True
        }
    }}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status")
async def providers_status():
    """Statut de tous les connecteurs externes disponibles."""
    groq_status = await check_groq()
    return {
        "providers": {
            "local_llm": {
                "status": "online",
                "model": "llama3.2:3b",
                "type": "local",
                "speed": "lent (CPU)",
                "power": "basique",
            },
            "groq": {
                **groq_status,
                "model": GROQ_MODEL,
                "type": "cloud gratuit",
                "speed": "< 3 secondes",
                "power": "expert (70B paramètres)",
            },
            "wikipedia": {
                "status": "online",
                "type": "encyclopédie libre",
                "api_key_required": False,
                "languages": ["fr", "en", "es", "de", "ar"],
            },
            "duckduckgo": {
                "status": "online",
                "type": "moteur de recherche",
                "api_key_required": False,
            },
        }
    }


@router.post("/groq/chat")
async def groq_chat(request: GroqChatRequest):
    """
    Chat direct avec Groq (Llama 3.3 70B).
    Utilise ce endpoint pour les questions complexes qui nécessitent
    plus de puissance que le modèle local.
    """
    response = await groq_generate(request.message, request.context)
    return {
        "response": response,
        "model": GROQ_MODEL,
        "provider": "groq",
    }


@router.post("/wikipedia")
async def wiki(request: WikiRequest):
    """
    Cherche sur Wikipedia.
    Si ingest=True (défaut), le résultat est automatiquement
    stocké en mémoire pour enrichir le cerveau.
    """
    if request.ingest:
        result = await wikipedia_ingest(request.query, request.lang)
        return {"action": "search + ingest", **result}
    else:
        result = await wikipedia_search(request.query, request.lang)
        return {"action": "search only", **result}


@router.post("/reload-config")
async def reload_config():
    """Recharge la configuration depuis le fichier .env et réinitialise les clients utiles."""
    reload_settings()
    reload_llm_client()
    reload_summarizer_client()
    await reload_memory()

    scheduler_config = load_scheduler_config()

    return {
        "message": "Configuration rechargée avec succès.",
        "settings": {
            "OLLAMA_HOST": settings.OLLAMA_HOST,
            "OLLAMA_MODEL": settings.OLLAMA_MODEL,
            "CHROMA_PATH": settings.CHROMA_PATH,
            "COLLECTION_NAME": settings.COLLECTION_NAME,
            "EMBED_MODEL": settings.EMBED_MODEL,
            "API_HOST": settings.API_HOST,
            "API_PORT": settings.API_PORT,
            "providers": {
                "groq_enabled": bool(settings.GROQ_API_KEY),
                "hf_enabled": bool(settings.HF_API_KEY),
                "anthropic_enabled": bool(settings.ANTHROPIC_API_KEY),
            },
        },
        "scheduler_config": scheduler_config,
    }

"""Package Provider Layer — Phase 4.

Fournit une interface unifiée pour tous les providers LLM supportés :
    - Ollama (local)
    - Groq (cloud, Llama 3.3 70B)
    - Extensible : Claude, OpenAI, Gemini, DeepSeek...

Exports publics :
    LLMProvider    : Classe de base abstraite pour tout provider.
    ProviderHealth : DTO du statut de santé d'un provider.
    OllamaProvider : Provider Ollama local.
    GroqProvider   : Provider Groq cloud.
    LLMRouter      : Routeur avec fallback automatique.
    get_router     : Singleton du routeur global.
"""
from core.provider_layer.base import LLMProvider, ProviderHealth
from core.provider_layer.ollama_provider import OllamaProvider
from core.provider_layer.groq_provider import GroqProvider
from core.provider_layer.router import LLMRouter, get_router

__all__ = [
    "LLMProvider",
    "ProviderHealth",
    "OllamaProvider",
    "GroqProvider",
    "LLMRouter",
    "get_router",
]

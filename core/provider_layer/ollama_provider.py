"""Provider Ollama — Phase 4.

Wrapper du client Ollama existant (core/llm.py) derrière l'interface LLMProvider.
Tous les appels au SDK Ollama (synchrone) sont délégués à asyncio.to_thread
pour ne pas bloquer la boucle asyncio de FastAPI.
"""
from __future__ import annotations

import asyncio
import time

import ollama

from core.config import settings
from core.provider_layer.base import LLMProvider, ProviderHealth
from core.system_prompts import SYSTEM_PROMPT


class OllamaProvider(LLMProvider):
    """Provider LLM local via Ollama.

    Utilise le même client Ollama que core/llm.py mais respecte
    l'interface LLMProvider — aucune exception ne remonte à l'appelant.
    """

    def __init__(
        self,
        host : str | None = None,
        model: str | None = None,
    ) -> None:
        self._host  = host  or settings.OLLAMA_HOST
        self._model = model or settings.OLLAMA_MODEL
        self._client = ollama.Client(host=self._host)

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    async def generate(
        self,
        prompt       : str,
        context      : str  = "",
        system_prompt: str | None = None,
    ) -> str | None:
        """Génère une réponse via Ollama (local).

        Returns:
            Texte généré, ou None si Ollama est hors ligne.
        """
        current_system = system_prompt or SYSTEM_PROMPT
        messages: list[dict] = []

        if context:
            messages.append({
                "role"   : "user",
                "content": (
                    f"Voici les souvenirs pertinents extraits de ma mémoire :\n\n"
                    f"{context}\n\n"
                    f"Utilise ces informations pour répondre à ma prochaine question."
                ),
            })
            messages.append({
                "role"   : "assistant",
                "content": "Compris. J'ai intégré ces souvenirs dans mon raisonnement.",
            })

        messages.append({"role": "user", "content": prompt})

        full_messages = [{"role": "system", "content": current_system}] + messages

        try:
            response = await asyncio.to_thread(
                self._client.chat,
                model   = self._model,
                messages= full_messages,
                options = {"temperature": 0.7, "num_predict": 1024},
            )
            return response.message.content
        except Exception:
            return None

    async def health(self) -> ProviderHealth:
        """Vérifie que Ollama est accessible et que le modèle est disponible."""
        t0 = time.monotonic()
        try:
            result = await asyncio.to_thread(self._client.list)
            latency = round((time.monotonic() - t0) * 1000, 1)
            model_names = [m.model for m in result.models]
            loaded = any(self._model in m for m in model_names)
            return ProviderHealth(
                name      = self.name,
                available = loaded,
                model     = self._model,
                latency_ms= latency,
                details   = {
                    "host"            : self._host,
                    "available_models": model_names,
                    "model_loaded"    : loaded,
                },
                error = None if loaded else f"Modèle '{self._model}' non trouvé dans Ollama.",
            )
        except Exception as exc:
            return ProviderHealth(
                name      = self.name,
                available = False,
                model     = self._model,
                error     = str(exc),
            )

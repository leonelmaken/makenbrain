"""Provider Groq — Phase 4.

Wrapper du client Groq existant (core/providers.py) derrière l'interface LLMProvider.
Le SDK Groq est synchrone — tous les appels sont wrappés dans asyncio.to_thread.
"""
from __future__ import annotations

import asyncio
import time

from core.config import settings
from core.provider_layer.base import LLMProvider, ProviderHealth
from core.system_prompts import SYSTEM_PROMPT

GROQ_MODEL_DEFAULT = "llama-3.3-70b-versatile"


class GroqProvider(LLMProvider):
    """Provider LLM cloud via Groq (Llama 3.3 70B).

    N'est pas instanciable si GROQ_API_KEY est absent.
    health() renvoie available=False dans ce cas.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or GROQ_MODEL_DEFAULT

    @property
    def name(self) -> str:
        return "groq"

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self):
        """Instancie le client Groq à la volée (lazy, pas de singleton).

        Raises:
            ValueError : si GROQ_API_KEY est manquant dans la config.
        """
        from groq import Groq as GroqClient
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY manquant dans .env")
        return GroqClient(api_key=settings.GROQ_API_KEY)

    async def generate(
        self,
        prompt       : str,
        context      : str  = "",
        system_prompt: str | None = None,
    ) -> str | None:
        """Génère une réponse via Groq (cloud).

        Returns:
            Texte généré, ou None si Groq est indisponible / mal configuré.
        """
        try:
            client = self._get_client()
        except ValueError:
            return None

        current_system = system_prompt or SYSTEM_PROMPT
        messages: list[dict] = []

        if context:
            messages.append({
                "role"   : "user",
                "content": f"Mémoire disponible :\n\n{context}",
            })
            messages.append({
                "role"   : "assistant",
                "content": "Compris, j'intègre ces souvenirs dans ma réponse.",
            })

        messages.append({"role": "user", "content": prompt})
        full_messages = [{"role": "system", "content": current_system}] + messages

        def _sync_call() -> str:
            response = client.chat.completions.create(
                model      = self._model,
                messages   = full_messages,
                temperature= 0.7,
                max_tokens = 2048,
            )
            return response.choices[0].message.content or ""

        try:
            return await asyncio.to_thread(_sync_call)
        except Exception:
            return None

    async def health(self) -> ProviderHealth:
        """Vérifie que la clé Groq est configurée et que l'API est joignable."""
        if not settings.GROQ_API_KEY:
            return ProviderHealth(
                name     = self.name,
                available= False,
                model    = self._model,
                error    = "GROQ_API_KEY manquant dans .env",
            )

        t0 = time.monotonic()
        try:
            client = self._get_client()

            def _list_models():
                return client.models.list()

            result  = await asyncio.to_thread(_list_models)
            latency = round((time.monotonic() - t0) * 1000, 1)
            names   = [m.id for m in result.data]
            return ProviderHealth(
                name      = self.name,
                available = True,
                model     = self._model,
                latency_ms= latency,
                details   = {"available_models_count": len(names)},
            )
        except Exception as exc:
            return ProviderHealth(
                name     = self.name,
                available= False,
                model    = self._model,
                error    = str(exc),
            )

"""LLM Router avec fallback automatique — Phase 4.

Le LLMRouter maintient une liste ordonnée de providers. Lorsqu'un
generate() est demandé, il essaie les providers dans l'ordre jusqu'à
obtenir une réponse non-None. Chaque tentative et chaque fallback est
enregistré dans MetricsCollector.

Ordre par défaut :
    1. Groq   (cloud, disponible si GROQ_API_KEY configuré)
    2. Ollama (local, toujours en dernier recours)

Ce comportement reproduit la logique de detect_provider() de core/providers.py
mais de façon explicite, testable et extensible.

Singleton :
    get_router() retourne l'instance globale. Elle est initialisée lors du
    premier appel et réutilisée pendant toute la durée de vie du processus.
"""
from __future__ import annotations

import threading
from typing import Sequence

from core.provider_layer.base import LLMProvider, ProviderHealth
from core.provider_layer.groq_provider import GroqProvider
from core.provider_layer.ollama_provider import OllamaProvider

# Message sentinelle retourné quand TOUS les providers ont échoué.
# Les agents peuvent comparer la sortie à cette constante pour détecter
# un échec total (le router ne lève jamais d'exception par contrat).
ALL_PROVIDERS_FAILED = (
    "Aucun provider LLM n'est disponible pour le moment. "
    "Vérifie qu'Ollama est lancé ou que GROQ_API_KEY est configuré."
)


class LLMRouter:
    """Routeur multi-provider avec fallback automatique.

    Args:
        providers : Liste ordonnée de providers (priorité décroissante).
                    Si vide, utilise la liste par défaut (Groq → Ollama).
    """

    def __init__(self, providers: Sequence[LLMProvider] | None = None) -> None:
        if providers is None:
            self._providers: list[LLMProvider] = [
                GroqProvider(),
                OllamaProvider(),
            ]
        else:
            self._providers = list(providers)

    @property
    def providers(self) -> list[LLMProvider]:
        return list(self._providers)

    async def generate(
        self,
        prompt       : str,
        context      : str  = "",
        system_prompt: str | None = None,
        preferred    : str | None = None,
    ) -> str:
        """Génère une réponse en essayant les providers dans l'ordre.

        Si `preferred` est fourni et correspond au nom d'un provider connu,
        ce provider est essayé en premier. Sinon, l'ordre par défaut s'applique.

        Args:
            prompt        : Message de l'utilisateur.
            context       : Contexte mémoire RAG optionnel.
            system_prompt : Prompt système (None = défaut du provider).
            preferred     : Nom du provider préféré (ex. "ollama", "groq").

        Returns:
            Texte généré. Ne lève jamais d'exception — retourne un message
            d'erreur lisible si tous les providers ont échoué.
        """
        from core.observability import get_metrics

        metrics = get_metrics()
        ordered = self._ordered_providers(preferred)

        first = True
        for provider in ordered:
            is_fallback = not first
            first = False

            result = await provider.generate(
                prompt       = prompt,
                context      = context,
                system_prompt= system_prompt,
            )

            success = result is not None
            metrics.record_llm_call(
                provider= provider.name,
                model   = provider.model,
                success = success,
                fallback= is_fallback,
            )

            if success:
                return result  # type: ignore[return-value]

        return ALL_PROVIDERS_FAILED

    async def health_all(self) -> list[ProviderHealth]:
        """Retourne le statut de santé de tous les providers enregistrés.

        Les checks sont exécutés séquentiellement pour éviter les surcharges
        réseau. Pour la Phase 7 Dashboard, on pourra les paralléliser.

        Returns:
            Liste de ProviderHealth dans l'ordre d'enregistrement.
        """
        results: list[ProviderHealth] = []
        for provider in self._providers:
            h = await provider.health()
            results.append(h)
        return results

    def _ordered_providers(self, preferred: str | None) -> list[LLMProvider]:
        """Retourne la liste des providers réordonnée selon `preferred`."""
        if not preferred:
            return self._providers

        head = [p for p in self._providers if p.name == preferred]
        tail = [p for p in self._providers if p.name != preferred]
        return head + tail


# ── Singleton global ───────────────────────────────────────────────────────────

_global_router: LLMRouter | None = None
_global_router_lock = threading.Lock()


def get_router() -> LLMRouter:
    """Retourne le singleton global LLMRouter.

    Thread-safe. Crée l'instance au premier appel avec la liste de
    providers par défaut (Groq → Ollama).

    Returns:
        Instance singleton de LLMRouter.
    """
    global _global_router
    if _global_router is None:
        with _global_router_lock:
            if _global_router is None:
                _global_router = LLMRouter()
    return _global_router

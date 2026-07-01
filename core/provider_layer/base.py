"""Abstractions de base du Provider Layer — Phase 4.

Tout nouveau provider LLM doit hériter de LLMProvider et implémenter
les deux méthodes abstraites : generate() et health().

Design :
    - generate() est toujours async, même pour les SDKs synchrones
      (les providers wrappent les appels sync dans asyncio.to_thread).
    - health() doit être rapide (< 2s) et ne pas générer de tokens.
    - Les providers ne lèvent pas d'exception vers l'appelant : ils
      retournent None en cas d'échec (le Router décide du fallback).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderHealth:
    """Statut de santé d'un provider LLM.

    Attributes:
        name         : Nom du provider (ex. "ollama", "groq").
        available    : True si le provider peut actuellement générer.
        model        : Modèle actif ou vide si non applicable.
        latency_ms   : Latence du dernier ping en ms (None si non mesuré).
        details      : Dict arbitraire de métadonnées supplémentaires.
        error        : Message d'erreur si available=False.
    """
    name      : str
    available : bool
    model     : str             = ""
    latency_ms: float | None    = None
    details   : dict[str, Any]  = field(default_factory=dict)
    error     : str | None      = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name"     : self.name,
            "available": self.available,
            "model"    : self.model,
        }
        if self.latency_ms is not None:
            d["latency_ms"] = self.latency_ms
        if self.details:
            d["details"] = self.details
        if self.error:
            d["error"] = self.error
        return d


class LLMProvider(ABC):
    """Classe de base abstraite pour tout provider LLM.

    Convention d'implémentation :
        - generate() ne lève jamais d'exception : retourne None si échec.
        - health() ne génère pas de tokens, répond en < 2 secondes.
        - Le champ name doit être unique dans l'application.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifiant unique du provider (ex. "ollama", "groq")."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Modèle LLM actif pour ce provider."""

    @abstractmethod
    async def generate(
        self,
        prompt       : str,
        context      : str  = "",
        system_prompt: str | None = None,
    ) -> str | None:
        """Génère une réponse texte.

        Args:
            prompt        : Le message de l'utilisateur.
            context       : Contexte mémoire optionnel (RAG).
            system_prompt : Prompt système. Si None, utilise le défaut du provider.

        Returns:
            Texte généré, ou None si le provider est indisponible.
        """

    @abstractmethod
    async def health(self) -> ProviderHealth:
        """Vérifie la disponibilité du provider.

        Returns:
            ProviderHealth avec available=True si le provider est prêt.
        """

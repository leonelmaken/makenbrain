"""Collecteur de métriques in-memory — Phase 4.

Architecture :
    MetricsCollector est un singleton thread-safe (RLock) qui agrège les
    métriques de toutes les requêtes HTTP et tous les appels LLM.
    Les métriques sont purement in-memory : pas de dépendance externe.

    Pour la Phase 7+ (Dashboard Super Admin), ce module exposera une
    méthode snapshot() que l'endpoint /metrics pourra sérialiser en JSON.

Métriques collectées :
    Requêtes HTTP :
        - Nombre total de requêtes par endpoint
        - Nombre d'erreurs par endpoint (status >= 400)
        - Latence cumulée par endpoint (pour calculer la moyenne)

    Appels LLM :
        - Nombre d'appels par provider (ollama, groq, claude, …)
        - Nombre d'appels par modèle
        - Nombre d'appels réussis vs échoués
        - Nombre de fallbacks déclenchés
        - Tokens consommés par provider

    Pipeline de raisonnement :
        - Nombre d'exécutions par étape (question_analyzer, hypothesis_engine, …)
        - Latence cumulée par étape
        - Nombre d'erreurs par étape

Usage :
    from core.observability import get_metrics

    metrics = get_metrics()
    metrics.record_request(endpoint="/reasoning/analyze", duration_ms=142.5, success=True)
    metrics.record_llm_call(provider="groq", model="llama-3.3-70b-versatile",
                            success=True, tokens=512)
    metrics.record_pipeline_step(step="synthesizer", duration_ms=38.2, success=True)

    # Pour le /health ou le futur Dashboard :
    snapshot = metrics.snapshot()
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ── Structures de données ──────────────────────────────────────────────────────

@dataclass
class _EndpointStats:
    total_requests : int   = 0
    total_errors   : int   = 0
    total_ms       : float = 0.0

    @property
    def avg_latency_ms(self) -> float | None:
        if self.total_requests == 0:
            return None
        return round(self.total_ms / self.total_requests, 2)

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.total_errors / self.total_requests, 4)


@dataclass
class _ProviderStats:
    total_calls   : int   = 0
    success_calls : int   = 0
    failed_calls  : int   = 0
    fallbacks     : int   = 0
    total_tokens  : int   = 0

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return round(self.success_calls / self.total_calls, 4)


@dataclass
class _StepStats:
    total_runs   : int   = 0
    total_errors : int   = 0
    total_ms     : float = 0.0

    @property
    def avg_latency_ms(self) -> float | None:
        if self.total_runs == 0:
            return None
        return round(self.total_ms / self.total_runs, 2)


# ── Collecteur principal ───────────────────────────────────────────────────────

class MetricsCollector:
    """Collecteur de métriques in-memory thread-safe.

    Singleton accessible via get_metrics(). Toutes les méthodes sont
    thread-safe grâce à un RLock unique.

    Note : les métriques ne survivent pas à un redémarrage du serveur.
    Pour la persistance, la Phase 7+ pourra vider les métriques vers
    Supabase ou une TSDB (TimescaleDB, InfluxDB) à intervalles réguliers.
    """

    def __init__(self) -> None:
        self._lock             : threading.RLock              = threading.RLock()
        self._start_time       : float                        = time.monotonic()
        self._endpoints        : dict[str, _EndpointStats]   = defaultdict(_EndpointStats)
        self._providers        : dict[str, _ProviderStats]   = defaultdict(_ProviderStats)
        self._models           : dict[str, _ProviderStats]   = defaultdict(_ProviderStats)
        self._pipeline_steps   : dict[str, _StepStats]       = defaultdict(_StepStats)

    # ── Requêtes HTTP ──────────────────────────────────────────────────────────

    def record_request(
        self,
        *,
        endpoint   : str,
        duration_ms: float,
        success    : bool,
    ) -> None:
        """Enregistre les métriques d'une requête HTTP terminée.

        Args:
            endpoint    : Chemin de l'endpoint (ex. "/reasoning/analyze").
            duration_ms : Durée totale de traitement en millisecondes.
            success     : True si HTTP status < 400.
        """
        with self._lock:
            s = self._endpoints[endpoint]
            s.total_requests += 1
            s.total_ms       += duration_ms
            if not success:
                s.total_errors += 1

    # ── Appels LLM ────────────────────────────────────────────────────────────

    def record_llm_call(
        self,
        *,
        provider  : str,
        model     : str,
        success   : bool,
        tokens    : int  = 0,
        fallback  : bool = False,
    ) -> None:
        """Enregistre un appel LLM vers un provider.

        Args:
            provider : Nom du provider (ex. "ollama", "groq", "claude").
            model    : Identifiant du modèle (ex. "llama3.2:3b").
            success  : True si l'appel a abouti à une réponse.
            tokens   : Nombre de tokens consommés (0 si inconnu).
            fallback : True si cet appel est le résultat d'un fallback.
        """
        with self._lock:
            p = self._providers[provider]
            p.total_calls += 1
            p.total_tokens += tokens
            if success:
                p.success_calls += 1
            else:
                p.failed_calls += 1
            if fallback:
                p.fallbacks += 1

            m = self._models[model]
            m.total_calls += 1
            m.total_tokens += tokens
            if success:
                m.success_calls += 1
            else:
                m.failed_calls += 1

    # ── Pipeline de raisonnement ───────────────────────────────────────────────

    def record_pipeline_step(
        self,
        *,
        step       : str,
        duration_ms: float,
        success    : bool,
    ) -> None:
        """Enregistre une exécution d'étape du pipeline de raisonnement.

        Args:
            step        : Nom de l'étape (ex. "synthesizer", "decision_engine").
            duration_ms : Durée de l'étape en millisecondes.
            success     : True si l'étape s'est terminée sans exception.
        """
        with self._lock:
            s = self._pipeline_steps[step]
            s.total_runs += 1
            s.total_ms   += duration_ms
            if not success:
                s.total_errors += 1

    # ── Snapshot ───────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Retourne un snapshot complet des métriques sous forme de dict JSON.

        Thread-safe. Aucune métrique n'est réinitialisée.

        Returns:
            Dict avec les clés :
                uptime_seconds : Durée depuis le démarrage du collecteur.
                endpoints      : Métriques par endpoint HTTP.
                providers      : Métriques par provider LLM.
                models         : Métriques par modèle LLM.
                pipeline_steps : Métriques par étape du pipeline.
        """
        with self._lock:
            uptime = round(time.monotonic() - self._start_time, 1)

            endpoints: dict[str, Any] = {}
            for ep, s in self._endpoints.items():
                endpoints[ep] = {
                    "total_requests": s.total_requests,
                    "total_errors"  : s.total_errors,
                    "error_rate"    : s.error_rate,
                    "avg_latency_ms": s.avg_latency_ms,
                }

            providers: dict[str, Any] = {}
            for prov, s in self._providers.items():
                providers[prov] = {
                    "total_calls"  : s.total_calls,
                    "success_calls": s.success_calls,
                    "failed_calls" : s.failed_calls,
                    "fallbacks"    : s.fallbacks,
                    "success_rate" : s.success_rate,
                    "total_tokens" : s.total_tokens,
                }

            models: dict[str, Any] = {}
            for mdl, s in self._models.items():
                models[mdl] = {
                    "total_calls"  : s.total_calls,
                    "success_calls": s.success_calls,
                    "failed_calls" : s.failed_calls,
                    "success_rate" : s.success_rate,
                    "total_tokens" : s.total_tokens,
                }

            steps: dict[str, Any] = {}
            for step, s in self._pipeline_steps.items():
                steps[step] = {
                    "total_runs"    : s.total_runs,
                    "total_errors"  : s.total_errors,
                    "avg_latency_ms": s.avg_latency_ms,
                }

            return {
                "uptime_seconds": uptime,
                "endpoints"     : endpoints,
                "providers"     : providers,
                "models"        : models,
                "pipeline_steps": steps,
            }

    def reset(self) -> None:
        """Réinitialise toutes les métriques (utile pour les tests unitaires)."""
        with self._lock:
            self._endpoints.clear()
            self._providers.clear()
            self._models.clear()
            self._pipeline_steps.clear()
            self._start_time = time.monotonic()


# ── Singleton global ───────────────────────────────────────────────────────────

_global_metrics: MetricsCollector | None = None
_global_metrics_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    """Retourne le singleton global MetricsCollector.

    Thread-safe via double-checked locking. Crée l'instance au premier
    appel et la réutilise pour toute la durée de vie du processus.

    Returns:
        Instance singleton de MetricsCollector.
    """
    global _global_metrics
    if _global_metrics is None:
        with _global_metrics_lock:
            if _global_metrics is None:
                _global_metrics = MetricsCollector()
    return _global_metrics

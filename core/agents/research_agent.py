"""ResearchAgent — moteur de recherche multi-sources — Phase 7.

Implémentation complète :
    - Recherche Wikipedia (API REST, async via aiohttp)
    - Recherche web DuckDuckGo (duckduckgo_search, wrappé asyncio.to_thread)
    - Fusion des sources + déduplication
    - Synthèse LLM via LLMRouter
    - Score de confiance basé sur le nombre et la qualité des sources

DTOs conservés de Phase 6 :
    ResearchDepth, SourceType, ResearchQuery, ResearchSource,
    ResearchResult, ResearchPlan
"""
from __future__ import annotations

import asyncio
import json
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

import aiohttp

from core.agents.base import AgentAutonomy, BaseAgent
from core.agents.models import AgentResult, AgentTask, ExecutionContext
from core.provider_layer.router import get_router


# ── DTOs (inchangés depuis Phase 6) ──────────────────────────────────────────

class ResearchDepth(StrEnum):
    """Niveau de profondeur d'une recherche."""
    QUICK    = "quick"    # 1-3 sources, réponse rapide
    STANDARD = "standard" # 3-7 sources, équilibre vitesse/qualité
    DEEP     = "deep"     # 7-15 sources, analyse exhaustive


class SourceType(StrEnum):
    """Type de source d'information."""
    WEB       = "web"
    WIKIPEDIA = "wikipedia"
    MEMORY    = "memory"
    DOCUMENT  = "document"
    API       = "api"
    UNKNOWN   = "unknown"


@dataclass
class ResearchQuery:
    """Paramètres d'entrée d'une recherche."""
    query       : str
    depth       : ResearchDepth          = ResearchDepth.STANDARD
    max_sources : int                    = 5
    language    : str                    = "fr"
    source_types: list[SourceType]       = field(default_factory=lambda: [
        SourceType.WIKIPEDIA, SourceType.WEB
    ])
    time_range  : str                    = "all"
    context     : str                    = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query"       : self.query,
            "depth"       : self.depth,
            "max_sources" : self.max_sources,
            "language"    : self.language,
            "source_types": [s for s in self.source_types],
            "time_range"  : self.time_range,
        }


@dataclass
class ResearchSource:
    """Une source d'information trouvée lors d'une recherche."""
    title         : str
    snippet       : str
    url           : str                  = ""
    relevance_score: float               = 0.0
    source_type   : SourceType           = SourceType.UNKNOWN
    retrieved_at  : str                  = ""
    metadata      : dict[str, Any]       = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "title"          : self.title,
            "snippet"        : self.snippet,
            "relevance_score": self.relevance_score,
            "source_type"    : self.source_type,
        }
        if self.url:          d["url"]          = self.url
        if self.retrieved_at: d["retrieved_at"] = self.retrieved_at
        if self.metadata:     d["metadata"]     = self.metadata
        return d


@dataclass
class ResearchResult:
    """Résultat agrégé d'une recherche complète."""
    query     : str
    sources   : list[ResearchSource]     = field(default_factory=list)
    summary   : str                      = ""
    confidence: float                    = 0.0
    metadata  : dict[str, Any]           = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query"     : self.query,
            "sources"   : [s.to_dict() for s in self.sources],
            "summary"   : self.summary,
            "confidence": self.confidence,
            **({"metadata": self.metadata} if self.metadata else {}),
        }


@dataclass
class ResearchPlan:
    """Stratégie de recherche avant son exécution."""
    steps            : list[str]          = field(default_factory=list)
    estimated_sources: int                = 0
    estimated_cost   : float              = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps"            : self.steps,
            "estimated_sources": self.estimated_sources,
            "estimated_cost"   : self.estimated_cost,
        }


# ── Constantes ────────────────────────────────────────────────────────────────

_DEPTH_MAX_SOURCES: dict[ResearchDepth, int] = {
    ResearchDepth.QUICK   : 3,
    ResearchDepth.STANDARD: 6,
    ResearchDepth.DEEP    : 12,
}

_WIKI_LANG = {"fr": "fr", "en": "en"}


# ── Agent ─────────────────────────────────────────────────────────────────────

class ResearchAgent(BaseAgent):
    """Agent de recherche multi-sources — Wikipedia + DuckDuckGo + synthèse LLM.

    Capacités :
        - Recherche Wikipedia (API REST, aiohttp async)
        - Recherche web DuckDuckGo (duckduckgo_search, asyncio.to_thread)
        - Fusion et déduplication des sources
        - Synthèse narrative via LLMRouter
        - Score de confiance pondéré par type et quantité de sources

    task.context optionnel :
        depth (str)         : "quick" | "standard" | "deep"
        language (str)      : "fr" | "en"
        max_sources (int)   : Nombre maximum de sources
        source_types (list) : ["wikipedia", "web"]
        extra_context (str) : Contexte additionnel pour la synthèse
    """

    name                : str           = "research_agent"
    description         : str           = (
        "Recherche des informations depuis Wikipedia et le web, "
        "fusionne les sources et génère une synthèse avec score de confiance."
    )
    capabilities        : list[str]     = [
        "research", "search", "web", "wikipedia", "information_retrieval",
    ]
    autonomy            : AgentAutonomy = AgentAutonomy.SANDBOXED_EXECUTE
    version             : str           = "1.0.0"

    cost_per_call       : float         = 1.5
    confidence_threshold: float         = 0.55

    # ── Entrée principale ─────────────────────────────────────────────────

    async def run(self, task: AgentTask, ctx: ExecutionContext) -> AgentResult:
        """Lance la recherche et stocke le résultat dans ctx.shared."""
        t0 = time.monotonic()

        # Lire les paramètres depuis task.context
        raw_depth = task.context.get("depth", ResearchDepth.STANDARD)
        try:
            depth = ResearchDepth(raw_depth)
        except ValueError:
            depth = ResearchDepth.STANDARD

        raw_types = task.context.get("source_types", [SourceType.WIKIPEDIA, SourceType.WEB])
        source_types = []
        for st in raw_types:
            try:
                source_types.append(SourceType(st))
            except ValueError:
                pass
        if not source_types:
            source_types = [SourceType.WIKIPEDIA, SourceType.WEB]

        query = ResearchQuery(
            query       = task.input,
            depth       = depth,
            max_sources = task.context.get("max_sources", _DEPTH_MAX_SOURCES[depth]),
            language    = task.context.get("language", "fr"),
            source_types= source_types,
            context     = task.context.get("extra_context", ""),
        )

        try:
            result = await self.research(query)
        except Exception as exc:
            return self._timed_result(
                task, t0,
                success = False,
                error   = f"Erreur de recherche : {exc}",
            )

        ctx.shared["research_result"] = result.to_dict()

        return self._timed_result(
            task, t0,
            success    = True,
            output     = json.dumps(result.to_dict(), ensure_ascii=False),
            confidence = result.confidence,
            metadata   = {
                "sources_count": len(result.sources),
                "depth"        : query.depth,
                "language"     : query.language,
            },
        )

    # ── Plan ──────────────────────────────────────────────────────────────

    async def plan_research(self, query: ResearchQuery) -> ResearchPlan:
        """Planifie la stratégie de recherche avant exécution."""
        lang_label = "française" if query.language == "fr" else "anglaise"
        steps: list[str] = []

        if SourceType.WIKIPEDIA in query.source_types:
            steps.append(f"Rechercher '{query.query}' sur Wikipedia ({lang_label})")
        if SourceType.WEB in query.source_types:
            steps.append(f"Rechercher '{query.query}' via DuckDuckGo")

        steps.append("Fusionner et dédupliquer les sources")
        steps.append("Générer une synthèse narrative via LLM")

        if query.depth == ResearchDepth.DEEP:
            steps.append("Affiner la recherche sur les sources les plus pertinentes")

        estimated = _DEPTH_MAX_SOURCES.get(query.depth, query.max_sources)
        return ResearchPlan(
            steps            = steps,
            estimated_sources= estimated,
            estimated_cost   = round(estimated * 0.3, 2),
        )

    # ── Recherche complète ────────────────────────────────────────────────

    async def research(self, query: ResearchQuery) -> ResearchResult:
        """Exécute la recherche multi-sources et retourne un ResearchResult."""
        t0  = time.monotonic()
        max_s = _DEPTH_MAX_SOURCES.get(query.depth, query.max_sources)

        # Lancer Wikipedia et DDG en parallèle
        coros = []
        if SourceType.WIKIPEDIA in query.source_types:
            coros.append(self._search_wikipedia(query.query, query.language, max_s))
        if SourceType.WEB in query.source_types:
            coros.append(self._search_ddg(query.query, max_s))

        raw = await asyncio.gather(*coros, return_exceptions=True)

        sources: list[ResearchSource] = []
        for r in raw:
            if isinstance(r, list):
                sources.extend(r)

        sources = _deduplicate(sources)
        sources.sort(key=lambda s: s.relevance_score, reverse=True)
        sources = sources[:max_s]

        if not sources:
            return ResearchResult(
                query    = query.query,
                summary  = "Aucune information trouvée pour cette requête.",
                metadata = {"duration_ms": round((time.monotonic() - t0) * 1000, 1)},
            )

        summary    = await self._summarize(query.query, sources, query.language, query.context)
        confidence = _compute_confidence(sources)

        return ResearchResult(
            query     = query.query,
            sources   = sources,
            summary   = summary,
            confidence= confidence,
            metadata  = {
                "duration_ms"   : round((time.monotonic() - t0) * 1000, 1),
                "depth"         : query.depth,
                "language"      : query.language,
                "sources_found" : len(sources),
            },
        )

    # ── Wikipedia ────────────────────────────────────────────────────────

    async def _search_wikipedia(
        self, query: str, lang: str, limit: int
    ) -> list[ResearchSource]:
        """Recherche Wikipedia via son API REST officielle (async aiohttp)."""
        lang_code = _WIKI_LANG.get(lang, "fr")
        sources: list[ResearchSource] = []
        now = datetime.now(timezone.utc).isoformat()

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:

                # 1. Chercher les titres de pages correspondants
                async with session.get(
                    f"https://{lang_code}.wikipedia.org/w/api.php",
                    params={
                        "action"  : "query",
                        "list"    : "search",
                        "srsearch": query,
                        "srlimit" : min(limit, 4),
                        "format"  : "json",
                        "utf8"    : 1,
                    },
                ) as resp:
                    if resp.status != 200:
                        return sources
                    data = await resp.json(content_type=None)

                titles = [
                    item["title"]
                    for item in data.get("query", {}).get("search", [])
                ]

                # 2. Récupérer le résumé structuré de chaque page
                for title in titles[:min(limit, 3)]:
                    safe_title = urllib.parse.quote(
                        title.replace(" ", "_"), safe=""
                    )
                    async with session.get(
                        f"https://{lang_code}.wikipedia.org"
                        f"/api/rest_v1/page/summary/{safe_title}"
                    ) as r:
                        if r.status != 200:
                            continue
                        s = await r.json(content_type=None)

                    extract = (s.get("extract") or "").strip()
                    if not extract:
                        continue

                    sources.append(ResearchSource(
                        title          = s.get("title", title),
                        snippet        = extract[:700],
                        url            = (
                            s.get("content_urls", {})
                             .get("desktop", {})
                             .get("page", "")
                        ),
                        relevance_score= 0.80,
                        source_type    = SourceType.WIKIPEDIA,
                        retrieved_at   = now,
                    ))

        except Exception:
            pass  # Wikipedia hors ligne → on continue avec ce qu'on a

        return sources

    # ── DuckDuckGo ───────────────────────────────────────────────────────

    async def _search_ddg(self, query: str, limit: int) -> list[ResearchSource]:
        """Recherche web via duckduckgo_search (sync wrappé asyncio.to_thread)."""

        def _sync(q: str, n: int) -> list[dict]:
            try:
                # `ddgs` = successeur officiel de `duckduckgo_search` (renommé)
                try:
                    from ddgs import DDGS  # noqa: PLC0415
                except ImportError:
                    from duckduckgo_search import DDGS  # noqa: PLC0415
                with DDGS() as ddgs:
                    return list(ddgs.text(q, max_results=n))
            except Exception:
                return []

        raw = await asyncio.to_thread(_sync, query, min(limit, 6))
        now = datetime.now(timezone.utc).isoformat()

        sources: list[ResearchSource] = []
        for item in raw:
            snippet = (item.get("body") or "").strip()
            if not snippet:
                continue
            sources.append(ResearchSource(
                title          = (item.get("title") or "")[:120],
                snippet        = snippet[:700],
                url            = item.get("href", ""),
                relevance_score= 0.65,
                source_type    = SourceType.WEB,
                retrieved_at   = now,
            ))

        return sources

    # ── Synthèse LLM ─────────────────────────────────────────────────────

    async def _summarize(
        self,
        query: str,
        sources: list[ResearchSource],
        lang: str,
        extra_context: str = "",
    ) -> str:
        """Synthétise les sources en un paragraphe via LLMRouter."""
        snippets = "\n\n".join(
            f"[{i+1}] {s.title} ({s.source_type})\n{s.snippet[:500]}"
            for i, s in enumerate(sources[:6])
        )
        lang_instr = "en français" if lang == "fr" else "in English"
        context_block = f"\nContexte supplémentaire : {extra_context}\n" if extra_context else ""

        prompt = (
            f"Question : {query}\n"
            f"{context_block}\n"
            f"Sources trouvées :\n{snippets}\n\n"
            f"Synthétise ces informations {lang_instr} en 3 à 5 phrases "
            "claires et directes. Commence directement par la réponse."
        )
        try:
            result = await get_router().generate(prompt=prompt)
            return result or _fallback_summary(sources)
        except Exception:
            return _fallback_summary(sources)


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _deduplicate(sources: list[ResearchSource]) -> list[ResearchSource]:
    """Élimine les doublons par titre normalisé."""
    seen: set[str] = set()
    out: list[ResearchSource] = []
    for s in sources:
        key = s.title.lower().strip()[:80]
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _compute_confidence(sources: list[ResearchSource]) -> float:
    """Score 0.0–1.0 pondéré par type de source et quantité."""
    if not sources:
        return 0.0
    type_weight = {SourceType.WIKIPEDIA: 1.0, SourceType.WEB: 0.72}
    weighted_avg = sum(
        s.relevance_score * type_weight.get(s.source_type, 0.5)
        for s in sources
    ) / len(sources)
    count_bonus = min(len(sources) / 5, 1.0) * 0.15
    return round(min(weighted_avg + count_bonus, 1.0), 2)


def _fallback_summary(sources: list[ResearchSource]) -> str:
    """Retourne le premier extrait si le LLM est indisponible."""
    return sources[0].snippet[:400] if sources else "Synthèse indisponible."

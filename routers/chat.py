"""
Chat hybride — Mémoire + Graphe + LLM + Historique persistant.
Auto-détection de domaine et auto-alimentation en arrière-plan.
"""
import asyncio
import json
import logging
import os
import random

import httpx

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from core.llm import generate, check_ollama_status
from core.memory import search_memory
from core.providers import groq_generate, detect_provider
from core.config import settings
from core.chat_history import add_message, create_session, generate_smart_topic, get_session_for_user
from core.auth import require_chat_user
from core.consciousness import consciousness
from core.system_prompts import build_system_prompt
from models.user import User, UserRole

router = APIRouter()
logger = logging.getLogger("makenbrain.chat")


# ── Fast Conversation Layer ────────────────────────────────────────────────────
# Détecte les messages conversationnels simples et retourne une réponse
# immédiate, sans appeler le LLM ni le pipeline de raisonnement.

_FAST_REPLIES: dict[str, list[str]] = {
    "greetings": [
        "Salut ! 😊\nSur quoi on travaille aujourd'hui ?",
        "Hey ! Qu'est-ce qu'on fait aujourd'hui ?",
        "Salut ! Content de te revoir.\nOn se lance ?",
        "Bonjour ! Tu veux qu'on attaque quoi ?",
        "Hey, re ! On reprend où on en était ?",
    ],
    "thanks": [
        "Avec plaisir !",
        "De rien, c'est pour ça que je suis là.",
        "Pas de problème !",
        "Toujours.",
        "C'est normal.",
    ],
    "ok": [
        "Parfait.",
        "Ok, nickel.",
        "Reçu.",
        "C'est noté.",
        "On y va.",
    ],
    "howru": [
        "Bien ! Et toi, quoi de neuf ?",
        "En forme. Tu travailles sur quoi en ce moment ?",
        "Top ! On attaque quoi ?",
        "Bien. Et toi ?",
    ],
    "bye": [
        "À bientôt !",
        "Bonne continuation !",
        "À plus !",
        "Bonne journée !",
        "Prends soin de toi.",
    ],
    "goodnight": [
        "Bonne nuit !",
        "Dors bien.",
        "Bonne nuit, on reprend demain.",
    ],
    "yes": [
        "Ok !",
        "Parfait.",
        "Noté.",
        "Go.",
    ],
    "no": [
        "Ok, pas de problème.",
        "Compris.",
        "Noté.",
        "Pas de souci.",
    ],
}

_FAST_PATTERNS: list[tuple[frozenset[str], str]] = [
    (frozenset(["salut", "hello", "bonjour", "bonsoir", "hey", "coucou", "hi", "allo", "yo"]), "greetings"),
    (frozenset(["merci", "thanks", "thank you", "thx", "merci beaucoup"]), "thanks"),
    (frozenset(["ok", "okay", "d'accord", "vu", "compris", "reçu", "parfait", "super", "cool", "nickel"]), "ok"),
    (frozenset(["ça va", "ca va", "comment tu vas", "comment ça va", "tu vas bien", "ça roule"]), "howru"),
    (frozenset(["bonne nuit", "bonne nuit !", "dors bien"]), "goodnight"),
    (frozenset(["au revoir", "bye", "à bientôt", "bonne soirée", "à plus", "ciao", "tchao"]), "bye"),
    (frozenset(["oui", "yes", "ouais", "yep", "yup", "mouais"]), "yes"),
    (frozenset(["non", "no", "nope", "nan", "pas vraiment"]), "no"),
]


def _fast_reply(message: str) -> str | None:
    """Retourne une réponse immédiate si le message est une conversation simple.

    Critères : message court (≤ 4 mots) ET correspondance exacte avec un
    pattern connu. Retourne None si le pipeline complet est nécessaire.
    """
    stripped = message.strip()
    if len(stripped.split()) > 4:
        return None
    normalized = stripped.lower().rstrip("!?. ")
    for patterns, category in _FAST_PATTERNS:
        if normalized in patterns:
            return random.choice(_FAST_REPLIES[category])
    return None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None     # Si fourni, sauvegarde dans l'historique
    use_memory: bool = True
    use_graph: bool = True
    n_context: int = 5
    relevance_threshold: float = 0.75
    provider: str = "auto"
    stream: bool = False

    model_config = {"json_schema_extra": {"example": {
        "message": "Explique-moi comment fonctionne un contrôleur PID en robotique",
        "session_id": "a3f9c2d1",
        "use_memory": True,
        "use_graph": True,
        "provider": "auto",
        "stream": False
    }}}


class ChatResponse(BaseModel):
    response: str
    confidence: float = 0.0
    risk_level: str = "high"
    suggested_sources: list[dict[str, str]] = Field(default_factory=list)
    memories_used: int
    graph_concepts: list[str]
    model: str
    provider: str
    session_id: Optional[str] = None
    domain_detected: Optional[str] = None
    auto_learning: bool = False
    context_preview: Optional[list[str]] = None
    web_sources: list[dict] = Field(default_factory=list)
    web_images: list[dict] = Field(default_factory=list)


async def _detect_domain_and_research(message: str) -> Optional[str]:
    """Détecte le domaine et lance une recherche autonome si besoin. Arrière-plan."""
    try:
        from core.extractor import extract_fast
        from core.domain_explorer import explore_domain, DOMAIN_CATALOG

        concepts = extract_fast(message)
        concept_names = set(c["name"].lower() for c in concepts)
        best_domain, best_score = None, 0

        for domain_key, domain_data in DOMAIN_CATALOG.items():
            score = 0
            if any(w in domain_key for w in message.lower().split()):
                score += 2
            for subtopic in domain_data.get("subtopics", []):
                if subtopic.lower() in message.lower():
                    score += 1
            for c in concept_names:
                if c in domain_key:
                    score += 1
            if score > best_score:
                best_score, best_domain = score, domain_key

        if not best_domain or best_score == 0:
            return None

        existing = await search_memory(best_domain, n_results=5)
        domain_memories = [m for m in existing if m.get("distance", 1) < 0.6]

        if len(domain_memories) < 3:
            logger.info("Auto-recherche domaine detecte : %s", best_domain)
            await explore_domain(best_domain, depth="rapide")

        return best_domain
    except Exception:
        return None


from core.user_profile import user_profile
from core.project_memory import project_manager

# Nombre de messages précédents injectés dans le contexte du modèle et
# taille maximale de chaque message (protège la fenêtre de contexte).
_HISTORY_MAX_MESSAGES = 10
_HISTORY_MAX_CHARS_PER_MESSAGE = 2000


# ── Recherche web ancrée (façon Perplexity) ──────────────────────────────────
# Quand la question est factuelle ou d'actualité, une VRAIE recherche web est
# faite en temps réel et injectée dans le contexte. Le modèle a l'interdiction
# de citer autre chose que ces sources — fin des références inventées.
_WEB_SEARCH_TRIGGERS = (
    "actuel", "actualité", "aujourd'hui", "récent", "récente", "dernier", "dernière",
    "2024", "2025", "2026", "source", "prouve", "preuve",
    "statistique", "marché", "prix de", "combien coûte", "qui est", "c'est qui",
    "news", "nouveauté", "tendance", "classement", "milliardaire", "vérifie",
    "cherche", "recherche sur", "article", "investissement", "parle moi de",
    "parle-moi de", "entreprise", "qui sont",
)


def _needs_web_search(message: str) -> bool:
    """Heuristique rapide par mots-clés (chemin sans latence)."""
    lowered = message.lower()
    return any(trigger in lowered for trigger in _WEB_SEARCH_TRIGGERS)


async def _classify_needs_web(message: str) -> bool:
    """Détection intelligente du besoin de recherche web.

    1. Mots-clés d'abord (0 latence). 2. Sinon, micro-classifieur LLM sur le
    modèle rapide (~300 ms) : couvre toutes les formulations humaines que des
    mots-clés ne peuvent pas prévoir. Défaut NON en cas d'échec.
    """
    if _needs_web_search(message):
        return True
    try:
        from core.providers import groq_generate, GROQ_FAST

        verdict = await asyncio.wait_for(
            groq_generate(
                f"Question : {message}\n\n"
                "Cette question porte-t-elle sur des faits vérifiables, des personnes, "
                "des entreprises, des événements réels ou des informations d'actualité "
                "qui gagneraient à être appuyés par des sources web ? "
                "Réponds UNIQUEMENT par OUI ou NON.",
                "",
                model=GROQ_FAST,
                system_prompt="Tu es un classifieur binaire. Réponds uniquement OUI ou NON.",
            ),
            timeout=6,
        )
        return "OUI" in str(verdict).upper()
    except Exception:
        return False


async def _expand_queries(message: str) -> list[str]:
    """Génère 2 requêtes de recherche complémentaires (angles différents).

    C'est ce qui transforme une recherche simple en recherche APPROFONDIE :
    le sujet est couvert sous plusieurs angles au lieu d'une seule requête.
    Best-effort : échec → aucune requête supplémentaire.
    """
    try:
        from core.providers import groq_generate, GROQ_FAST

        raw = await asyncio.wait_for(
            groq_generate(
                f"Sujet : {message[:300]}\n\n"
                "Génère 2 requêtes de recherche web courtes et complémentaires "
                "(angles différents : actualité récente, chiffres/faits, contexte) "
                "pour documenter ce sujet en profondeur. Une par ligne, sans numérotation.",
                "",
                model=GROQ_FAST,
                system_prompt="Tu génères des requêtes de recherche web. Réponds uniquement avec les requêtes, une par ligne.",
            ),
            timeout=8,
        )
        return [q.strip("-•* ").strip() for q in str(raw).strip().splitlines() if q.strip()][:2]
    except Exception:
        return []


async def _deep_web_research(message: str) -> tuple[list[dict], list[dict]]:
    """Recherche approfondie : requête originale + requêtes complémentaires,
    toutes lancées en PARALLÈLE sur tous les moteurs (DDGS + Wikipédia).

    Résultat : jusqu'à 10 sources uniques et 8 images — la matière d'une
    réponse structurée et citée, pas d'un simple chat.
    """
    queries = [message[:200]] + await _expand_queries(message)
    results = await asyncio.gather(*[_web_search_sources(q) for q in queries])
    sources: list[dict] = []
    images: list[dict] = []
    seen_urls: set[str] = set()
    seen_imgs: set[str] = set()
    for source_list, image_list in results:
        for s in source_list:
            if s["url"] not in seen_urls:
                seen_urls.add(s["url"])
                sources.append(s)
        for img in image_list:
            if img["thumbnail"] not in seen_imgs:
                seen_imgs.add(img["thumbnail"])
                images.append(img)
    return sources[:10], images[:8]


async def _ddgs_search(
    query: str,
    max_results: int = 5,
    with_images: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Recherche DuckDuckGo/multi-moteurs via `ddgs`. Best-effort."""
    def _search() -> tuple[list[dict], list[dict]]:
        # `ddgs` est le successeur officiel de `duckduckgo_search` (renommé) :
        # l'ancien paquet retourne des résultats vides et des rate-limits.
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        sources: list[dict] = []
        images: list[dict] = []
        with DDGS(timeout=10) as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                url = r.get("href") or r.get("url") or ""
                if not url:
                    continue
                sources.append({
                    "title": r.get("title", "") or url,
                    "url": url,
                    "snippet": (r.get("body", "") or "")[:300],
                })
            if with_images and sources:
                try:
                    for img in ddgs.images(query, max_results=4):
                        if img.get("thumbnail"):
                            images.append({
                                "title": img.get("title", ""),
                                "thumbnail": img.get("thumbnail", ""),
                                "image": img.get("image", ""),
                                "url": img.get("url", ""),
                            })
                except Exception:
                    pass  # les images sont un bonus, jamais bloquantes
        return sources, images

    try:
        return await asyncio.wait_for(asyncio.to_thread(_search), timeout=25)
    except Exception as exc:
        logger.warning("DDGS indisponible pour le grounding : %s: %s", type(exc).__name__, exc)
        return [], []


async def _wikipedia_search(query: str, limit: int = 4) -> tuple[list[dict], list[dict]]:
    """Recherche Wikipédia (API gratuite, CDN mondial — fiable sur réseau lent).

    Retourne des sources encyclopédiques avec extraits, URLs canoniques et
    vignettes d'images officielles. Essaie le français puis l'anglais.
    """
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": query, "gsrlimit": limit,
        "prop": "pageimages|extracts|info", "inprop": "url",
        "piprop": "thumbnail", "pithumbsize": 320,
        "exintro": 1, "explaintext": 1, "exchars": 300,
        "redirects": 1,
    }
    sources: list[dict] = []
    images: list[dict] = []
    headers = {"User-Agent": "MakenBrain/1.0 (https://github.com/leonelmaken/makenbrain)"}
    try:
        async with httpx.AsyncClient(timeout=12, headers=headers, follow_redirects=True) as client:
            for lang in ("fr", "en"):
                # Chaque langue est isolée : une erreur sur le français
                # (proxy FAI qui renvoie du HTML, timeout…) ne doit jamais
                # empêcher la tentative en anglais.
                try:
                    resp = await client.get(f"https://{lang}.wikipedia.org/w/api.php", params=params)
                    if resp.status_code != 200:
                        logger.warning("Wikipedia %s → HTTP %s", lang, resp.status_code)
                        continue
                    try:
                        data = resp.json()
                    except Exception:
                        # Réponse non-JSON = interception proxy/page d'erreur HTML.
                        logger.warning("Wikipedia %s → réponse non-JSON (proxy/interception ?)", lang)
                        continue
                    pages = ((data.get("query") or {}).get("pages") or {})
                    ranked = sorted(pages.values(), key=lambda p: p.get("index", 99))
                    for p in ranked:
                        title = p.get("title", "")
                        url = p.get("fullurl") or f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
                        sources.append({
                            "title": f"{title} — Wikipédia",
                            "url": url,
                            "snippet": (p.get("extract") or "")[:300],
                        })
                        thumb = (p.get("thumbnail") or {}).get("source")
                        if thumb:
                            images.append({"title": title, "thumbnail": thumb, "image": thumb, "url": url})
                    if sources:
                        break  # cette langue a répondu — inutile de continuer
                except Exception as exc:
                    logger.warning("Wikipedia %s indisponible : %s: %s", lang, type(exc).__name__, exc)
    except Exception as exc:
        logger.warning("Wikipedia indisponible pour le grounding : %s: %s", type(exc).__name__, exc)
    return sources, images


async def _brave_search(query: str, max_results: int = 5) -> tuple[list[dict], list[dict]]:
    """Brave Search API — 2 000 requêtes/mois gratuites (clé BRAVE_API_KEY dans .env).

    Excellente qualité d'actualité, infrastructure indépendante — précieux
    quand DuckDuckGo est bloqué ou lent sur le réseau local.
    """
    key = os.getenv("BRAVE_API_KEY", "")
    if not key:
        return [], []
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": max_results},
                headers={"X-Subscription-Token": key, "Accept": "application/json"},
            )
            if resp.status_code != 200:
                logger.warning("Brave Search → HTTP %s", resp.status_code)
                return [], []
            results = ((resp.json().get("web") or {}).get("results") or [])
            sources = [
                {
                    "title": r.get("title", "") or r.get("url", ""),
                    "url": r.get("url", ""),
                    "snippet": (r.get("description", "") or "")[:300],
                }
                for r in results if r.get("url")
            ]
            return sources, []
    except Exception as exc:
        logger.warning("Brave indisponible : %s: %s", type(exc).__name__, exc)
        return [], []


async def _tavily_search(query: str, max_results: int = 5) -> tuple[list[dict], list[dict]]:
    """Tavily API — 1 000 requêtes/mois gratuites (clé TAVILY_API_KEY dans .env).

    Moteur conçu pour les IA : résultats pré-nettoyés + images liées au sujet.
    """
    key = os.getenv("TAVILY_API_KEY", "")
    if not key:
        return [], []
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": key, "query": query,
                    "max_results": max_results, "include_images": True,
                },
            )
            if resp.status_code != 200:
                logger.warning("Tavily → HTTP %s", resp.status_code)
                return [], []
            data = resp.json()
            sources = [
                {
                    "title": r.get("title", "") or r.get("url", ""),
                    "url": r.get("url", ""),
                    "snippet": (r.get("content", "") or "")[:300],
                }
                for r in data.get("results", []) if r.get("url")
            ]
            images = [
                {"title": "", "thumbnail": u, "image": u, "url": u}
                for u in (data.get("images") or [])[:4] if isinstance(u, str)
            ]
            return sources, images
    except Exception as exc:
        logger.warning("Tavily indisponible : %s: %s", type(exc).__name__, exc)
        return [], []


async def _web_search_sources(
    query: str,
    max_results: int = 5,
    with_images: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Grounding multi-moteurs : Brave + Tavily + DuckDuckGo + Wikipédia en PARALLÈLE.

    Chaque moteur est best-effort et indépendant : il suffit qu'UN SEUL
    réponde pour que le chat ait des preuves. Brave et Tavily ne s'activent
    que si leur clé gratuite est présente dans .env. Priorité de fusion :
    Brave/Tavily (qualité), DuckDuckGo (couverture), Wikipédia (fiabilité
    + images officielles).
    """
    results = await asyncio.gather(
        _brave_search(query, max_results),
        _tavily_search(query, max_results),
        _ddgs_search(query, max_results, with_images),
        _wikipedia_search(query),
    )
    sources: list[dict] = []
    images: list[dict] = []
    seen_urls: set[str] = set()
    for source_list, image_list in results:
        for s in source_list:
            if s["url"] and s["url"] not in seen_urls:
                seen_urls.add(s["url"])
                sources.append(s)
        images.extend(image_list)
    if not sources:
        logger.warning("Grounding sans résultat : tous les moteurs indisponibles.")
    return sources[:7], images[:6]


_GROUNDING_RULES = (
    "\n\nSOURCES WEB : des résultats de recherche web RÉELS et ACTUELS te sont "
    "fournis dans le contexte. Règles STRICTES :\n"
    "- Appuie chaque affirmation factuelle sur ces sources en les citant par numéro : [1], [2]…\n"
    "- Ne mentionne JAMAIS un article, un titre, une étude ou une URL absents de cette liste.\n"
    "- Si les sources fournies ne suffisent pas pour répondre, dis-le explicitement au lieu d'inventer.\n"
    "- Structure ta réponse en markdown : titres (##), listes, tableau comparatif si pertinent. "
    "Produis une synthèse APPROFONDIE et organisée, pas un résumé superficiel."
)


def _build_conversation_history(session_id: str | None, user_id: str) -> list[dict]:
    """Construit l'historique de conversation pour le LLM.

    Retourne les derniers messages de la session (rôle user/assistant),
    uniquement si la session appartient bien à l'utilisateur courant.
    Best-effort : toute erreur retourne un historique vide sans casser le chat.
    """
    if not session_id:
        return []
    try:
        session = get_session_for_user(session_id, user_id)
        if not session:
            return []
        history: list[dict] = []
        for message in session.get("messages", [])[-_HISTORY_MAX_MESSAGES:]:
            role = message.get("role")
            content = str(message.get("content", ""))
            if role not in ("user", "assistant") or not content:
                continue
            history.append({"role": role, "content": content[:_HISTORY_MAX_CHARS_PER_MESSAGE]})
        return history
    except Exception:
        return []

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_chat_user),
):
    """
    Chat intelligent avec mémoire persistante, contexte utilisateur et projets.
    """
    # ── Session auto-créée si absente ────────────────────────────────────────
    # Sans session, rien n'était persisté (pas d'historique, pas de titre).
    # Chaque conversation est désormais rattachée à une session dès le
    # premier message, comme sur ChatGPT/Claude. Le frontend récupère
    # l'identifiant via ChatResponse.session_id.
    if not request.session_id:
        request.session_id = create_session(user_id=str(current_user.id))

    # ── Fast Conversation Layer ──────────────────────────────────────────────
    fast = _fast_reply(request.message)
    if fast is not None:
        if request.session_id:
            add_message(request.session_id, "user", request.message, str(current_user.id))
            add_message(request.session_id, "assistant", fast, str(current_user.id))
        return ChatResponse(
            response=fast,
            confidence=1.0,
            risk_level="low",
            suggested_sources=[],
            memories_used=0,
            graph_concepts=[],
            model="fast-reply",
            provider="local",
            session_id=request.session_id,
        )

    context_parts, memories_used, graph_concepts, context_preview = [], 0, [], []

    # 1. INJECTION DU PROFIL UTILISATEUR & PROJETS — SuperAdmin uniquement
    # Le profil et les projets locaux appartiennent au SuperAdmin.
    # Les autres utilisateurs n'ont accès qu'à leur mémoire personnelle (ci-dessous).
    if current_user.role == UserRole.SUPERADMIN:
        user_summary = user_profile.get_summary()
        projects_summary = project_manager.get_projects_summary()
        context_parts.append(f"--- IDENTITÉ UTILISATEUR ---\n{user_summary}")
        context_parts.append(f"--- CONTEXTE PROJETS ---\n{projects_summary}")

    # 1.b. INJECTION MEMOIRE UTILISATEUR PERSONNALISEE
    user_memory_context = consciousness.get_user_memory_context(
        str(current_user.id),
        request.message,
    )
    user_memory_count = 0
    if user_memory_context:
        user_memory_count = len(user_memory_context.splitlines())
        context_parts.append(f"--- MEMOIRE UTILISATEUR ---\n{user_memory_context}")
        context_preview.extend(
            [
                line[:100] + "..." if len(line) > 100 else line
                for line in user_memory_context.splitlines()
            ]
        )

    # 2. RECHERCHE MÉMOIRE VECTORIELLE
    if request.use_memory:
        memories = await search_memory(request.message, n_results=request.n_context)
        relevant = [m for m in memories if m["distance"] < request.relevance_threshold]
        if relevant:
            for i, m in enumerate(relevant):
                context_parts.append(f"[Souvenir {i+1}] {m['content']}")
                preview = m["content"][:100] + "..." if len(m["content"]) > 100 else m["content"]
                context_preview.append(preview)
            memories_used = len(relevant)

    if request.use_graph:
        try:
            from core.graph import neuron_graph
            from core.extractor import extract_fast
            concepts = extract_fast(request.message)
            for c in concepts[:3]:
                name = c.get("name", "").lower()
                if not name or len(name) < 3:
                    continue
                result = neuron_graph.explore(name, depth=2)
                if not result.get("found"):
                    continue
                neighbors = result.get("neighbors", {})
                direct = [n for n, info in neighbors.items()
                          if info["distance"] == 1 and info["strength"] > 0.5][:5]
                if direct:
                    context_parts.append(f"[Graphe] '{name}' → {', '.join(direct)}")
                    graph_concepts.extend(direct)
        except Exception:
            pass

    # ── RECHERCHE WEB APPROFONDIE (sources réelles, façon Perplexity) ────────
    web_sources: list[dict] = []
    web_images: list[dict] = []
    web_attempted = await _classify_needs_web(request.message)
    if web_attempted:
        web_sources, web_images = await _deep_web_research(request.message)
        if web_sources:
            source_lines = []
            for i, s in enumerate(web_sources, start=1):
                source_lines.append(f"[{i}] {s['title']}\n    URL : {s['url']}\n    Extrait : {s['snippet']}")
            context_parts.append(
                "--- SOURCES WEB (recherche temps réel — SEULES sources citables) ---\n"
                + "\n".join(source_lines)
            )

    context = "\n\n".join(context_parts)
    chosen  = detect_provider(request.message, request.provider)
    # Prompt système construit dynamiquement selon le rôle :
    # SuperAdmin → prompt personnel ; tout autre utilisateur → prompt générique.
    system_prompt = build_system_prompt(current_user.role)
    if web_sources:
        system_prompt += _GROUNDING_RULES
    elif web_attempted:
        # La recherche a été tentée mais AUCUN moteur n'a répondu (réseau).
        # Sans cette règle, le modèle fabrique une section "Sources" avec des
        # URLs plausibles mais inventées dès que l'utilisateur insiste.
        system_prompt += (
            "\n\nRECHERCHE WEB ÉCHOUÉE : la recherche de sources n'a rien retourné "
            "(problème réseau temporaire). Tu n'as donc AUCUNE source vérifiable. "
            "INTERDICTION ABSOLUE de produire une section Sources, des références, "
            "des URLs ou des citations [n] — même si l'utilisateur en demande. "
            "Réponds avec tes connaissances générales, indique clairement au début "
            "que les sources n'ont pas pu être récupérées cette fois, et propose "
            "de reposer la question dans quelques minutes."
        )

    # ── Mémoire conversationnelle ────────────────────────────────────────────
    # Injecte les derniers échanges de la session pour que le modèle ait le
    # fil de la conversation (comme ChatGPT/Claude). Les messages sont
    # sauvegardés APRÈS génération : l'historique ne contient donc que les
    # tours précédents, jamais le message courant.
    conversation_history = _build_conversation_history(request.session_id, str(current_user.id))

    # DÉTECTION DOMAINE (Arrière-plan)
    background_tasks.add_task(_detect_domain_and_research, request.message)

    if request.stream:
        return StreamingResponse(
            chat_streamer(
                request,
                context,
                chosen,
                background_tasks,
                str(current_user.id),
                user_memory_count,
                memories_used,
                len(graph_concepts),
                system_prompt,
                conversation_history,
            ),
            media_type="text/event-stream"
        )

    # MODE NORMAL (Non-streaming)
    if chosen == "groq":
        response_text = await groq_generate(
            request.message, context,
            system_prompt=system_prompt, history=conversation_history,
        )
        model_name    = "llama-3.3-70b-versatile (Groq)"
    else:
        response_text = await generate(
            request.message, context,
            system_prompt=system_prompt, history=conversation_history,
        )
        model_name    = settings.OLLAMA_MODEL + " (local)"

    response_evaluation = consciousness.evaluate_response(
        question=request.message,
        answer=response_text,
        user_memory_count=user_memory_count,
        supabase_data_count=user_memory_count,
        vector_memory_count=memories_used,
        graph_context_count=len(graph_concepts),
    )
    response_text = response_evaluation["answer"]

    # Sauvegarder dans l'historique de conversation (sources incluses pour
    # que les preuves soient ré-affichées au rechargement de la session)
    if request.session_id:
        add_message(request.session_id, "user", request.message, str(current_user.id))
        assistant_extras = (
            {"web_sources": web_sources, "web_images": web_images}
            if (web_sources or web_images) else None
        )
        add_message(request.session_id, "assistant", response_text, str(current_user.id),
                    extras=assistant_extras)
        background_tasks.add_task(generate_smart_topic, request.session_id)

    return ChatResponse(
        response        = response_text,
        confidence      = response_evaluation["confidence"],
        risk_level      = response_evaluation["risk_level"],
        suggested_sources = response_evaluation["suggested_sources"],
        memories_used   = memories_used,
        graph_concepts  = list(set(graph_concepts))[:10],
        model           = model_name,
        provider        = chosen,
        session_id      = request.session_id,
        auto_learning   = True,
        context_preview = context_preview if context_preview else None,
        web_sources     = web_sources,
        web_images      = web_images,
    )


async def chat_streamer(
    request: ChatRequest,
    context: str,
    chosen: str,
    background_tasks: BackgroundTasks,
    user_id: str,
    user_memory_count: int,
    memories_used: int,
    graph_context_count: int,
    system_prompt: str | None = None,
    conversation_history: list[dict] | None = None,
):
    """Générateur SSE pour le streaming token-par-token."""
    full_response = ""

    if chosen == "groq":
        stream = await groq_generate(request.message, context, system_prompt=system_prompt,
                                     history=conversation_history, stream=True)
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"
    else:
        stream = await generate(request.message, context, system_prompt=system_prompt,
                                history=conversation_history, stream=True)
        for chunk in stream:
            token = chunk['message']['content']
            full_response += token
            yield f"data: {json.dumps({'token': token})}\n\n"

    response_evaluation = consciousness.evaluate_response(
        question=request.message,
        answer=full_response,
        user_memory_count=user_memory_count,
        supabase_data_count=user_memory_count,
        vector_memory_count=memories_used,
        graph_context_count=graph_context_count,
    )
    yield (
        "data: "
        + json.dumps(
            {
                "confidence": response_evaluation["confidence"],
                "risk_level": response_evaluation["risk_level"],
                "suggested_sources": response_evaluation["suggested_sources"],
                "final_response": response_evaluation["answer"],
            }
        )
        + "\n\n"
    )

    # Fin du stream
    yield "data: [DONE]\n\n"

    # Sauvegarde historique (une fois le stream fini)
    if request.session_id:
        add_message(request.session_id, "user", request.message, user_id)
        add_message(request.session_id, "assistant", response_evaluation["answer"], user_id)
        background_tasks.add_task(generate_smart_topic, request.session_id)


@router.get("/status")
async def status():
    return await check_ollama_status()

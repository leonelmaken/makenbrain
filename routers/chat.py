"""
Chat hybride — Mémoire + Graphe + LLM + Historique persistant.
Auto-détection de domaine et auto-alimentation en arrière-plan.
"""
import json
import logging
import random

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from core.llm import generate, check_ollama_status
from core.memory import search_memory
from core.providers import groq_generate, detect_provider
from core.config import settings
from core.chat_history import add_message, generate_smart_topic
from core.auth import require_chat_user
from core.consciousness import consciousness
from models.user import User

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

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_chat_user),
):
    """
    Chat intelligent avec mémoire persistante, contexte utilisateur et projets.
    """
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

    # 1. INJECTION DU PROFIL UTILISATEUR & PROJETS (Phase 1)
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

    context = "\n\n".join(context_parts)
    chosen  = detect_provider(request.message, request.provider)

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
            ),
            media_type="text/event-stream"
        )

    # MODE NORMAL (Non-streaming)
    if chosen == "groq":
        response_text = await groq_generate(request.message, context)
        model_name    = "llama-3.3-70b-versatile (Groq)"
    else:
        response_text = await generate(request.message, context)
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

    # Sauvegarder dans l'historique de conversation
    if request.session_id:
        add_message(request.session_id, "user", request.message, str(current_user.id))
        add_message(request.session_id, "assistant", response_text, str(current_user.id))
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
):
    """Générateur SSE pour le streaming token-par-token."""
    full_response = ""
    
    if chosen == "groq":
        stream = await groq_generate(request.message, context, stream=True)
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"
    else:
        stream = await generate(request.message, context, stream=True)
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

"""
Connecteurs externes — Phase 2.2
- Groq : Llama 3.3 70B gratuit, ultra-rapide, 20x plus puissant que le modèle local
- Wikipedia : connaissance encyclopédique automatique, sans clé API
- Routage intelligent : le cerveau choisit seul le bon outil
"""
import logging
import httpx
from groq import Groq as GroqClient
from core.config import settings
from core.audit import audit_event

logger = logging.getLogger("makenbrain.providers")

# ── Constantes ────────────────────────────────────────────────────────────────

GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_FAST    = "llama-3.1-8b-instant"     # fallback ultra-rapide
GROQ_VISION  = "meta-llama/llama-4-scout-17b-16e-instruct"  # multimodal (analyse d'images)

# Mots-clés qui déclenchent le mode Groq en auto
COMPLEX_KEYWORDS = {
    "analyse", "analyser", "analysez", "compare", "comparer",
    "explique", "expliquer", "comment", "pourquoi", "architecture",
    "stratégie", "recommande", "recommander", "meilleure", "optimise",
    "optimiser", "solution", "problème", "debug", "erreur", "implémente",
    "implémenter", "conçois", "concevoir", "planifie", "planifier",
}

# Prompt générique importé depuis la source unique — jamais de données
# personnelles dans un prompt par défaut (voir core/system_prompts.py).
from core.system_prompts import SYSTEM_PROMPT


# ── Routage intelligent ───────────────────────────────────────────────────────

def should_use_groq(message: str) -> bool:
    """
    Décide si la question mérite Groq (70B) plutôt que le modèle local (3B).
    Critères : longueur > 15 mots OU présence de mots-clés complexes.
    """
    words = message.lower().split()
    if len(words) > 15:
        return True
    return any(kw in message.lower() for kw in COMPLEX_KEYWORDS)


def detect_provider(message: str, requested: str = "auto") -> str:
    """
    Retourne le provider à utiliser. 
    Priorise le Cloud (Groq/Gemini) si configuré pour éviter les pannes locales.
    """
    if requested == "local":
        return "local"
    if requested == "groq" or requested == "gemini":
        return "groq" if settings.GROQ_API_KEY else "local"
    
    # En mode auto : si Groq est configuré, on l'utilise pour garantir la puissance et la disponibilité
    if settings.GROQ_API_KEY:
        return "groq"
        
    return "local"


# ── Groq ──────────────────────────────────────────────────────────────────────

async def groq_vision(prompt: str, image_data_url: str, system_prompt: str = None) -> str:
    """Analyse une image via le modèle multimodal Groq (gratuit).

    `image_data_url` : data URL base64 (data:image/jpeg;base64,...) envoyée
    par le frontend. Le modèle voit l'image et répond à la question posée.
    """
    client = _groq_client()
    current_system = system_prompt or SYSTEM_PROMPT
    logger.info("Groq vision request model=%s prompt_chars=%s image_chars=%s",
                GROQ_VISION, len(prompt), len(image_data_url))
    response = client.chat.completions.create(
        model=GROQ_VISION,
        messages=[
            # Note : les modèles vision Groq n'acceptent pas de message
            # system avec des images — instructions fusionnées dans le texte.
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{current_system}\n\n---\n\n{prompt}"},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        temperature=0.5,
        max_tokens=2048,
    )
    return response.choices[0].message.content or ""


def _groq_client() -> GroqClient:
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY manquant dans .env — ajoute ta clé Groq.")
    return GroqClient(api_key=settings.GROQ_API_KEY)


async def groq_generate(
    prompt: str,
    context: str = "",
    model: str = GROQ_MODEL,
    system_prompt: str = None,
    stream: bool = False,
    history: list[dict] | None = None,
):
    """
    Génère une réponse via Groq (Llama 3.3 70B).
    `history` : messages précédents de la conversation ({role, content}),
    injectés avant le message courant pour donner le fil au modèle.
    Supporte le streaming si stream=True.
    """
    try:
        client = _groq_client()
    except Exception as exc:
        audit_event(
            action="provider.groq.init",
            tool="groq",
            result=str(exc),
            success=False,
        )
        logger.exception("Groq client initialization failed")
        raise

    messages = []
    
    current_system = system_prompt or SYSTEM_PROMPT

    if context:
        messages.append({
            "role": "user",
            "content": f"Mémoire disponible :\n\n{context}"
        })
        messages.append({
            "role": "assistant",
            "content": "Compris, j'intègre ces souvenirs dans ma réponse."
        })

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": prompt})

    try:
        logger.info("Groq request model=%s stream=%s chars=%s", model, stream, len(prompt))
        if stream:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": current_system}] + messages,
                temperature=0.7,
                max_tokens=4096,
                stream=True
            )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": current_system}] + messages,
            temperature=0.7,
            max_tokens=4096,
        )
        content = response.choices[0].message.content or ""
        audit_event(
            action="provider.groq.generate",
            tool="groq",
            result="ok",
            success=True,
            details={"model": model, "prompt_chars": len(prompt), "response_chars": len(content)},
        )
        return content
    except Exception as exc:
        audit_event(
            action="provider.groq.generate",
            tool="groq",
            result=str(exc),
            success=False,
            details={"model": model, "prompt_chars": len(prompt)},
        )
        logger.exception("Groq generation failed")
        raise


async def check_groq() -> dict:
    """Vérifie la disponibilité de Groq."""
    if not settings.GROQ_API_KEY:
        return {
            "status": "non configuré",
            "tip": "Ajoute GROQ_API_KEY=gsk_... dans ton fichier .env",
        }
    try:
        client = _groq_client()
        models = client.models.list()
        names  = [m.id for m in models.data]
        return {
            "status": "online",
            "active_model": GROQ_MODEL,
            "available": names,
        }
    except Exception as e:
        return {"status": "erreur", "error": str(e)}


# ── Wikipedia ─────────────────────────────────────────────────────────────────

async def wikipedia_search(query: str, lang: str = "fr") -> dict:
    """Cherche un article Wikipedia et retourne son extrait."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # 1. Chercher via l'API search (plus fiable qu'opensearch)
            search = await client.get(
                f"https://{lang}.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": 3,
                    "format": "json",
                    "utf8": 1,
                },
            )
            data = search.json()
            results = data.get("query", {}).get("search", [])

            if not results:
                if lang == "fr":
                    return await wikipedia_search(query, lang="en")
                return {"found": False, "query": query}

            title = results[0]["title"]

            # 2. Récupérer l'extrait complet
            extract_resp = await client.get(
                f"https://{lang}.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "titles": title,
                    "prop": "extracts",
                    "exsentences": 15,
                    "exintro": True,
                    "explaintext": True,
                    "format": "json",
                    "utf8": 1,
                },
            )
            pages   = extract_resp.json()["query"]["pages"]
            page    = list(pages.values())[0]
            extract = page.get("extract", "")

            if not extract and lang == "fr":
                return await wikipedia_search(query, lang="en")

            url = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"

            return {
                "found": True,
                "title": title,
                "extract": extract,
                "url": url,
                "lang": lang,
            }
    except Exception as e:
        return {"found": False, "error": str(e), "query": query}
async def wikipedia_ingest(query: str, lang: str = "fr") -> dict:
    """Cherche sur Wikipedia et ingère le résultat directement en mémoire."""
    from core.memory import add_memory
    from core.dedup import is_duplicate, register
    from core.ingestion import chunk_text

    result = await wikipedia_search(query, lang)
    if not result.get("found"):
        return {"ingested": False, "reason": "Article Wikipedia non trouvé", "query": query}

    content = f"[WIKIPEDIA] {result['title']}\n\n{result['extract']}"

    if is_duplicate(content):
        return {"ingested": False, "reason": "déjà connu du cerveau", "title": result["title"]}

    register(content)
    chunks = chunk_text(content)
    for i, chunk in enumerate(chunks):
        await add_memory(
            content=chunk,
            metadata={
                "source": result["url"],
                "title": result["title"],
                "tags": f"wikipedia,{lang},encyclopédie",
                "type": "wikipedia",
                "chunk_index": str(i),
                "total_chunks": str(len(chunks)),
            },
        )

    return {
        "ingested": True,
        "title": result["title"],
        "url": result["url"],
        "chunks_created": len(chunks),
        "extract_preview": result["extract"][:200] + "...",
    }

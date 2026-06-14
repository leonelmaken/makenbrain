"""
Extracteur de concepts — Phase 3
Deux modes :
  fast=True  → patterns regex + dictionnaire domaine (< 1 ms, sans LLM)
  fast=False → LLM (Groq ou local) pour une extraction plus précise
"""
import re
import json
from core.config import settings

# ── Dictionnaire du domaine SmartBudget Africa ────────────────────────────────

DOMAIN = {
    "technologie": [
        "spring boot", "java", "react", "typescript", "postgresql", "neon",
        "flyway", "jwt", "expo", "react native", "vite", "fastapi", "python",
        "chromadb", "ollama", "groq", "docker", "github", "maven", "gradle",
        "hibernate", "jpa", "redis", "kafka", "nginx", "swagger", "openapi",
        "bcrypt", "hmac", "sha512", "axios", "zustand", "tailwind", "duckduckgo",
        "networkx", "sentence-transformers", "langchain", "llama", "llm",
        "sqlite", "neon", "supabase", "vercel", "render", "railway",
    ],
    "domaine": [
        "tontine", "mobile money", "mtn momo", "orange money", "fintech",
        "microfinance", "épargne", "crédit", "paiement", "transaction",
        "authentification", "sécurité", "kyc", "aml", "compliance",
        "transfert", "remittance", "wallet", "compte", "solde",
        "diaspora", "afrique", "cameroun", "nigeria", "sénégal", "côte ivoire",
        "banque", "insurance", "assurance",
    ],
    "projet": [
        "smartbudget africa", "makenbrain", "backend", "frontend", "mobile",
        "admin", "dashboard", "api", "database", "monorepo", "pnpm",
        "pitch deck", "investor", "roadmap", "mvp", "déploiement",
    ],
    "architecture": [
        "microservice", "rest", "graphql", "websocket", "webhook",
        "authentication", "authorization", "rbac", "role", "permission",
        "entity", "repository", "service", "controller", "dto",
        "migration", "seed", "endpoint", "middleware", "interceptor",
        "circuit breaker", "rate limiting", "caching", "pagination",
    ],
}

# Tous les concepts connus dans un seul set pour lookup rapide
ALL_KNOWN = {c: t for t, concepts in DOMAIN.items() for c in concepts}


def _normalize(text: str) -> str:
    return text.lower().strip()


def extract_fast(text: str) -> list[dict]:
    """
    Extraction par patterns + dictionnaire. Rapide, sans LLM.
    Retourne max 20 concepts.
    """
    text_lower = text.lower()
    found: dict[str, dict] = {}

    # 1. Concepts du dictionnaire domaine
    for concept, ctype in ALL_KNOWN.items():
        if concept in text_lower:
            found[concept] = {"name": concept, "type": ctype, "confidence": 0.9}

    # 2. CamelCase (ex: SmartBudgetAfrica, AuthController, UserService)
    camel = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
    for term in camel:
        key = term.lower()
        if key not in found and len(term) > 5:
            found[key] = {"name": key, "type": "technique", "confidence": 0.75}

    # 3. Acronymes techniques (JWT, API, SQL, MVP, etc.)
    acronyms = re.findall(r'\b([A-Z]{2,6})\b', text)
    skip = {"I", "A", "OK", "GET", "PUT", "SET", "AS", "IN", "IS", "OR", "ON"}
    for acr in acronyms:
        key = acr.lower()
        if acr not in skip and key not in found:
            found[key] = {"name": key, "type": "acronyme", "confidence": 0.65}

    return list(found.values())[:20]


async def extract_llm(text: str) -> list[dict]:
    """
    Extraction via LLM. Utilise Groq (rapide) ou Ollama en fallback.
    """
    prompt = (
        "Extrais les concepts techniques et métier de ce texte.\n"
        "Réponds UNIQUEMENT en JSON valide, sans texte avant ni après.\n"
        'Format: {"concepts": [{"name": "...", "type": "technologie|projet|domaine|architecture", "relations": ["..."]}]}\n\n'
        f"Texte:\n{text[:600]}\n\nJSON:"
    )
    raw = ""
    try:
        if settings.GROQ_API_KEY:
            from groq import Groq as GroqClient
            client = GroqClient(api_key=settings.GROQ_API_KEY)
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content
        else:
            import ollama as _ollama
            resp = _ollama.generate(
                model=settings.OLLAMA_MODEL,
                prompt=prompt,
                options={"temperature": 0.1},
            )
            raw = resp.get("response", "")

        # Nettoyer et parser
        raw = raw.strip()
        if "```" in raw:
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        data = json.loads(raw)
        return data.get("concepts", [])

    except Exception:
        return extract_fast(text)


async def extract_and_graph(
    text: str,
    memory_id: str = "",
    fast: bool = True,
) -> dict:
    """
    Extrait les concepts d'un texte et les injecte dans le graphe de neurones.

    Retourne un résumé de ce qui a été ajouté.
    """
    from core.graph import neuron_graph

    if fast:
        concepts = extract_fast(text)
    else:
        concepts = await extract_llm(text)

    added: list[str] = []

    for c in concepts:
        name = _normalize(c.get("name", ""))
        if not name or len(name) < 3:
            continue
        neuron_graph.add_concept(
            name=name,
            concept_type=c.get("type", "concept"),
            memory_id=memory_id,
        )
        added.append(name)

    # Liens de co-occurrence entre tous les concepts du même chunk
    for i, a in enumerate(added):
        for b in added[i + 1:]:
            neuron_graph.add_connection(a, b, relationship="co-occurs", weight=0.6)

    # Liens explicites depuis extraction LLM
    if not fast:
        for c in concepts:
            name = _normalize(c.get("name", ""))
            for rel in c.get("relations", []):
                rel = _normalize(rel)
                if rel and rel != name and len(rel) > 2:
                    neuron_graph.add_connection(name, rel, relationship="relates_to", weight=0.8)

    neuron_graph.save()

    return {
        "concepts_added": len(added),
        "concepts": added[:10],
        "graph_nodes": neuron_graph.G.number_of_nodes(),
        "graph_edges": neuron_graph.G.number_of_edges(),
    }

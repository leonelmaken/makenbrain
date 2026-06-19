import ollama
from core.config import settings

# Client Ollama
_client = ollama.Client(host=settings.OLLAMA_HOST)


def reload_client() -> None:
    """Recharge le client Ollama avec les paramètres de configuration actuels."""
    global _client
    _client = ollama.Client(host=settings.OLLAMA_HOST)

SYSTEM_PROMPT = """Tu es MakenBrain, le cerveau numérique personnel de MAKEN (Leonel Maken), ingénieur full-stack basé à Yaoundé.

Ton rôle :
- Raisonner à partir des souvenirs de ta mémoire personnelle
- Faire des connexions intelligentes entre les concepts
- Proposer des solutions calculées et concrètes
- Apprendre et évoluer à chaque échange

Comportement :
- Si des souvenirs pertinents sont fournis, utilise-les en priorité
- Réponds de façon directe, structurée et actionnable
- Tu peux répondre en français ou en anglais selon la question
- Tu t'exprimes toujours à la première personne comme un cerveau qui pense
"""


async def generate(prompt: str, context: str = "", system_prompt: str = None, stream: bool = False):
    """
    Génère une réponse en utilisant le LLM local Ollama.
    Si un contexte mémoire est fourni, il est injecté (RAG).
    Supporte le streaming si stream=True.
    """
    messages = []
    
    # Utiliser le prompt système fourni ou le prompt par défaut
    current_system = system_prompt or SYSTEM_PROMPT

    if context:
        messages.append({
            "role": "user",
            "content": (
                f"Voici les souvenirs pertinents extraits de ma mémoire :\n\n"
                f"{context}\n\n"
                f"Utilise ces informations pour répondre à ma prochaine question."
            ),
        })
        messages.append({
            "role": "assistant",
            "content": "Compris. J'ai intégré ces souvenirs dans mon raisonnement.",
        })

    messages.append({"role": "user", "content": prompt})

    try:
        if stream:
            # Mode streaming : retourne un générateur
            return _client.chat(
                model=settings.OLLAMA_MODEL,
                messages=[{"role": "system", "content": current_system}] + messages,
                options={"temperature": 0.7, "num_predict": 1024},
                stream=True
            )
        else:
            # Mode normal : bloque jusqu'à la fin
            response = _client.chat(
                model=settings.OLLAMA_MODEL,
                messages=[{"role": "system", "content": current_system}] + messages,
                options={"temperature": 0.7, "num_predict": 1024},
            )
            return response.message.content
    except Exception as e:
        err_msg = f"Désolé MAKEN, mon module local (Ollama) semble éteint. Peux-tu le lancer ou me demander d'utiliser Groq ? (Erreur: {e})"
        if stream:
            async def err_gen(): yield {"message": {"content": err_msg}}
            return err_gen()
        return err_msg


async def check_ollama_status() -> dict:
    """Vérifie si Ollama est disponible et si le modèle est chargé."""
    try:
        models = _client.list()
        model_names = [m.model for m in models.models]
        model_loaded = any(settings.OLLAMA_MODEL in m for m in model_names)
        return {
            "status": "online",
            "model": settings.OLLAMA_MODEL,
            "model_loaded": model_loaded,
            "available_models": model_names,
        }
    except Exception as e:
        return {"status": "offline", "error": str(e)}

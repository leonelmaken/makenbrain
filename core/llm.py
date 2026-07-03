import ollama
from core.config import settings
from core.system_prompts import SYSTEM_PROMPT  # générique par défaut — jamais de données personnelles

# Client Ollama
_client = ollama.Client(host=settings.OLLAMA_HOST)


def reload_client() -> None:
    """Recharge le client Ollama avec les paramètres de configuration actuels."""
    global _client
    _client = ollama.Client(host=settings.OLLAMA_HOST)


async def generate(
    prompt: str,
    context: str = "",
    system_prompt: str = None,
    stream: bool = False,
    history: list[dict] | None = None,
):
    """
    Génère une réponse en utilisant le LLM local Ollama.
    Si un contexte mémoire est fourni, il est injecté (RAG).
    `history` : messages précédents de la conversation ({role, content}),
    injectés avant le message courant pour donner le fil au modèle.
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

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": prompt})

    try:
        if stream:
            # Mode streaming : retourne un générateur
            return _client.chat(
                model=settings.OLLAMA_MODEL,
                messages=[{"role": "system", "content": current_system}] + messages,
                options={"temperature": 0.7, "num_predict": 2048},
                stream=True
            )
        else:
            # Mode normal : bloque jusqu'à la fin
            response = _client.chat(
                model=settings.OLLAMA_MODEL,
                messages=[{"role": "system", "content": current_system}] + messages,
                options={"temperature": 0.7, "num_predict": 2048},
            )
            return response.message.content
    except Exception as e:
        err_msg = f"Le module local (Ollama) semble éteint. Relance-le ou utilise le provider cloud. (Erreur: {e})"
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

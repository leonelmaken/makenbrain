import ollama
from core.config import settings
from core.system_prompts import SYSTEM_PROMPT  # générique par défaut — jamais de données personnelles

# Client Ollama
_client = ollama.Client(host=settings.OLLAMA_HOST)


def reload_client() -> None:
    """Recharge le client Ollama avec les paramètres de configuration actuels."""
    global _client, _installed_models
    _client = ollama.Client(host=settings.OLLAMA_HOST)
    _installed_models = None


# ── Routage local intelligent ─────────────────────────────────────────────────
# Les questions de CODE sont dirigées vers un modèle spécialisé installé
# localement (bien meilleur que le modèle généraliste sur ce domaine).
# Ordre de préférence : du plus capable au plus léger.
_CODER_MODELS_PREFERENCE = (
    "qwen2.5-coder:7b",
    "qwen2.5-coder:latest",
    "deepseek-coder:6.7b",
    "qwen2.5-coder:1.5b",
)

_CODE_KEYWORDS = (
    "code", "fonction", "function", "bug", "debug", "erreur", "error",
    "script", "python", "javascript", "typescript", "java", "sql", "html",
    "css", "api", "classe", "class", "def ", "import", "variable", "boucle",
    "algorithme", "compile", "syntaxe", "refactor", "regex", "json", "backend",
    "frontend", "framework", "librairie", "library", "programme", "coder",
)

_installed_models: list[str] | None = None


def _get_installed_models() -> list[str]:
    """Liste des modèles Ollama installés (mise en cache par processus)."""
    global _installed_models
    if _installed_models is None:
        try:
            _installed_models = [m.model for m in _client.list().models]
        except Exception:
            _installed_models = []
    return _installed_models


def pick_local_model(message: str) -> str:
    """Choisit le meilleur modèle local pour la question.

    Question de code → premier modèle codeur installé (préférence qualité).
    Sinon → modèle généraliste configuré (OLLAMA_MODEL).
    """
    lowered = message.lower()
    if any(kw in lowered for kw in _CODE_KEYWORDS):
        installed = _get_installed_models()
        for candidate in _CODER_MODELS_PREFERENCE:
            if candidate in installed:
                return candidate
    return settings.OLLAMA_MODEL


async def generate(
    prompt: str,
    context: str = "",
    system_prompt: str = None,
    stream: bool = False,
    history: list[dict] | None = None,
    model: str | None = None,
):
    """
    Génère une réponse en utilisant le LLM local Ollama.
    Si un contexte mémoire est fourni, il est injecté (RAG).
    `history` : messages précédents de la conversation ({role, content}),
    injectés avant le message courant pour donner le fil au modèle.
    `model` : modèle Ollama explicite (sinon routage automatique via
    pick_local_model — les questions de code vont aux modèles codeurs).
    Supporte le streaming si stream=True.
    """
    chosen_model = model or pick_local_model(prompt)
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
                model=chosen_model,
                messages=[{"role": "system", "content": current_system}] + messages,
                options={"temperature": 0.7, "num_predict": 2048},
                stream=True
            )
        else:
            # Mode normal : bloque jusqu'à la fin
            response = _client.chat(
                model=chosen_model,
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

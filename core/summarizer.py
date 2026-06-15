"""
Résumé automatique après chaque ingestion.
Le cerveau génère une synthèse de ce qu'il vient d'apprendre.
"""
import ollama
from core.config import settings

_client = ollama.Client(host=settings.OLLAMA_HOST)


def reload_client() -> None:
    """Recharge le client Ollama pour le résumé avec les paramètres actuels."""
    global _client
    _client = ollama.Client(host=settings.OLLAMA_HOST)

SUMMARY_PROMPT = """Tu es un assistant de synthèse expert. Résume le document suivant en 3 à 5 phrases.
Le résumé doit être :
- Factuel et précis
- En français
- Centré sur les informations clés
- Directement utilisable comme référence

Titre : {title}

Contenu :
{content}

Résumé :"""


async def summarize(text: str, title: str = "Sans titre") -> str:
    """
    Génère un résumé concis d'un texte via le LLM local.
    Tronque automatiquement les textes trop longs.
    """
    # Tronquer à 3000 chars pour rester rapide sur CPU
    content_preview = text[:3000] + ("..." if len(text) > 3000 else "")

    try:
        response = _client.chat(
            model=settings.OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": SUMMARY_PROMPT.format(
                        title=title, content=content_preview
                    ),
                }
            ],
            options={"temperature": 0.2, "num_predict": 300},
        )
        return response.message.content.strip()
    except Exception as e:
        return f"[Résumé indisponible : {e}]"

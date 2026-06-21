"""Centralized application settings for MakenBrain.

All configuration -- local LLM, vector memory, external connectors and
security -- is loaded once here from environment variables / `.env` and
exposed through the shared `settings` instance. Nothing else in the
codebase should read `os.environ` directly.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM Local
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"

    # Mémoire vectorielle
    CHROMA_PATH: str = "./brain_data/chroma"
    COLLECTION_NAME: str = "makenbrain"
    EMBED_MODEL: str = "all-MiniLM-L6-v2"

    # API
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"

    # Connecteurs externes (Phase 5)
    GROQ_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    HF_API_KEY: str = ""

    # ── Sécurité (Phase 1) ────────────────────────────────
    # Clé d'authentification locale, exigée via l'en-tête `X-API-Key`
    # pour les endpoints sensibles (agent, scheduler, audit, écriture mémoire).
    # Sans clé configurée, ces endpoints refusent tout accès (échec fermé).
    ADMIN_API_KEY: str = ""

    # Origines autorisées en CORS, séparées par des virgules.
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # Limitation de débit appliquée aux endpoints sensibles.
    RATE_LIMIT_MAX_REQUESTS: int = 20
    RATE_LIMIT_WINDOW_SECONDS: float = 60.0

    # ── Supabase (Phase 2.1) ──────────────────────
    # Backend Supabase (Postgres + RLS). Utilisé uniquement via
    # core/supabase_client.py -- aucun autre module ne doit lire ces
    # variables directement.
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def cors_origins(self) -> list[str]:
        """Parse ALLOWED_ORIGINS into a clean list for CORSMiddleware."""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]


def reload_settings(env_file: str = ".env") -> Settings:
    """Recharge la configuration à partir du fichier .env et des variables d'environnement."""
    new_settings = Settings(_env_file=env_file)
    for field, value in new_settings.model_dump().items():
        setattr(settings, field, value)
    return settings


settings = Settings()

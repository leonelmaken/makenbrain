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

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

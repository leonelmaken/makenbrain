from pydantic import BaseModel, Field
from typing import Optional


# ── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Message à envoyer au cerveau")
    use_memory: bool = Field(True, description="Utiliser la mémoire pour contextualiser")
    n_context: int = Field(5, ge=1, le=20, description="Nombre de souvenirs à récupérer")
    relevance_threshold: float = Field(0.75, ge=0.0, le=1.0, description="Seuil de pertinence (distance cosinus)")

    model_config = {"json_schema_extra": {
        "example": {
            "message": "Quel est mon projet fintech principal?",
            "use_memory": True,
            "n_context": 5,
        }
    }}


class ChatResponse(BaseModel):
    response: str
    memories_used: int
    model: str
    context_preview: Optional[list[str]] = None


# ── Mémoire ───────────────────────────────────────────────────────────────────

class MemoryAddRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Contenu du souvenir")
    source: Optional[str] = Field("manual", description="Origine du souvenir")
    tags: Optional[str] = Field("", description="Tags séparés par des virgules")
    title: Optional[str] = Field("", description="Titre optionnel")

    model_config = {"json_schema_extra": {
        "example": {
            "content": "SmartBudget Africa est une application fintech panafricaine.",
            "source": "projet",
            "tags": "smartbudget,fintech,afrique",
            "title": "SmartBudget Africa",
        }
    }}


class MemoryAddResponse(BaseModel):
    id: str
    message: str


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    n_results: int = Field(5, ge=1, le=50)

    model_config = {"json_schema_extra": {
        "example": {"query": "fintech Afrique", "n_results": 5}
    }}


class MemoryItem(BaseModel):
    id: str
    content: str
    metadata: dict
    distance: float


class MemorySearchResponse(BaseModel):
    query: str
    total_found: int
    results: list[MemoryItem]


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Texte brut à ingérer")
    title: Optional[str] = Field("", description="Titre du document")
    source: Optional[str] = Field("text_input")
    tags: Optional[str] = Field("")
    chunk_size: int = Field(400, ge=100, le=2000, description="Taille des fragments")

    model_config = {"json_schema_extra": {
        "example": {
            "text": "La tontine est un système d'épargne collectif très répandu en Afrique...",
            "title": "Tontine Africa",
            "source": "recherche",
            "tags": "tontine,fintech,afrique",
        }
    }}


class IngestUrlRequest(BaseModel):
    url: str = Field(..., description="URL à ingérer")
    tags: Optional[str] = Field("")

    model_config = {"json_schema_extra": {
        "example": {
            "url": "https://example.com/article",
            "tags": "web,recherche",
        }
    }}


class IngestResponse(BaseModel):
    message: str
    chunks_created: int
    ids: list[str]
    title: str

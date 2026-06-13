import uuid
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions

from core.config import settings

# Singletons
_chroma_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


async def init_memory() -> None:
    """Initialise ChromaDB et la collection principale."""
    global _chroma_client, _collection

    _chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.EMBED_MODEL
    )

    _collection = _chroma_client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def _get_collection() -> chromadb.Collection:
    if _collection is None:
        raise RuntimeError("Mémoire non initialisée. Appelle init_memory() d'abord.")
    return _collection


async def add_memory(content: str, metadata: dict | None = None) -> str:
    """
    Ajoute un souvenir dans la mémoire vectorielle.
    Retourne l'identifiant unique du souvenir.
    """
    col = _get_collection()
    memory_id = str(uuid.uuid4())

    meta = {
        "created_at": datetime.now().isoformat(),
        "source": "manual",
        "tags": "",
        "title": "",
    }
    if metadata:
        meta.update({k: str(v) for k, v in metadata.items()})

    col.add(
        documents=[content],
        metadatas=[meta],
        ids=[memory_id],
    )
    return memory_id


async def search_memory(query: str, n_results: int = 5) -> list[dict]:
    """
    Recherche sémantique dans la mémoire.
    Retourne les souvenirs les plus proches du query.
    """
    col = _get_collection()
    count = col.count()
    if count == 0:
        return []

    actual_n = min(n_results, count)
    results = col.query(
        query_texts=[query],
        n_results=actual_n,
        include=["documents", "metadatas", "distances"],
    )

    memories = []
    for i, doc in enumerate(results["documents"][0]):
        memories.append({
            "id": results["ids"][0][i],
            "content": doc,
            "metadata": results["metadatas"][0][i],
            "distance": round(results["distances"][0][i], 4),
        })
    return memories


async def delete_memory(memory_id: str) -> bool:
    """Supprime un souvenir par son identifiant."""
    col = _get_collection()
    try:
        col.delete(ids=[memory_id])
        return True
    except Exception:
        return False


async def get_memory_stats() -> dict:
    """Retourne les statistiques de la mémoire."""
    col = _get_collection()
    total = col.count()
    return {
        "total_memories": total,
        "collection": settings.COLLECTION_NAME,
        "embed_model": settings.EMBED_MODEL,
        "storage_path": settings.CHROMA_PATH,
    }

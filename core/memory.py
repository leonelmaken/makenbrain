import time
import uuid
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from core.config import settings

# Singletons
_chroma_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


async def init_memory() -> None:
    """Initialise ChromaDB et la collection principale.

    Désactive la télémétrie anonyme de ChromaDB (activée par défaut dans le
    SDK) : elle déclenche un appel réseau sortant à chaque instanciation du
    client, sans aucun lien fonctionnel avec la mémoire vectorielle --
    seulement un risque de lenteur supplémentaire sur réseau lent/restreint.
    """
    global _chroma_client, _collection

    _chroma_client = chromadb.PersistentClient(
        path=settings.CHROMA_PATH,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    embed_start = time.monotonic()
    # Chargement HORS-LIGNE d'abord : une fois le modèle en cache local
    # (~/.cache/huggingface), aucune requête réseau vers Hugging Face.
    # Sans ça, chaque démarrage revalidait le modèle en ligne — jusqu'à
    # 4+ minutes sur réseau lent. Repli en ligne uniquement au tout
    # premier lancement (modèle pas encore téléchargé).
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBED_MODEL, local_files_only=True
        )
        print(f"[STARTUP]   modèle d'embedding '{settings.EMBED_MODEL}' chargé depuis le cache local en {time.monotonic() - embed_start:.1f}s.")
    except Exception:
        print("[STARTUP]   modèle absent du cache local — téléchargement depuis Hugging Face…")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBED_MODEL
        )
        print(f"[STARTUP]   modèle d'embedding '{settings.EMBED_MODEL}' téléchargé et chargé en {time.monotonic() - embed_start:.1f}s.")

    _collection = _chroma_client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


async def reload_memory() -> None:
    """Recharge la mémoire vectorielle si la configuration a changé."""
    global _chroma_client, _collection
    _chroma_client = None
    _collection = None
    await init_memory()


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

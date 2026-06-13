from fastapi import APIRouter, HTTPException
from models.schemas import (
    MemoryAddRequest, MemoryAddResponse,
    MemorySearchRequest, MemorySearchResponse, MemoryItem,
)
from core.memory import add_memory, search_memory, delete_memory, get_memory_stats

router = APIRouter()


@router.post("/add", response_model=MemoryAddResponse)
async def add(request: MemoryAddRequest):
    """Ajoute un souvenir dans la mémoire vectorielle."""
    memory_id = await add_memory(
        content=request.content,
        metadata={"source": request.source, "tags": request.tags, "title": request.title},
    )
    return MemoryAddResponse(id=memory_id, message=f"Souvenir ajouté (id: {memory_id[:8]}...).")


@router.post("/search", response_model=MemorySearchResponse)
async def search(request: MemorySearchRequest):
    """Recherche sémantique dans la mémoire."""
    results = await search_memory(request.query, n_results=request.n_results)
    items = [MemoryItem(**r) for r in results]
    return MemorySearchResponse(query=request.query, total_found=len(items), results=items)


@router.delete("/delete/{memory_id}")
async def delete(memory_id: str):
    """Supprime un souvenir par son identifiant."""
    success = await delete_memory(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Souvenir introuvable.")
    return {"message": f"Souvenir {memory_id[:8]}... supprimé."}


@router.get("/stats")
async def stats():
    """Statistiques globales de la mémoire."""
    return await get_memory_stats()

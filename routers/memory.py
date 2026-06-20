"""Memory API routes with audit logging for state-changing operations."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from core.audit import audit_event
from core.auth import SECURE
from core.memory import add_memory, delete_memory, get_memory_stats, search_memory
from models.schemas import (
    MemoryAddRequest,
    MemoryAddResponse,
    MemoryItem,
    MemorySearchRequest,
    MemorySearchResponse,
)

router = APIRouter()


@router.post("/add", response_model=MemoryAddResponse, dependencies=SECURE)
async def add(request: MemoryAddRequest) -> MemoryAddResponse:
    """Add one memory entry to the vector store and audit the state change."""
    memory_id = await add_memory(
        content=request.content,
        metadata={"source": request.source, "tags": request.tags, "title": request.title},
    )
    audit_event(
        action="memory.add",
        tool="memory",
        endpoint="/memory/add",
        result=memory_id,
        success=True,
        details={"source": request.source, "tags": request.tags, "title": request.title},
    )
    return MemoryAddResponse(id=memory_id, message=f"Souvenir ajouté (id: {memory_id[:8]}...).")


@router.post("/search", response_model=MemorySearchResponse)
async def search(request: MemorySearchRequest) -> MemorySearchResponse:
    """Search memories semantically without changing persisted state."""
    results = await search_memory(request.query, n_results=request.n_results)
    items = [MemoryItem(**result) for result in results]
    return MemorySearchResponse(query=request.query, total_found=len(items), results=items)


@router.delete("/delete/{memory_id}", dependencies=SECURE)
async def delete(memory_id: str) -> dict[str, str]:
    """Delete one memory by id and audit both success and failure."""
    success = await delete_memory(memory_id)
    if not success:
        audit_event(
            action="memory.delete",
            tool="memory",
            endpoint="/memory/delete/{memory_id}",
            result="not found",
            success=False,
            details={"memory_id": memory_id},
        )
        raise HTTPException(status_code=404, detail="Souvenir introuvable.")

    audit_event(
        action="memory.delete",
        tool="memory",
        endpoint="/memory/delete/{memory_id}",
        result="deleted",
        success=True,
        details={"memory_id": memory_id},
    )
    return {"message": f"Souvenir {memory_id[:8]}... supprimé."}


@router.get("/stats")
async def stats() -> dict:
    """Return memory store statistics."""
    return await get_memory_stats()

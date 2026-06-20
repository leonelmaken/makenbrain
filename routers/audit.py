"""Read-only API for the centralized JSON audit log."""
from __future__ import annotations

from fastapi import APIRouter

from core.audit import get_audit_log

router = APIRouter()


@router.get("/log")
async def audit_log(limit: int = 100) -> dict:
    """Return recent audit entries for security and debugging."""
    return {"entries": get_audit_log(limit=limit)}

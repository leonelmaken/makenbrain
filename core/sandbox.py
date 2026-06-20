"""Strict filesystem sandbox for MakenBrain file operations."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import HTTPException

from core.audit import audit_event

logger = logging.getLogger("makenbrain.sandbox")

ROOT = Path.cwd().resolve()
ALLOWED_ROOT_NAMES = ("workspace", "projects", "uploads")
ALLOWED_ROOTS = tuple((ROOT / name).resolve() for name in ALLOWED_ROOT_NAMES)
DENIED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "windows",
    "system32",
    "program files",
    "program files (x86)",
}


class SandboxViolation(ValueError):
    """Raised when a path escapes the allowed MakenBrain sandbox."""


def ensure_sandbox_roots() -> None:
    """Create allowed sandbox roots if they are missing."""
    for root in ALLOWED_ROOTS:
        root.mkdir(parents=True, exist_ok=True)


def sandbox_roots() -> list[str]:
    """Return allowed sandbox roots as strings."""
    ensure_sandbox_roots()
    return [str(root) for root in ALLOWED_ROOTS]


def resolve_sandbox_path(path: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve and validate a path against the strict allowed roots."""
    ensure_sandbox_roots()
    raw = str(path)
    if not raw or "\x00" in raw:
        raise SandboxViolation("Chemin invalide.")
    candidate = Path(raw)
    if ".." in candidate.parts:
        raise SandboxViolation("Chemin relatif dangereux interdit.")
    resolved = candidate.resolve()
    lowered_parts = {part.lower() for part in resolved.parts}
    if lowered_parts & DENIED_PARTS:
        raise SandboxViolation("Chemin système ou protégé interdit.")
    if not any(resolved == root or root in resolved.parents for root in ALLOWED_ROOTS):
        raise SandboxViolation("Accès hors sandbox interdit. Utilise workspace/, projects/ ou uploads/.")
    if must_exist and not resolved.exists():
        raise SandboxViolation(f"Chemin introuvable : {resolved}")
    return resolved


def http_sandbox_path(path: str | Path, *, action: str, endpoint: str, must_exist: bool = False) -> Path:
    """Resolve a sandbox path or raise HTTP 403 with an audit entry."""
    try:
        return resolve_sandbox_path(path, must_exist=must_exist)
    except SandboxViolation as exc:
        audit_event(
            action=action,
            tool="sandbox",
            endpoint=endpoint,
            file_path=str(path),
            result=str(exc),
            success=False,
        )
        logger.warning("Sandbox violation on %s: %s", path, exc)
        raise HTTPException(status_code=403, detail=str(exc))

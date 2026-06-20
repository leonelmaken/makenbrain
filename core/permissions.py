"""Session permissions constrained by the MakenBrain sandbox."""
from __future__ import annotations

import logging
from pathlib import Path

from core.audit import audit_event
from core.sandbox import SandboxViolation, resolve_sandbox_path, sandbox_roots

logger = logging.getLogger("makenbrain.permissions")


class PermissionManager:
    """Track allowed sandbox paths for sensitive file operations."""

    def __init__(self) -> None:
        self.allowed_paths: set[str] = set(sandbox_roots())

    def grant_access(self, path: str) -> bool:
        """Grant access to a path only if it is inside the strict sandbox."""
        try:
            abs_path = str(resolve_sandbox_path(path, must_exist=False))
        except SandboxViolation as exc:
            audit_event(
                action="permission.grant",
                tool="permissions",
                file_path=path,
                result=str(exc),
                success=False,
            )
            logger.warning("Permission grant denied for %s: %s", path, exc)
            return False

        self.allowed_paths.add(abs_path)
        audit_event(
            action="permission.grant",
            tool="permissions",
            file_path=abs_path,
            result="granted",
            success=True,
        )
        return True

    def revoke_access(self, path: str) -> None:
        """Revoke access from a sandbox path."""
        abs_path = str(resolve_sandbox_path(path, must_exist=False))
        self.allowed_paths.discard(abs_path)

    def is_allowed(self, path: str) -> bool:
        """Return True when the path is sandboxed and under an allowed root."""
        try:
            abs_path = Path(resolve_sandbox_path(path, must_exist=False))
        except SandboxViolation:
            return False

        for allowed in self.allowed_paths:
            allowed_path = Path(allowed)
            if abs_path == allowed_path or allowed_path in abs_path.parents:
                return True
        return False

    def require(self, path: str, *, action: str, endpoint: str = "") -> Path:
        """Return a sandbox path or raise PermissionError."""
        resolved = resolve_sandbox_path(path, must_exist=False)
        if not self.is_allowed(str(resolved)):
            audit_event(
                action=action,
                tool="permissions",
                endpoint=endpoint,
                file_path=str(resolved),
                result="permission denied",
                success=False,
            )
            raise PermissionError(f"Permission refusée pour : {resolved}")
        return resolved

    def list_permissions(self) -> list[str]:
        """List current sandbox permissions."""
        return sorted(self.allowed_paths)


permission_manager = PermissionManager()

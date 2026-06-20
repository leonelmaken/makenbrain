"""Phase 1 security tests for sandbox, permissions, backups and file agent APIs."""
from __future__ import annotations

from pathlib import Path
import shutil

import pytest
from fastapi.testclient import TestClient

from core.backups import create_backup, restore_backup
from core.config import settings
from core.permissions import permission_manager
from core.rate_limit import RateLimiter
from core.sandbox import SandboxViolation, resolve_sandbox_path
from main import app


@pytest.fixture(autouse=True)
def cleanup_phase1_workspace():
    """Remove test-only sandbox files after each test run."""
    yield
    test_root = resolve_sandbox_path("workspace/tests_phase1", must_exist=False)
    if test_root.exists():
        shutil.rmtree(test_root)


def _workspace_file(name: str) -> Path:
    """Return a unique file path inside the allowed workspace sandbox."""
    root = resolve_sandbox_path("workspace/tests_phase1", must_exist=False)
    root.mkdir(parents=True, exist_ok=True)
    return root / name


def _auth_headers() -> dict[str, str]:
    """Return the admin API key header expected by protected endpoints."""
    return {"X-API-Key": settings.ADMIN_API_KEY}


def test_sandbox_rejects_parent_traversal() -> None:
    """Paths containing parent traversal must never enter the sandbox."""
    with pytest.raises(SandboxViolation):
        resolve_sandbox_path("../secret.txt")


def test_sandbox_allows_workspace_paths() -> None:
    """Regular files under workspace/ are accepted by the strict sandbox."""
    path = resolve_sandbox_path("workspace/tests_phase1/allowed.txt", must_exist=False)
    assert "workspace" in path.parts


def test_permission_manager_denies_system_paths() -> None:
    """Permission grants outside workspace/projects/uploads must fail closed."""
    assert permission_manager.grant_access("C:/Windows/System32/config") is False


def test_backup_create_and_restore() -> None:
    """Backups preserve the previous file content and can restore it."""
    target = _workspace_file("backup_restore.txt")
    target.write_text("before", encoding="utf-8")

    backup = create_backup(target, reason="pytest")
    assert backup is not None

    target.write_text("after", encoding="utf-8")
    restore_backup(backup["id"])

    assert target.read_text(encoding="utf-8") == "before"


def test_agent_write_dry_run_does_not_create_file() -> None:
    """The write endpoint must produce a diff without touching disk in dry-run mode."""
    target = _workspace_file("dry_run_write.txt")
    target.unlink(missing_ok=True)
    client = TestClient(app)

    response = client.post(
        "/agent/write",
        headers=_auth_headers(),
        json={"file_path": str(target), "content": "hello", "dry_run": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dry_run"] is True
    assert "diff" in body
    assert not target.exists()


def test_agent_write_requires_apply_changes_for_real_write() -> None:
    """Real writes are blocked unless apply_changes=true is explicitly provided."""
    target = _workspace_file("blocked_write.txt")
    client = TestClient(app)

    response = client.post(
        "/agent/write",
        headers=_auth_headers(),
        json={"file_path": str(target), "content": "hello", "dry_run": False},
    )

    assert response.status_code == 400
    assert not target.exists()


def test_agent_write_and_delete_with_apply_changes() -> None:
    """Dangerous file operations work only inside sandbox with explicit apply_changes."""
    target = _workspace_file("apply_write_delete.txt")
    target.unlink(missing_ok=True)
    client = TestClient(app)

    write_response = client.post(
        "/agent/write",
        headers=_auth_headers(),
        json={
            "file_path": str(target),
            "content": "created",
            "dry_run": False,
            "apply_changes": True,
            "auto_ingest": False,
        },
    )
    assert write_response.status_code == 200
    assert target.read_text(encoding="utf-8") == "created"

    delete_response = client.request(
        "DELETE",
        "/agent/delete",
        headers=_auth_headers(),
        json={
            "file_path": str(target),
            "confirm": True,
            "dry_run": False,
            "apply_changes": True,
        },
    )
    assert delete_response.status_code == 200
    assert not target.exists()


def test_protected_agent_endpoint_rejects_missing_api_key() -> None:
    """Protected endpoints must reject calls without X-API-Key."""
    target = _workspace_file("missing_key.txt")
    client = TestClient(app)

    response = client.post(
        "/agent/write",
        json={"file_path": str(target), "content": "hello", "dry_run": True},
    )

    assert response.status_code == 401


def test_cors_uses_configured_origins() -> None:
    """CORS must allow configured origins without using a wildcard policy."""
    client = TestClient(app)
    origin = settings.cors_origins[0]

    response = client.options(
        "/",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_rate_limiter_blocks_after_limit() -> None:
    """The sliding-window limiter must block requests after the configured quota."""
    limiter = RateLimiter(max_requests=2, window_seconds=60)

    assert limiter.check("client:/critical") is True
    assert limiter.check("client:/critical") is True
    assert limiter.check("client:/critical") is False

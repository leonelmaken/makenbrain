"""Backup and rollback utilities for sensitive file changes."""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from core.audit import audit_event
from core.sandbox import resolve_sandbox_path

BACKUP_ROOT = Path("brain_data/backups").resolve()
BACKUP_INDEX = BACKUP_ROOT / "index.json"


def _load_index() -> list[dict[str, Any]]:
    if not BACKUP_INDEX.exists():
        return []
    try:
        return json.loads(BACKUP_INDEX.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_index(entries: list[dict[str, Any]]) -> None:
    BACKUP_INDEX.parent.mkdir(parents=True, exist_ok=True)
    BACKUP_INDEX.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def create_backup(path: str | Path, *, reason: str = "") -> dict[str, Any] | None:
    """Create a backup for an existing sandboxed file."""
    source = resolve_sandbox_path(path, must_exist=True)
    if not source.is_file():
        return None

    backup_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = source.name.replace(" ", "_")
    destination = BACKUP_ROOT / timestamp / f"{backup_id}_{safe_name}.bak"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

    entry = {
        "id": backup_id,
        "timestamp": datetime.now().isoformat(),
        "source_path": str(source),
        "backup_path": str(destination),
        "reason": reason,
    }
    entries = _load_index()
    entries.append(entry)
    _save_index(entries)
    audit_event(
        action="backup.create",
        tool="backup",
        file_path=str(source),
        result=str(destination),
        success=True,
        details={"backup_id": backup_id, "reason": reason},
    )
    return entry


def list_backups(path: str | Path | None = None) -> list[dict[str, Any]]:
    """List backups, optionally filtered by source path."""
    entries = _load_index()
    if path is None:
        return entries
    source = str(resolve_sandbox_path(path, must_exist=False))
    return [entry for entry in entries if entry.get("source_path") == source]


def restore_backup(backup_id: str) -> dict[str, Any]:
    """Restore a backup by id to its original sandboxed path."""
    entries = _load_index()
    entry = next((item for item in entries if item.get("id") == backup_id), None)
    if not entry:
        raise FileNotFoundError(f"Backup introuvable : {backup_id}")

    source_path = resolve_sandbox_path(entry["source_path"], must_exist=False)
    backup_path = Path(entry["backup_path"]).resolve()
    if not backup_path.exists() or not backup_path.is_file():
        raise FileNotFoundError(f"Fichier backup introuvable : {backup_path}")

    source_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_path, source_path)
    audit_event(
        action="backup.restore",
        tool="backup",
        file_path=str(source_path),
        result="restored",
        success=True,
        details={"backup_id": backup_id, "backup_path": str(backup_path)},
    )
    return {"restored": True, "backup_id": backup_id, "path": str(source_path)}

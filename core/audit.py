"""Central JSON audit log for sensitive MakenBrain actions."""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

AUDIT_LOG_FILE = Path("brain_data/audit_log.json")
logger = logging.getLogger("makenbrain.audit")


def _load_entries() -> list[dict[str, Any]]:
    if not AUDIT_LOG_FILE.exists():
        return []
    try:
        return json.loads(AUDIT_LOG_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to read audit log: %s", exc)
        return []


def _save_entries(entries: list[dict[str, Any]]) -> None:
    AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_LOG_FILE.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def audit_event(
    *,
    action: str,
    tool: str,
    endpoint: str = "",
    file_path: str = "",
    result: str = "",
    success: bool = True,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a structured audit entry and mirror it to the application logger."""
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "tool": tool,
        "endpoint": endpoint,
        "file": Path(file_path).name if file_path else "",
        "path": file_path,
        "result": result,
        "success": success,
        "details": details or {},
    }
    entries = _load_entries()
    entries.append(entry)
    _save_entries(entries)
    log_fn = logger.info if success else logger.warning
    log_fn("audit action=%s endpoint=%s success=%s path=%s", action, endpoint, success, file_path)
    return entry


def get_audit_log(limit: int = 100) -> list[dict[str, Any]]:
    """Return the most recent audit entries."""
    return _load_entries()[-limit:]

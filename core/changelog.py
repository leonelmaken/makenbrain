"""
Changelog — trace toutes les actions de l'agent.
Chaque proposition, validation et modification est documentée.
Stocké dans brain_data/changelog.json
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

CHANGELOG_FILE  = Path("brain_data/changelog.json")
PROPOSALS_FILE  = Path("brain_data/proposals.json")


# ── Changelog ─────────────────────────────────────────────────────────────────

def _load(path: Path) -> list:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def log_action(
    action: str,
    file_path: str = "",
    description: str = "",
    status: str = "done",
    proposal_id: str = "",
    details: dict | None = None,
) -> dict:
    """Enregistre une action dans le changelog."""
    entry = {
        "id":          str(uuid.uuid4())[:8],
        "timestamp":   datetime.now().isoformat(),
        "action":      action,
        "file":        Path(file_path).name if file_path else "",
        "path":        file_path,
        "description": description,
        "status":      status,
        "proposal_id": proposal_id,
        "details":     details or {},
    }
    log = _load(CHANGELOG_FILE)
    log.append(entry)
    _save(CHANGELOG_FILE, log)
    return entry


def get_changelog(limit: int = 50) -> list:
    return _load(CHANGELOG_FILE)[-limit:]


# ── Proposals (workflow de validation) ────────────────────────────────────────

def create_proposal(
    title: str,
    description: str,
    action_type: str,
    file_path: str = "",
    action_data: dict | None = None,
    risks: list | None = None,
) -> dict:
    """
    Crée une proposition en attente de validation.
    Le cerveau propose, MAKEN valide, puis le cerveau agit.
    """
    proposal = {
        "id":          str(uuid.uuid4())[:8],
        "created_at":  datetime.now().isoformat(),
        "title":       title,
        "description": description,
        "action_type": action_type,
        "file_path":   file_path,
        "action_data": action_data or {},
        "risks":       risks or [],
        "status":      "pending",       # pending | approved | rejected | done
        "approved_at": None,
        "done_at":     None,
    }
    proposals = _load(PROPOSALS_FILE)
    proposals.append(proposal)
    _save(PROPOSALS_FILE, proposals)
    return proposal


def get_proposals(status: Optional[str] = None) -> list:
    proposals = _load(PROPOSALS_FILE)
    if status:
        return [p for p in proposals if p["status"] == status]
    return proposals


def get_proposal(proposal_id: str) -> dict | None:
    proposals = _load(PROPOSALS_FILE)
    for p in proposals:
        if p["id"] == proposal_id:
            return p
    return None


def update_proposal_status(proposal_id: str, status: str) -> dict | None:
    proposals = _load(PROPOSALS_FILE)
    for p in proposals:
        if p["id"] == proposal_id:
            p["status"] = status
            if status == "approved":
                p["approved_at"] = datetime.now().isoformat()
            elif status == "done":
                p["done_at"] = datetime.now().isoformat()
            _save(PROPOSALS_FILE, proposals)
            return p
    return None

"""Secure file agent routes with sandbox, permissions, dry-run and rollback."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.audit import audit_event
from core.backups import create_backup, list_backups, restore_backup
from core.dedup import is_duplicate, register
from core.diff import unified_diff
from core.memory import add_memory, search_memory
from core.permissions import permission_manager
from core.providers import groq_generate

router = APIRouter()
logger = logging.getLogger("makenbrain.agent")

ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")


class ReadRequest(BaseModel):
    file_path: str
    analyze: bool = True
    provider: str = "groq"
    auto_ingest: bool = True


class WriteRequest(BaseModel):
    file_path: str
    content: Optional[str] = None
    prompt: Optional[str] = None
    provider: str = "groq"
    auto_ingest: bool = True
    dry_run: bool = True
    apply_changes: bool = False


class ModifyRequest(BaseModel):
    file_path: str
    instruction: str
    provider: str = "groq"
    backup: bool = True
    dry_run: bool = True
    apply_changes: bool = False


class SuggestRequest(BaseModel):
    file_path: str
    goal: Optional[str] = ""
    provider: str = "groq"


class DeleteRequest(BaseModel):
    file_path: str
    confirm: bool = False
    dry_run: bool = True
    apply_changes: bool = False


class ExploreRequest(BaseModel):
    folder_path: str
    analyze: bool = True


class RestoreBackupRequest(BaseModel):
    backup_id: str


def _require_path(path: str, *, action: str, endpoint: str, must_exist: bool = False) -> Path:
    try:
        resolved = permission_manager.require(path, action=action, endpoint=endpoint)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    if must_exist and not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Chemin introuvable : {resolved}")
    return resolved


def _read_file(path: Path) -> str:
    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return ""


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _strip_markdown_fences(content: str) -> str:
    cleaned = re.sub(r"^```[\w]*\n?", "", content.strip())
    return re.sub(r"\n?```$", "", cleaned.strip())


def _require_apply(dry_run: bool, apply_changes: bool) -> None:
    if dry_run:
        return
    if not apply_changes:
        raise HTTPException(
            status_code=400,
            detail="Action bloquée: ajoute apply_changes=true ou utilise dry_run=true.",
        )


@router.post("/read")
async def read_file(req: ReadRequest):
    """Read and optionally analyze a sandboxed file."""
    endpoint = "/agent/read"
    p = _require_path(req.file_path, action="file.read", endpoint=endpoint, must_exist=True)
    if not p.is_file():
        raise HTTPException(400, "Ce chemin n'est pas un fichier.")

    content = _read_file(p)
    if not content.strip():
        raise HTTPException(400, "Fichier vide ou illisible.")

    ingest_result = None
    if req.auto_ingest and not is_duplicate(content):
        register(content)
        from routers.files import chunk_text
        for chunk in chunk_text(content):
            await add_memory(chunk, metadata={
                "source": str(p), "title": p.name,
                "tags": "agent,fichier", "type": "agent_read",
            })
        ingest_result = "ingéré en mémoire"
    elif req.auto_ingest:
        ingest_result = "déjà connu"

    analysis = None
    if req.analyze:
        prompt = (
            f"Analyse ce fichier de code ({p.name}) et donne-moi :\n"
            "1. Ce que fait ce fichier\n2. Les points forts\n"
            "3. Les risques ou améliorations suggérées\n\n"
            f"```{p.suffix[1:] if p.suffix else 'text'}\n{content[:4000]}\n```"
        )
        related = await search_memory(p.name, n_results=3)
        ctx = "\n".join([m["content"][:200] for m in related if m["distance"] < 0.7])
        analysis = await groq_generate(prompt, ctx)

    audit_event(
        action="file.read",
        tool="agent",
        endpoint=endpoint,
        file_path=str(p),
        result="ok",
        success=True,
        details={"analyze": req.analyze, "auto_ingest": req.auto_ingest},
    )
    return {
        "file": p.name,
        "path": str(p),
        "size_chars": len(content),
        "lines": content.count("\n") + 1,
        "extension": p.suffix,
        "ingest": ingest_result,
        "analysis": analysis,
        "content_preview": content[:500] + "..." if len(content) > 500 else content,
    }


@router.post("/write")
async def write_file(req: WriteRequest):
    """Create or replace a sandboxed file after dry-run/apply confirmation."""
    endpoint = "/agent/write"
    _require_apply(req.dry_run, req.apply_changes)
    p = _require_path(req.file_path, action="file.write", endpoint=endpoint)

    if req.content is not None:
        final_content = req.content
        generated = False
    elif req.prompt:
        related = await search_memory(req.prompt, n_results=5)
        ctx = "\n".join([m["content"][:300] for m in related if m["distance"] < 0.75])
        ext = p.suffix.lstrip(".") or "text"
        system_prompt = (
            f"Tu es MakenBrain. Génère uniquement le contenu du fichier {p.name} ({ext}), "
            "sans explication, sans markdown, juste le contenu brut."
        )
        full_prompt = f"Contexte du projet :\n{ctx}\n\nTâche : {req.prompt}" if ctx else req.prompt
        final_content = _strip_markdown_fences(
            await groq_generate(full_prompt, system_prompt=system_prompt)
        )
        generated = True
    else:
        raise HTTPException(400, "Fournis 'content' ou 'prompt'.")

    original = _read_file(p) if p.exists() else ""
    diff = unified_diff(original, final_content, fromfile=str(p), tofile=f"{p} (proposé)")
    if req.dry_run:
        audit_event(
            action="file.write.dry_run",
            tool="agent",
            endpoint=endpoint,
            file_path=str(p),
            result="diff generated",
            success=True,
        )
        return {
            "dry_run": True,
            "would_write": True,
            "file": p.name,
            "path": str(p),
            "generated_by": "LLM (Groq)" if generated else "contenu direct",
            "diff": diff,
            "explanation": "Aucune écriture réelle. Relance avec dry_run=false et apply_changes=true.",
        }

    backup = create_backup(p, reason="agent.write") if p.exists() else None
    try:
        _write_file(p, final_content)
    except Exception as exc:
        if backup:
            restore_backup(backup["id"])
        audit_event(action="file.write", tool="agent", endpoint=endpoint, file_path=str(p), result=str(exc), success=False)
        raise HTTPException(500, f"Ecriture échouée: {exc}")

    if req.auto_ingest:
        await add_memory(
            f"[CRÉÉ/MODIFIÉ] {p.name} : {final_content[:500]}",
            metadata={"source": str(p), "title": p.name, "tags": "agent,write", "type": "agent_write"},
        )

    audit_event(action="file.write", tool="agent", endpoint=endpoint, file_path=str(p), result="written", success=True)
    return {
        "created": True,
        "file": p.name,
        "path": str(p),
        "generated_by": "LLM (Groq)" if generated else "contenu direct",
        "lines_written": final_content.count("\n") + 1,
        "backup": backup,
    }


@router.post("/modify")
async def modify_file(req: ModifyRequest):
    """Modify a sandboxed file from a natural-language instruction."""
    endpoint = "/agent/modify"
    _require_apply(req.dry_run, req.apply_changes)
    p = _require_path(req.file_path, action="file.modify", endpoint=endpoint, must_exist=True)
    if not p.is_file():
        raise HTTPException(400, "Ce chemin n'est pas un fichier.")

    original = _read_file(p)
    if not original.strip():
        raise HTTPException(400, "Fichier vide ou illisible.")

    related = await search_memory(req.instruction, n_results=3)
    ctx = "\n".join([m["content"][:200] for m in related if m["distance"] < 0.75])
    prompt = (
        f"Voici un fichier ({p.name}) à modifier.\n\n"
        f"Instruction : {req.instruction}\n\n"
        f"Fichier original :\n```{p.suffix.lstrip('.')}\n{original[:6000]}\n```\n\n"
        "Retourne UNIQUEMENT le fichier modifié complet, sans explication, sans markdown."
    )
    modified = _strip_markdown_fences(await groq_generate(prompt, ctx))
    diff = unified_diff(original, modified, fromfile=str(p), tofile=f"{p} (proposé)")

    if req.dry_run:
        audit_event(
            action="file.modify.dry_run",
            tool="agent",
            endpoint=endpoint,
            file_path=str(p),
            result="diff generated",
            success=True,
        )
        return {
            "dry_run": True,
            "would_modify": True,
            "file": p.name,
            "instruction": req.instruction,
            "diff": diff,
            "explanation": "Aucune modification réelle. Relance avec dry_run=false et apply_changes=true.",
        }

    backup = create_backup(p, reason=f"agent.modify: {req.instruction}") if req.backup else None
    try:
        _write_file(p, modified)
    except Exception as exc:
        if backup:
            restore_backup(backup["id"])
        audit_event(action="file.modify", tool="agent", endpoint=endpoint, file_path=str(p), result=str(exc), success=False)
        raise HTTPException(500, f"Modification échouée: {exc}")

    await add_memory(
        f"[MODIFIÉ] {p.name} - {req.instruction}",
        metadata={"source": str(p), "title": p.name, "tags": "agent,modify", "type": "agent_modify"},
    )
    audit_event(action="file.modify", tool="agent", endpoint=endpoint, file_path=str(p), result="modified", success=True)
    return {
        "modified": True,
        "file": p.name,
        "instruction": req.instruction,
        "backup": backup,
        "lines_before": original.count("\n") + 1,
        "lines_after": modified.count("\n") + 1,
        "diff": diff,
    }


@router.post("/suggest")
async def suggest(req: SuggestRequest):
    """Analyze a sandboxed file without modifying it."""
    endpoint = "/agent/suggest"
    p = _require_path(req.file_path, action="file.suggest", endpoint=endpoint, must_exist=True)
    content = _read_file(p)
    if not content.strip():
        raise HTTPException(400, "Fichier vide ou illisible.")

    related = await search_memory(p.name + " " + (req.goal or ""), n_results=5)
    ctx = "\n".join([m["content"][:200] for m in related if m["distance"] < 0.75])
    goal_str = f"\nObjectif spécifique : {req.goal}" if req.goal else ""
    prompt = (
        f"Analyse ce fichier ({p.name}) et propose des améliorations concrètes.{goal_str}\n\n"
        "Donne résumé, problèmes, améliorations et priorité.\n\n"
        f"```{p.suffix.lstrip('.')}\n{content[:4000]}\n```"
    )
    suggestions = await groq_generate(prompt, ctx)
    audit_event(action="file.suggest", tool="agent", endpoint=endpoint, file_path=str(p), result="ok", success=True)
    return {"file": p.name, "path": str(p), "goal": req.goal or "général", "suggestions": suggestions, "modified": False}


@router.delete("/delete")
async def delete_file(req: DeleteRequest):
    """Delete a sandboxed file only after confirm and apply_changes."""
    endpoint = "/agent/delete"
    p = _require_path(req.file_path, action="file.delete", endpoint=endpoint, must_exist=True)
    if not p.is_file():
        raise HTTPException(400, "Ce chemin n'est pas un fichier.")
    if not req.confirm:
        return {"deleted": False, "message": "Ajoute confirm=true pour confirmer.", "file": str(p)}
    _require_apply(req.dry_run, req.apply_changes)
    if req.dry_run:
        audit_event(action="file.delete.dry_run", tool="agent", endpoint=endpoint, file_path=str(p), result="would delete", success=True)
        return {"dry_run": True, "would_delete": True, "file": p.name, "path": str(p)}

    backup = create_backup(p, reason="agent.delete")
    try:
        p.unlink()
    except Exception as exc:
        if backup:
            restore_backup(backup["id"])
        audit_event(action="file.delete", tool="agent", endpoint=endpoint, file_path=str(p), result=str(exc), success=False)
        raise HTTPException(500, f"Suppression échouée: {exc}")
    audit_event(action="file.delete", tool="agent", endpoint=endpoint, file_path=str(p), result="deleted", success=True)
    return {"deleted": True, "file": p.name, "backup": backup}


@router.post("/explore")
async def explore_folder(req: ExploreRequest):
    """List and optionally analyze a sandboxed folder."""
    endpoint = "/agent/explore"
    folder = _require_path(req.folder_path, action="folder.explore", endpoint=endpoint, must_exist=True)
    if not folder.is_dir():
        raise HTTPException(400, "Ce chemin n'est pas un dossier.")

    skip = {".git", "node_modules", ".venv", "__pycache__", "build", "dist", "target", ".idea"}
    extensions = {".java", ".ts", ".tsx", ".js", ".jsx", ".py", ".md", ".sql", ".yml", ".yaml", ".json"}
    files_info = []
    total_lines = 0
    for f in sorted(folder.rglob("*")):
        if not f.is_file() or any(d in f.parts for d in skip) or f.suffix not in extensions:
            continue
        lines = _read_file(f).count("\n") + 1
        total_lines += lines
        files_info.append({"file": f.name, "path": str(f.relative_to(folder)), "ext": f.suffix, "lines": lines})

    by_ext: dict[str, int] = {}
    for info in files_info:
        by_ext[info["ext"]] = by_ext.get(info["ext"], 0) + 1

    overview = None
    if req.analyze and files_info:
        file_list = "\n".join([f"- {item['path']} ({item['lines']} lignes)" for item in files_info[:30]])
        overview = await groq_generate(
            f"Voici la structure du dossier '{folder.name}' :\n\n{file_list}\n\n"
            "Donne une vue d'ensemble concise, les forces et les améliorations."
        )
    audit_event(action="folder.explore", tool="agent", endpoint=endpoint, file_path=str(folder), result="ok", success=True)
    return {"folder": folder.name, "total_files": len(files_info), "total_lines": total_lines, "by_extension": by_ext, "files": files_info[:50], "overview": overview}


@router.get("/backups")
async def backups(file_path: Optional[str] = None):
    """List backups for all files or one sandboxed file."""
    return {"backups": list_backups(file_path)}


@router.post("/backups/restore")
async def restore(req: RestoreBackupRequest):
    """Restore a backup by id."""
    try:
        return restore_backup(req.backup_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))

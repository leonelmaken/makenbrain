"""Secure ingestion and file watcher endpoints."""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional

import fitz
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from core.archiver import archiver
from core.audit import audit_event
from core.auth import SECURE
from core.dedup import is_duplicate, register
from core.memory import add_memory
from core.permissions import permission_manager
from core.sandbox import resolve_sandbox_path
from core.summarizer import summarize
from core.watcher import get_log, get_status, start_watcher, stop_watcher

router = APIRouter()
logger = logging.getLogger("makenbrain.files")

SUPPORTED_EXT = {
    ".ts", ".tsx", ".js", ".jsx", ".json", ".md", ".txt", ".csv", ".log",
    ".java", ".xml", ".yml", ".yaml", ".properties", ".sql", ".gradle", ".http",
    ".pdf", ".html", ".env.example", ".kt", ".swift",
}
IGNORED_DIRS = {".git", "node_modules", ".venv", "__pycache__", "brain_data", ".idea", "dist", "build", "target"}


class IngestFileRequest(BaseModel):
    file_path: str
    tags: Optional[str] = ""


class IngestFolderRequest(BaseModel):
    folder_path: str
    tags: Optional[str] = ""
    extensions: Optional[list[str]] = None
    max_files: Optional[int] = 200
    with_summary: bool = False


class WatchRequest(BaseModel):
    folder_path: str


def chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    """Split text into sentence-aware chunks."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = current + " " + sentence if current else sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if len(chunk) > 20]


def read_pdf(file_path: str) -> str:
    """Extract text from a PDF."""
    doc = fitz.open(file_path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def read_text_file(file_path: str) -> str:
    """Read a text file with common encodings and clean HTML when needed."""
    p = Path(file_path)
    raw = ""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            raw = p.read_text(encoding=enc, errors="ignore")
            break
        except Exception:
            continue
    if p.suffix.lower() == ".html" and raw:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    return raw


async def ingest_single_file(file_path: str, tags: str = "", with_summary: bool = True) -> dict:
    """Ingest one sandboxed file into memory."""
    p = permission_manager.require(file_path, action="ingest.file", endpoint="/files/ingest-file")
    if not p.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")
    if not p.is_file():
        raise ValueError(f"Ce chemin n'est pas un fichier : {file_path}")
    if p.suffix.lower() not in SUPPORTED_EXT:
        return {"chunks_created": 0, "skipped": True, "reason": "format non supporté", "file": p.name}

    text = read_pdf(str(p)) if p.suffix.lower() == ".pdf" else read_text_file(str(p))
    if not text or len(text.strip()) < 20:
        return {"chunks_created": 0, "skipped": True, "reason": "contenu vide ou trop court", "file": p.name}
    if is_duplicate(text):
        return {"chunks_created": 0, "skipped": True, "reason": "déjà connu du cerveau", "file": p.name}

    register(text)
    summary = ""
    if with_summary:
        summary = await summarize(text, title=p.name)
        await add_memory(
            content=f"[RÉSUMÉ] {p.name} : {summary}",
            metadata={"source": str(p), "title": p.name, "tags": f"resume,{tags}".strip(","), "type": "summary", "file_type": p.suffix},
        )

    chunks = chunk_text(text)
    ids = []
    for i, chunk in enumerate(chunks):
        mid = await add_memory(
            content=chunk,
            metadata={
                "source": str(p),
                "title": p.name,
                "tags": tags,
                "chunk_index": str(i),
                "total_chunks": str(len(chunks)),
                "file_type": p.suffix,
                "type": "chunk",
            },
        )
        ids.append(mid)

    return {"file": p.name, "path": str(p), "chunks_created": len(chunks), "summary": summary, "skipped": False, "ids": ids}


@router.post("/ingest-file", dependencies=SECURE)
async def ingest_file(request: IngestFileRequest):
    """Ingest one sandboxed file."""
    try:
        result = await ingest_single_file(request.file_path, request.tags or "")
        audit_event(action="ingest.file", tool="files", endpoint="/files/ingest-file", file_path=result.get("path", request.file_path), result="ok", success=True)
        return result
    except FileNotFoundError as exc:
        audit_event(action="ingest.file", tool="files", endpoint="/files/ingest-file", file_path=request.file_path, result=str(exc), success=False)
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError as exc:
        audit_event(action="ingest.file", tool="files", endpoint="/files/ingest-file", file_path=request.file_path, result=str(exc), success=False)
        raise HTTPException(status_code=403, detail=str(exc))
    except Exception as exc:
        audit_event(action="ingest.file", tool="files", endpoint="/files/ingest-file", file_path=request.file_path, result=str(exc), success=False)
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/upload", dependencies=SECURE)
async def upload_files(files: list[UploadFile] = File(...), tags: str = Form("")):
    """Store uploads under uploads/ and ingest them."""
    uploads_dir = resolve_sandbox_path("uploads", must_exist=False)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for uploaded in files:
        suffix = Path(uploaded.filename).suffix.lower()
        safe_name = Path(uploaded.filename).name.replace(" ", "_")
        upload_path = uploads_dir / safe_name
        data = await uploaded.read()
        upload_path.write_bytes(data)

        if suffix == ".zip":
            try:
                extract_dir = archiver.extract_zip(upload_path)
                internal_files = archiver.list_files(extract_dir)
                zip_report = {"archive": uploaded.filename, "files_count": len(internal_files), "processed": []}
                ok_count = 0
                for internal_file in internal_files:
                    try:
                        res = await ingest_single_file(str(internal_file), tags=f"zip,{uploaded.filename},{tags}", with_summary=False)
                        zip_report["processed"].append(res)
                        if not res.get("skipped"):
                            ok_count += 1
                    except Exception as exc:
                        zip_report["processed"].append({"file": internal_file.name, "error": str(exc)})
                zip_report["message"] = f"{ok_count} fichiers de l'archive ingérés."
                results.append(zip_report)
                archiver.cleanup(extract_dir)
            finally:
                upload_path.unlink(missing_ok=True)
            continue

        if suffix not in SUPPORTED_EXT:
            results.append({"file": uploaded.filename, "skipped": True, "reason": "format non supporté"})
            continue

        try:
            result = await ingest_single_file(str(upload_path), tags, with_summary=False)
            result["file"] = uploaded.filename
            results.append(result)
        except Exception as exc:
            results.append({"file": uploaded.filename, "skipped": True, "reason": str(exc)})

    audit_event(action="ingest.upload", tool="files", endpoint="/files/upload", result=f"{len(files)} item(s)", success=True)
    return {"message": f"Traitement terminé pour {len(files)} élément(s).", "results": results}


@router.post("/ingest-folder", dependencies=SECURE)
async def ingest_folder(request: IngestFolderRequest):
    """Recursively ingest a sandboxed folder."""
    try:
        folder = permission_manager.require(request.folder_path, action="ingest.folder", endpoint="/files/ingest-folder")
    except PermissionError as exc:
        audit_event(action="ingest.folder", tool="files", endpoint="/files/ingest-folder", file_path=request.folder_path, result=str(exc), success=False)
        raise HTTPException(status_code=403, detail=str(exc))

    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"Dossier introuvable : {request.folder_path}")

    allowed_ext = set(request.extensions) if request.extensions else SUPPORTED_EXT
    report = {"folder": str(folder), "processed": [], "skipped_dedup": [], "skipped_unsupported": 0, "errors": [], "total_files_ingested": 0, "total_chunks": 0}
    candidates = [
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in allowed_ext and not any(d in f.parts for d in IGNORED_DIRS)
    ]
    if request.max_files and len(candidates) > request.max_files:
        candidates = candidates[:request.max_files]

    for file_path in candidates:
        try:
            result = await ingest_single_file(str(file_path), request.tags or "", request.with_summary)
            if result.get("skipped"):
                report["skipped_dedup"].append({"file": result["file"], "reason": result.get("reason", "")})
            else:
                report["processed"].append({"file": result["file"], "chunks": result["chunks_created"], "summary": (result.get("summary") or "mode rapide")[:120]})
                report["total_files_ingested"] += 1
                report["total_chunks"] += result["chunks_created"]
        except Exception as exc:
            report["errors"].append({"file": str(file_path), "error": str(exc)})

    report["message"] = f"{report['total_files_ingested']} fichier(s) ingéré(s) - {len(report['skipped_dedup'])} déjà connus - {report['total_chunks']} fragments créés."
    audit_event(action="ingest.folder", tool="files", endpoint="/files/ingest-folder", file_path=str(folder), result=report["message"], success=True)
    return report


@router.post("/watch", dependencies=SECURE)
async def watch_folder(request: WatchRequest):
    """Start automatic ingestion watcher on a sandboxed folder."""
    try:
        folder = permission_manager.require(request.folder_path, action="watch.start", endpoint="/files/watch")
    except PermissionError as exc:
        audit_event(action="watch.start", tool="files", endpoint="/files/watch", file_path=request.folder_path, result=str(exc), success=False)
        raise HTTPException(status_code=403, detail=str(exc))
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Dossier introuvable : {request.folder_path}")
    loop = asyncio.get_event_loop()
    start_watcher(str(folder), loop)
    audit_event(action="watch.start", tool="files", endpoint="/files/watch", file_path=str(folder), result="active", success=True)
    return {"message": f"Surveillance active sur : {folder}", "path": str(folder), "active": True}


@router.delete("/watch", dependencies=SECURE)
async def stop_watching():
    """Stop automatic folder watching."""
    stop_watcher()
    audit_event(action="watch.stop", tool="files", endpoint="/files/watch", result="stopped", success=True)
    return {"message": "Surveillance arrêtée."}


@router.get("/watch/status")
async def watcher_status():
    """Return watcher status."""
    return get_status()


@router.get("/report")
async def ingestion_report():
    """Return recent automatic ingestion log."""
    log = get_log()
    return {"total_events": len(log), "log": log[-50:]}

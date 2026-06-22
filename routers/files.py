"""
Secure ingestion and file watcher endpoints for MakenBrain.

Ce module agit comme une couche API stricte pour la gestion des fichiers.
Il délègue toute la logique métier (parsing, chunking, déduplication, stockage)
au module centralisé `core.ingestion`.

Architecture :
- Routers : Validation HTTP, gestion des erreurs, audit, permissions sandbox.
- Core : Logique métier pure, indépendante du framework web.

Règle d'or : Aucune logique de parsing ou de traitement de fichier ne doit
exister dans ce router. Tout passe par `ingest_single_file`.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

# ── Imports Centralisés (Core Logic) ──────────────────────────────────────────
# Importation unique de la logique métier et des constantes partagées.
# Cela évite les imports locaux répétitifs et garantit la cohérence.
from core.archiver import archiver
from core.audit import audit_event
from core.auth import SECURE
from core.dedup import is_duplicate, register
from core.ingestion import ingest_single_file, SUPPORTED_EXT
from core.memory import add_memory
from core.permissions import permission_manager
from core.sandbox import resolve_sandbox_path
from core.summarizer import summarize
from core.watcher import get_log, get_status, start_watcher, stop_watcher

router = APIRouter()
logger = logging.getLogger("makenbrain.files")

# ── Configuration Locale du Router ────────────────────────────────────────────
# Dossiers ignorés lors de l'exploration récursive (sécurité et performance).
IGNORED_DIRS = {
    ".git", "node_modules", ".venv", "__pycache__", "brain_data", 
    ".idea", "dist", "build", "target"
}

# ── Modèles Pydantic (Request Schemas) ────────────────────────────────────────

class IngestFileRequest(BaseModel):
    """Requête pour l'ingestion d'un fichier unique."""
    file_path: str
    tags: Optional[str] = ""

class IngestFolderRequest(BaseModel):
    """Requête pour l'ingestion récursive d'un dossier."""
    folder_path: str
    tags: Optional[str] = ""
    extensions: Optional[list[str]] = None
    max_files: Optional[int] = 200
    with_summary: bool = False

class WatchRequest(BaseModel):
    """Requête pour démarrer la surveillance automatique d'un dossier."""
    folder_path: str

# ── API Endpoints ─────────────────────────────────────────────────────────────

@router.post("/ingest-file", dependencies=SECURE)
async def ingest_file(request: IngestFileRequest):
    """
    Ingest a single sandboxed file into the brain's memory.
    
    Délègue le traitement complet (lecture, chunking, résumé, stockage) 
    à `core.ingestion.ingest_single_file`.
    
    Args:
        request: Contient le chemin du fichier et les tags optionnels.
        
    Returns:
        Résultat de l'ingestion (chunks créés, résumé, IDs).
        
    Raises:
        HTTPException: 404 si fichier introuvable, 403 si hors sandbox.
    """
    try:
        result = await ingest_single_file(request.file_path, request.tags or "")
        audit_event(
            action="ingest.file", 
            tool="files", 
            endpoint="/files/ingest-file", 
            file_path=result.get("path", request.file_path), 
            result="ok", 
            success=True
        )
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
    """
    Upload files to the secure 'uploads/' sandbox and ingest them.
    
    Gère également l'extraction et l'ingestion des archives ZIP.
    Utilise `SUPPORTED_EXT` importé globalement pour valider les formats.
    
    Args:
        files: Liste des fichiers uploadés.
        tags: Tags à associer aux fichiers ingérés.
        
    Returns:
        Rapport de traitement pour chaque fichier.
    """
    uploads_dir = resolve_sandbox_path("uploads", must_exist=False)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for uploaded in files:
        suffix = Path(uploaded.filename).suffix.lower()
        safe_name = Path(uploaded.filename).name.replace(" ", "_")
        upload_path = uploads_dir / safe_name
        data = await uploaded.read()
        upload_path.write_bytes(data)

        # Gestion spéciale des archives ZIP
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

        # Validation du format via la constante globale SUPPORTED_EXT
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
    """
    Recursively ingest all supported files from a sandboxed folder.
    
    Applique les filtres d'extensions et ignore les dossiers systèmes.
    Utilise `SUPPORTED_EXT` importé globalement.
    
    Args:
        request: Contient le chemin du dossier et les options de filtrage.
        
    Returns:
        Rapport détaillé de l'ingestion (fichiers traités, ignorés, erreurs).
    """
    try:
        folder = permission_manager.require(request.folder_path, action="ingest.folder", endpoint="/files/ingest-folder")
    except PermissionError as exc:
        audit_event(action="ingest.folder", tool="files", endpoint="/files/ingest-folder", file_path=request.folder_path, result=str(exc), success=False)
        raise HTTPException(status_code=403, detail=str(exc))

    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"Dossier introuvable : {request.folder_path}")

    # Utilisation de la constante globale SUPPORTED_EXT si aucune extension n'est spécifiée
    allowed_ext = set(request.extensions) if request.extensions else SUPPORTED_EXT
    
    report = {
        "folder": str(folder), 
        "processed": [], 
        "skipped_dedup": [], 
        "skipped_unsupported": 0, 
        "errors": [], 
        "total_files_ingested": 0, 
        "total_chunks": 0
    }
    
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
                report["processed"].append({
                    "file": result["file"], 
                    "chunks": result["chunks_created"], 
                    "summary": (result.get("summary") or "mode rapide")[:120]
                })
                report["total_files_ingested"] += 1
                report["total_chunks"] += result["chunks_created"]
        except Exception as exc:
            report["errors"].append({"file": str(file_path), "error": str(exc)})

    report["message"] = f"{report['total_files_ingested']} fichier(s) ingéré(s) - {len(report['skipped_dedup'])} déjà connus - {report['total_chunks']} fragments créés."
    audit_event(action="ingest.folder", tool="files", endpoint="/files/ingest-folder", file_path=str(folder), result=report["message"], success=True)
    return report


@router.post("/watch", dependencies=SECURE)
async def watch_folder(request: WatchRequest):
    """
    Start automatic ingestion watcher on a sandboxed folder.
    
    Active le daemon Watchdog qui détecte les créations/modifications de fichiers
    et les ingère automatiquement via `core.watcher`.
    
    Args:
        request: Chemin du dossier à surveiller.
        
    Returns:
        Statut de activation de la surveillance.
    """
    try:
        folder = permission_manager.require(request.folder_path, action="watch.start", endpoint="/files/watch")
    except PermissionError as exc:
        audit_event(action="watch.start", tool="files", endpoint="/files/watch", file_path=request.folder_path, result=str(exc), success=False)
        raise HTTPException(status_code=403, detail=str(exc))
        
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Dossier introuvable : {request.folder_path}")
        
    loop = asyncio.get_running_loop()
    start_watcher(str(folder), loop)
    
    audit_event(action="watch.start", tool="files", endpoint="/files/watch", file_path=str(folder), result="active", success=True)
    return {"message": f"Surveillance active sur : {folder}", "path": str(folder), "active": True}


@router.delete("/watch", dependencies=SECURE)
async def stop_watching():
    """
    Stop automatic folder watching.
    
    Désactive le daemon Watchdog et vide la liste des chemins surveillés.
    """
    stop_watcher()
    audit_event(action="watch.stop", tool="files", endpoint="/files/watch", result="stopped", success=True)
    return {"message": "Surveillance arrêtée."}


@router.get("/watch/status")
async def watcher_status():
    """
    Return current status of the file watcher.
    
    Returns:
        Dictionnaire contenant l'état actif, les chemins surveillés et les logs récents.
    """
    return get_status()


@router.get("/report")
async def ingestion_report():
    """
    Return recent automatic ingestion log.
    
    Utile pour déboguer les ingestions automatiques déclenchées par le watcher.
    
    Returns:
        Liste des 50 derniers événements d'ingestion automatique.
    """
    log = get_log()
    return {"total_events": len(log), "log": log[-50:]}
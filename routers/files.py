"""
Endpoints Phase 2 :
- Ingestion d'un fichier unique (texte ou PDF)
- Ingestion d'un dossier entier
- Surveillance automatique de dossiers
"""
import asyncio
import re
import tempfile
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from core.dedup import is_duplicate, register
from core.memory import add_memory
from core.summarizer import summarize
from core.watcher import start_watcher, stop_watcher, get_status, get_log

router = APIRouter()

SUPPORTED_EXT = {
    # Frontend / Mobile
    ".ts", ".tsx", ".js", ".jsx", ".json", ".md", ".txt", ".csv", ".log",
    # Backend Java / Spring Boot
    ".java", ".xml", ".yml", ".yaml", ".properties", ".sql", ".gradle", ".http",
    # Docs & Config
    ".pdf", ".html", ".env.example", ".kt", ".swift",
}
IGNORED_DIRS  = {".git", "node_modules", ".venv", "__pycache__", "brain_data", ".idea", "dist", "build", "target"}


# ── Schemas ───────────────────────────────────────────────────────────────────

class IngestFileRequest(BaseModel):
    file_path: str
    tags: Optional[str] = ""

    model_config = {"json_schema_extra": {
        "example": {
            "file_path": "C:/Users/MAKEN/Documents/Projet/smartbudget/README.md",
            "tags": "smartbudget,doc"
        }
    }}


class IngestFolderRequest(BaseModel):
    folder_path: str
    tags: Optional[str] = ""
    extensions: Optional[list[str]] = None
    max_files: Optional[int] = 200
    with_summary: bool = False  # False = rapide (pas de LLM par fichier)

    model_config = {"json_schema_extra": {
        "example": {
            "folder_path": "C:/Users/MAKEN/Documents/Projet/SmartBudget Africa",
            "tags": "smartbudget,projet",
            "max_files": 100,
            "with_summary": False
        }
    }}


class WatchRequest(BaseModel):
    folder_path: str

    model_config = {"json_schema_extra": {
        "example": {
            "folder_path": "C:/Users/MAKEN/Documents/Projet"
        }
    }}


# ── Utilitaires ───────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    """Découpe le texte en fragments cohérents (sur fins de phrase)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= chunk_size:
            current = current + " " + s if current else s
        else:
            if current:
                chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 20]


def read_pdf(file_path: str) -> str:
    """Extrait le texte d'un PDF page par page via PyMuPDF."""
    doc = fitz.open(file_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def read_text_file(file_path: str) -> str:
    """Lit un fichier texte avec gestion des encodages."""
    p = Path(file_path)
    raw = ""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            raw = p.read_text(encoding=enc, errors="ignore")
            break
        except Exception:
            continue
    # Extraction propre du texte pour les fichiers HTML
    if p.suffix.lower() == ".html" and raw:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    return raw


# ── Fonction centrale d'ingestion ─────────────────────────────────────────────

async def ingest_single_file(file_path: str, tags: str = "", with_summary: bool = True) -> dict:
    """
    Ingère un fichier unique dans la mémoire.
    Gère : déduplication, extraction, résumé, fragmentation.
    """
    p = Path(file_path)

    if not p.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")
    if not p.is_file():
        raise ValueError(f"Ce chemin n'est pas un fichier : {file_path}")

    # Lecture selon le type
    if p.suffix.lower() == ".pdf":
        text = read_pdf(str(p))
    else:
        text = read_text_file(str(p))

    if not text or len(text.strip()) < 20:
        return {"chunks_created": 0, "skipped": True, "reason": "contenu vide ou trop court", "file": p.name}

    # Déduplication
    if is_duplicate(text):
        return {"chunks_created": 0, "skipped": True, "reason": "déjà connu du cerveau", "file": p.name}

    # Enregistrer le hash
    register(text)

    # Résumé automatique (désactivable pour les ingestions massives)
    summary = ""
    if with_summary:
        summary = await summarize(text, title=p.name)
        await add_memory(
            content=f"[RÉSUMÉ] {p.name} : {summary}",
            metadata={
                "source": str(p),
                "title": p.name,
                "tags": f"resume,{tags}".strip(","),
                "type": "summary",
                "file_type": p.suffix,
            },
        )

    # Découper et stocker les fragments
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

    return {
        "file": p.name,
        "path": str(p),
        "chunks_created": len(chunks),
        "summary": summary,
        "skipped": False,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/ingest-file")
async def ingest_file(request: IngestFileRequest):
    """Ingère un fichier unique (texte ou PDF) dans la mémoire du cerveau."""
    try:
        result = await ingest_single_file(request.file_path, request.tags)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


from core.archiver import archiver

@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...), tags: str = Form("")):
    """
    Reçoit des fichiers ou des archives ZIP.
    Si c'est un ZIP, il est extrait et chaque fichier interne est ingéré.
    """
    results = []
    for f in files:
        suffix = Path(f.filename).suffix.lower()
        
        # Cas spécial : Archive ZIP
        if suffix == ".zip":
            data = await f.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                tmp_zip.write(data)
                zip_path = Path(tmp_zip.name)
            
            try:
                extract_dir = archiver.extract_zip(zip_path)
                internal_files = archiver.list_files(extract_dir)
                
                ok_count = 0
                zip_report = {"archive": f.filename, "files_count": len(internal_files), "processed": []}
                for int_f in internal_files:
                    try:
                        res = await ingest_single_file(str(int_f), tags=f"zip,{f.filename},{tags}", with_summary=False)
                        zip_report["processed"].append(res)
                        if not res.get("skipped"): ok_count += 1
                    except Exception as e:
                        zip_report["processed"].append({"file": int_f.name, "error": str(e)})
                
                zip_report["message"] = f"{ok_count} fichiers de l'archive ingérés."
                results.append(zip_report)
                archiver.cleanup(extract_dir)
            finally:
                zip_path.unlink(missing_ok=True)
            continue

        # Cas normal : Fichier unique
        if suffix not in SUPPORTED_EXT:
            results.append({"file": f.filename, "skipped": True, "reason": "format non supporté"})
            continue

        data = await f.read()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            result = await ingest_single_file(tmp_path, tags, with_summary=False)
            result["file"] = f.filename
            results.append(result)
        except Exception as e:
            results.append({"file": f.filename, "skipped": True, "reason": str(e)})
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    return {
        "message": f"Traitement terminé pour {len(files)} élément(s).",
        "results": results,
    }


@router.post("/ingest-folder")
async def ingest_folder(request: IngestFolderRequest):
    """
    Ingère récursivement un dossier entier.
    Ignore automatiquement : .git, node_modules, .venv, __pycache__, etc.
    Idéal pour nourrir le cerveau avec un projet complet.
    """
    folder = Path(request.folder_path)
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"Dossier introuvable : {request.folder_path}")

    allowed_ext = set(request.extensions) if request.extensions else SUPPORTED_EXT

    report = {
        "folder": str(folder),
        "processed": [],
        "skipped_dedup": [],
        "skipped_unsupported": 0,
        "errors": [],
        "total_files_ingested": 0,
        "total_chunks": 0,
    }

    files = [
        f for f in folder.rglob("*")
        if f.is_file()
        and f.suffix.lower() in allowed_ext
        and not any(d in f.parts for d in IGNORED_DIRS)
    ]

    # Limiter le nombre de fichiers
    if request.max_files and len(files) > request.max_files:
        files = files[:request.max_files]

    for file_path in files:
        try:
            result = await ingest_single_file(str(file_path), request.tags, request.with_summary)
            if result.get("skipped"):
                report["skipped_dedup"].append({
                    "file": result["file"],
                    "reason": result.get("reason", ""),
                })
            else:
                report["processed"].append({
                    "file": result["file"],
                    "chunks": result["chunks_created"],
                    "summary": (result.get("summary") or "mode rapide")[:120],
                })
                report["total_files_ingested"] += 1
                report["total_chunks"] += result["chunks_created"]
        except Exception as e:
            report["errors"].append({"file": str(file_path), "error": str(e)})

    report["message"] = (
        f"{report['total_files_ingested']} fichier(s) ingéré(s) — "
        f"{len(report['skipped_dedup'])} déjà connus — "
        f"{report['total_chunks']} fragments créés."
    )
    return report


@router.post("/watch")
async def watch_folder(request: WatchRequest):
    """
    Démarre la surveillance automatique d'un dossier.
    Chaque fichier créé ou modifié est automatiquement ingéré dans la mémoire.
    """
    folder = Path(request.folder_path)
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Dossier introuvable : {request.folder_path}")
    loop = asyncio.get_event_loop()
    start_watcher(str(folder), loop)
    return {
        "message": f"Surveillance active sur : {folder}",
        "path": str(folder),
        "active": True,
        "info": "Chaque fichier modifié ou créé sera automatiquement ingéré.",
    }


@router.delete("/watch")
async def stop_watching():
    """Arrête la surveillance automatique."""
    stop_watcher()
    return {"message": "Surveillance arrêtée."}


@router.get("/watch/status")
async def watcher_status():
    """Statut du watcher et dernières ingestions automatiques."""
    return get_status()


@router.get("/report")
async def ingestion_report():
    """Rapport complet des ingestions automatiques."""
    log = get_log()
    return {
        "total_events": len(log),
        "log": log[-50:],
    }

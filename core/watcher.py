"""
Watcher de fichiers — le cerveau surveille tes dossiers et apprend automatiquement.
Utilise Watchdog pour détecter les créations et modifications de fichiers.

Architecture :
Ce module dépend uniquement de 'core.ingestion'.
Aucune dépendance vers 'routers' n'est autorisée.
"""
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import corrigé : Plus de dépendance vers routers.files
from core.ingestion import ingest_single_file
from core.consciousness import consciousness

logger = logging.getLogger("makenbrain.watcher")

SUPPORTED_EXT = {".txt", ".md", ".py", ".java", ".js", ".ts", ".jsx", ".tsx", ".json", ".pdf"}
IGNORED_DIRS  = {".git", "node_modules", ".venv", "__pycache__", "brain_data", ".idea", "dist", "build", "target"}

# État global du watcher
_observer: Observer | None = None
_loop: asyncio.AbstractEventLoop | None = None
_watched_paths: dict[str, str] = {}
_ingestion_log: list[dict] = []


class BrainEventHandler(FileSystemEventHandler):
    """Détecte les changements de fichiers et déclenche l'ingestion."""

    def on_created(self, event):
        if not event.is_directory:
            self._schedule(event.src_path, "créé")

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule(event.src_path, "modifié")

    def _schedule(self, path: str, event_type: str):
        p = Path(path)
        if p.suffix not in SUPPORTED_EXT:
            return
        if any(d in p.parts for d in IGNORED_DIRS):
            return
        if _loop and _loop.is_running():
            asyncio.run_coroutine_threadsafe(
                _auto_ingest(path, event_type), _loop
            )


async def _auto_ingest(file_path: str, event_type: str) -> None:
    """Ingère automatiquement un fichier modifié via le core centralisé."""
    try:
        # Appel direct à la fonction centrale
        result = await ingest_single_file(file_path)
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "file": Path(file_path).name,
            "path": file_path,
            "event": event_type,
            "chunks_created": result.get("chunks_created", 0),
            "skipped": result.get("skipped", False),
            "reason": result.get("reason", ""),
            "status": "ok",
        }
        _ingestion_log.append(entry)
        
        if not result.get("skipped"):
            logger.info(f"Auto-ingéré [{event_type}]: {Path(file_path).name} ({result.get('chunks_created', 0)} fragments)")
            
            # --- CONNECTION AU COEUR DE CONSCIENCE ---
            asyncio.create_task(consciousness.observe_event(
                event_type=f"file_{event_type}",
                data={
                    "file": Path(file_path).name,
                    "path": file_path,
                    "summary": result.get("summary", "Pas de résumé disponible")
                }
            ))
            # ------------------------------------------
    except Exception as e:
        _ingestion_log.append({
            "timestamp": datetime.now().isoformat(),
            "file": Path(file_path).name,
            "path": file_path,
            "event": event_type,
            "status": "erreur",
            "error": str(e),
        })
        logger.error(f"Erreur auto-ingestion {file_path}: {e}")


def start_watcher(path: str, loop: asyncio.AbstractEventLoop) -> bool:
    global _observer, _loop
    _loop = loop
    if _observer and _observer.is_alive():
        _observer.stop()
        _observer.join()
    handler = BrainEventHandler()
    _observer = Observer()
    _observer.schedule(handler, path, recursive=True)
    _observer.start()
    _watched_paths[path] = "surveillance active"
    logger.info(f"Surveillance démarrée : {path}")
    return True


def stop_watcher() -> None:
    global _observer
    if _observer and _observer.is_alive():
        _observer.stop()
        _observer.join()
        _observer = None
    _watched_paths.clear()


def get_status() -> dict:
    return {
        "active": _observer is not None and _observer.is_alive(),
        "watched_paths": _watched_paths,
        "total_ingestions": len(_ingestion_log),
        "recent_events": _ingestion_log[-10:],
    }


def get_log() -> list:
    return _ingestion_log
import zipfile
import shutil
from pathlib import Path
from typing import List, Optional

from core.sandbox import resolve_sandbox_path

class Archiver:
    def __init__(self, supported_extensions: Optional[List[str]] = None):
        self.supported_extensions = supported_extensions or {
            ".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".md", ".txt", ".java", 
            ".xml", ".yml", ".yaml", ".sql", ".html", ".css", ".c", ".cpp", ".h"
        }
        self.ignored_dirs = {
            ".git", "node_modules", ".venv", "__pycache__", "brain_data", 
            ".idea", "dist", "build", "target", ".vscode"
        }

    def extract_zip(self, zip_path: Path) -> Path:
        """Extrait un fichier ZIP dans un dossier temporaire et retourne le chemin."""
        uploads_root = resolve_sandbox_path("uploads", must_exist=False)
        uploads_root.mkdir(parents=True, exist_ok=True)
        temp_dir = uploads_root / f"makenbrain_zip_{zip_path.stem}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                target = (temp_dir / member.filename).resolve()
                if temp_dir not in target.parents and target != temp_dir:
                    raise ValueError("Archive ZIP dangereuse: chemin hors extraction.")
            zip_ref.extractall(temp_dir)
        return temp_dir

    def list_files(self, directory: Path) -> List[Path]:
        """Liste récursivement les fichiers supportés dans un dossier."""
        files = []
        for f in directory.rglob("*"):
            if f.is_file() and f.suffix.lower() in self.supported_extensions:
                if not any(ignored in f.parts for ignored in self.ignored_dirs):
                    files.append(f)
        return files

    def cleanup(self, directory: Path):
        """Supprime un dossier temporaire."""
        if directory.exists() and "makenbrain_zip_" in directory.name:
            shutil.rmtree(directory)

# Singleton
archiver = Archiver()

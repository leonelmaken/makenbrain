import zipfile
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

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
        temp_dir = Path(tempfile.mkdtemp(prefix="makenbrain_zip_"))
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
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

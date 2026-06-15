"""
Gestion des Permissions — makenBrain
Gère les autorisations d'accès aux fichiers et dossiers pour le cerveau.
"""
import os
from pathlib import Path

class PermissionManager:
    def __init__(self):
        # Liste des chemins (fichiers ou dossiers) autorisés pour cette session
        self.allowed_paths = set()

    def grant_access(self, path: str) -> bool:
        """Ajoute un chemin à la liste des autorisations."""
        abs_path = str(Path(path).absolute())
        self.allowed_paths.add(abs_path)
        return True

    def revoke_access(self, path: str):
        """Retire un chemin des autorisations."""
        abs_path = str(Path(path).absolute())
        if abs_path in self.allowed_paths:
            self.allowed_paths.remove(abs_path)

    def is_allowed(self, path: str) -> bool:
        """Vérifie si l'accès à un chemin est autorisé."""
        abs_path = str(Path(path).absolute())
        # Vérification directe ou si le chemin est dans un dossier autorisé
        for allowed in self.allowed_paths:
            if abs_path.startswith(allowed):
                return True
        return False

    def list_permissions(self):
        return list(self.allowed_paths)

# Singleton
permission_manager = PermissionManager()

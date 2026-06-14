"""
Déduplication par hash SHA256.
Le cerveau ne réapprend jamais ce qu'il sait déjà.
"""
import hashlib
import json
from pathlib import Path

_HASH_FILE = Path("brain_data/known_hashes.json")
_known_hashes: set[str] = set()


def _load_hashes() -> None:
    global _known_hashes
    if _HASH_FILE.exists():
        try:
            data = json.loads(_HASH_FILE.read_text(encoding="utf-8"))
            _known_hashes = set(data)
        except Exception:
            _known_hashes = set()


def _save_hashes() -> None:
    _HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    _HASH_FILE.write_text(
        json.dumps(list(_known_hashes), ensure_ascii=False),
        encoding="utf-8",
    )


def compute_hash(content: str) -> str:
    """Calcule le hash SHA256 d'un contenu."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def is_duplicate(content: str) -> bool:
    """Retourne True si le contenu est déjà connu du cerveau."""
    return compute_hash(content) in _known_hashes


def register(content: str) -> str:
    """Enregistre un contenu comme connu. Retourne son hash."""
    h = compute_hash(content)
    _known_hashes.add(h)
    _save_hashes()
    return h


def get_stats() -> dict:
    return {"known_hashes": len(_known_hashes)}


# Chargement au démarrage
_load_hashes()

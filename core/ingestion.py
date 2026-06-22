"""
MakenBrain - Ingestion Layer
============================

Ce module centralise TOUTE la logique d'ingestion de fichiers.
Il sert de point d'entrée unique pour :
- Le router API (/files)
- Le watcher automatique (core/watcher.py)
- Les agents futurs

Règle d'architecture stricte :
Aucun import depuis 'routers' n'est autorisé ici.
Toutes les dépendances doivent venir de 'core' ou de bibliothèques tierces.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path


# Dépendances Core (pas de routers !)
from core.dedup import is_duplicate, register
from core.memory import add_memory
from core.permissions import permission_manager
from core.summarizer import summarize

logger = logging.getLogger("makenbrain.ingestion")

# --- Configuration Globale ---
SUPPORTED_EXT = {
    ".ts", ".tsx", ".js", ".jsx", ".json", ".md", ".txt", ".csv", ".log",
    ".java", ".xml", ".yml", ".yaml", ".properties", ".sql", ".gradle", ".http",
    ".pdf", ".html", ".env.example", ".kt", ".swift",
}

# --- Utilitaires de Parsing ---

def chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    """
    Découpe un texte en fragments cohérents basés sur les phrases.
    
    Préserve l'intégrité des phrases pour améliorer la qualité des embeddings.
    
    Args:
        text: Texte brut à découper.
        chunk_size: Taille maximale approximative d'un fragment.
        
    Returns:
        Liste de chaînes de caractères (fragments).
    """
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
    """
    Extrait le texte d'un fichier PDF via PyMuPDF (fitz).
    
    Args:
        file_path: Chemin vers le fichier PDF.
        
    Returns:
        Texte extrait concaténé.
        
    Raises:
        ImportError: Si fitz n'est pas installé.
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF (fitz) est requis pour lire les PDF. Installez-le via 'pip install PyMuPDF'.")
        
    doc = fitz.open(file_path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def read_text_file(file_path: str) -> str:
    """
    Lit un fichier texte avec gestion robuste des encodages et nettoyage HTML.
    
    Essaie plusieurs encodages courants. Si le fichier est détecté comme HTML,
    il nettoie les balises script/style et extrait le texte visible.
    
    Args:
        file_path: Chemin vers le fichier texte.
        
    Returns:
        Contenu textuel nettoyé.
    """
    p = Path(file_path)
    raw = ""
    # Essai séquentiel des encodages courants
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            raw = p.read_text(encoding=enc, errors="ignore")
            break
        except Exception:
            continue
            
    # Nettoyage HTML si nécessaire
    if p.suffix.lower() == ".html" and raw:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            return soup.get_text(separator=" ", strip=True)
        except ImportError:
            logger.warning("BeautifulSoup non installé, retour du HTML brut.")
            
    return raw


# --- Logique Métier Principale ---

async def ingest_single_file(
    file_path: str, 
    tags: str = "", 
    with_summary: bool = True
) -> dict:
    """
    Ingestion complète d'un fichier sandboxé dans la mémoire du cerveau.
    
    Cette fonction orchestre :
    1. Validation des permissions et du chemin (Sandbox).
    2. Lecture du contenu (PDF ou Texte).
    3. Vérification de déduplication (Hash SHA256).
    4. Génération optionnelle de résumé (LLM local).
    5. Stockage en mémoire vectorielle (Chunks + Résumé).
    
    Args:
        file_path: Chemin absolu ou relatif (résolu par sandbox).
        tags: Tags associés au fichier pour la recherche.
        with_summary: Si True, génère un résumé avant ingestion.
        
    Returns:
        Dictionnaire contenant le statut, le nombre de chunks, le résumé, etc.
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        PermissionError: Si le fichier est hors sandbox.
        ValueError: Si le chemin n'est pas un fichier.
    """
    # 1. Validation Sandbox & Permissions
    p = permission_manager.require(file_path, action="ingest.file", endpoint="/files/ingest-file")
    
    if not p.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")
    if not p.is_file():
        raise ValueError(f"Ce chemin n'est pas un fichier : {file_path}")
    if p.suffix.lower() not in SUPPORTED_EXT:
        return {"chunks_created": 0, "skipped": True, "reason": "format non supporté", "file": p.name}
    
    # 2. Lecture du contenu
    text = read_pdf(str(p)) if p.suffix.lower() == ".pdf" else read_text_file(str(p))
    
    if not text or len(text.strip()) < 20:
        return {"chunks_created": 0, "skipped": True, "reason": "contenu vide ou trop court", "file": p.name}
    
    # 3. Déduplication
    if is_duplicate(text):
        return {"chunks_created": 0, "skipped": True, "reason": "déjà connu du cerveau", "file": p.name}
    
    register(text)
    
       # 4. Résumé (Optionnel)
    summary = ""

    if with_summary:
        try:
            summary = await summarize(text, title=p.name)
        except Exception as e:
            logger.warning(f"Échec du résumé pour {p.name}: {e}")
            summary = "[Résumé indisponible]"

        # 5. Stockage du résumé uniquement si demandé
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

    # 6. Chunking et stockage des fragments
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
        "ids": ids,
    }
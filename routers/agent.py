"""
Agent Fichiers — Phase 4
Le cerveau peut lire, créer, modifier et analyser tes fichiers.
Toujours avec backup avant modification. Jamais destructif sans confirmation.
"""
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.providers import groq_generate, detect_provider
from core.memory import add_memory, search_memory
from core.dedup import is_duplicate, register

router = APIRouter()

BACKUP_DIR   = Path("brain_data/backups")
PROTECTED    = {"Windows", "System32", "Program Files", "Python314", ".git"}
ENCODINGS    = ("utf-8", "utf-8-sig", "latin-1", "cp1252")


# ── Schemas ───────────────────────────────────────────────────────────────────

class ReadRequest(BaseModel):
    file_path: str
    analyze:   bool   = True      # Demander une analyse au LLM
    provider:  str    = "groq"    # groq recommandé pour l'analyse
    auto_ingest: bool = True      # Ingérer en mémoire automatiquement

    model_config = {"json_schema_extra": {"example": {
        "file_path": "C:/Users/MAKEN/Documents/Projet/SmartBudget Africa/backend/src/main/java/com/smartbudget/security/JwtFilter.java",
        "analyze": True,
        "provider": "groq"
    }}}


class WriteRequest(BaseModel):
    file_path: str
    content:   Optional[str] = None   # Contenu direct OU
    prompt:    Optional[str] = None   # Générer avec LLM
    provider:  str = "groq"
    auto_ingest: bool = True

    model_config = {"json_schema_extra": {"example": {
        "file_path": "C:/Users/MAKEN/Documents/Projet/SmartBudget Africa/backend/src/main/java/com/smartbudget/service/ReportService.java",
        "prompt": "Génère un service Spring Boot pour générer des rapports PDF des transactions mensuelles.",
        "provider": "groq"
    }}}


class ModifyRequest(BaseModel):
    file_path:   str
    instruction: str                   # ex: "Ajoute la gestion des erreurs JWT"
    provider:    str    = "groq"
    backup:      bool   = True         # Toujours True recommandé

    model_config = {"json_schema_extra": {"example": {
        "file_path": "C:/Users/MAKEN/Documents/Projet/SmartBudget Africa/backend/src/main/java/com/smartbudget/security/JwtFilter.java",
        "instruction": "Ajoute une meilleure gestion des erreurs avec des messages clairs en cas de token expiré ou invalide.",
        "provider": "groq"
    }}}


class SuggestRequest(BaseModel):
    file_path: str
    goal:      Optional[str] = ""      # "improve security", "add tests", etc.
    provider:  str = "groq"

    model_config = {"json_schema_extra": {"example": {
        "file_path": "C:/Users/MAKEN/Documents/Projet/SmartBudget Africa/backend/src/main/java/com/smartbudget/service/TontineService.java",
        "goal": "améliorer les performances et la gestion des erreurs",
        "provider": "groq"
    }}}


class DeleteRequest(BaseModel):
    file_path: str
    confirm:   bool = False

    model_config = {"json_schema_extra": {"example": {
        "file_path": "C:/Users/MAKEN/Documents/Projet/test_temp.txt",
        "confirm": True
    }}}


class ExploreRequest(BaseModel):
    folder_path: str
    analyze:     bool = True

    model_config = {"json_schema_extra": {"example": {
        "folder_path": "C:/Users/MAKEN/Documents/Projet/SmartBudget Africa/backend/src",
        "analyze": True
    }}}


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _read_file(path: Path) -> str:
    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return ""


def _is_protected(path: Path) -> bool:
    return any(p in PROTECTED for p in path.parts)


def _backup(path: Path) -> str:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"{path.stem}_{ts}{path.suffix}.bak"
    shutil.copy2(path, dst)
    return str(dst)


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/read")
async def read_file(req: ReadRequest):
    """
    Lit un fichier et l'analyse avec le LLM.
    Le cerveau explique ce que fait le fichier, les points forts, les risques.
    Option auto_ingest=True → stocke le fichier en mémoire.
    """
    p = Path(req.file_path)
    if not p.exists():
        raise HTTPException(404, f"Fichier introuvable : {req.file_path}")
    if not p.is_file():
        raise HTTPException(400, "Ce chemin n'est pas un fichier.")

    content = _read_file(p)
    if not content.strip():
        raise HTTPException(400, "Fichier vide ou illisible.")

    # Ingestion mémoire
    ingest_result = None
    if req.auto_ingest and not is_duplicate(content):
        register(content)
        from routers.files import chunk_text
        for i, chunk in enumerate(chunk_text(content)):
            await add_memory(chunk, metadata={
                "source": str(p), "title": p.name,
                "tags": "agent,fichier", "type": "agent_read",
            })
        ingest_result = "ingéré en mémoire"
    elif req.auto_ingest:
        ingest_result = "déjà connu"

    # Analyse LLM
    analysis = None
    if req.analyze:
        prompt = (
            f"Analyse ce fichier de code ({p.name}) et donne-moi :\n"
            f"1. Ce que fait ce fichier (en 2-3 phrases)\n"
            f"2. Les points forts\n"
            f"3. Les problèmes potentiels ou améliorations suggérées\n\n"
            f"```{p.suffix[1:] if p.suffix else 'text'}\n"
            f"{content[:4000]}\n```"
        )
        # Chercher contexte mémoire lié
        related = await search_memory(p.name, n_results=3)
        ctx = "\n".join([m["content"][:200] for m in related if m["distance"] < 0.7])
        analysis = await groq_generate(prompt, ctx)

    return {
        "file":        p.name,
        "path":        str(p),
        "size_chars":  len(content),
        "lines":       content.count("\n") + 1,
        "extension":   p.suffix,
        "ingest":      ingest_result,
        "analysis":    analysis,
        "content_preview": content[:500] + "..." if len(content) > 500 else content,
    }


@router.post("/write")
async def write_file(req: WriteRequest):
    """
    Crée un nouveau fichier.
    Mode 1 : contenu direct (content)
    Mode 2 : généré par LLM (prompt) → le cerveau écrit le fichier tout seul
    """
    p = Path(req.file_path)
    if _is_protected(p):
        raise HTTPException(403, "Chemin protégé — opération refusée.")

    # Déterminer le contenu
    if req.content:
        final_content = req.content
        generated = False
    elif req.prompt:
        # Chercher contexte dans la mémoire
        related = await search_memory(req.prompt, n_results=5)
        ctx = "\n".join([m["content"][:300] for m in related if m["distance"] < 0.75])

        ext = p.suffix.lstrip(".") or "text"
        sys_prompt = (
            f"Tu es MakenBrain, l'assistant de MAKEN. "
            f"Génère uniquement le contenu du fichier {p.name} ({ext}), "
            f"sans explication, sans balises markdown, juste le code/texte brut."
        )
        full_prompt = req.prompt
        if ctx:
            full_prompt = f"Contexte du projet :\n{ctx}\n\nTâche : {req.prompt}"

        final_content = await groq_generate(full_prompt, system=sys_prompt)
        # Nettoyer les balises markdown si présentes
        import re
        final_content = re.sub(r"^```[\w]*\n?", "", final_content.strip())
        final_content = re.sub(r"\n?```$", "", final_content.strip())
        generated = True
    else:
        raise HTTPException(400, "Fournis 'content' ou 'prompt'.")

    # Backup si le fichier existe déjà
    backup_path = None
    if p.exists():
        backup_path = _backup(p)

    _write_file(p, final_content)

    # Ingestion
    if req.auto_ingest:
        await add_memory(
            f"[CRÉÉ] {p.name} : {final_content[:500]}",
            metadata={"source": str(p), "title": p.name, "tags": "agent,write", "type": "agent_write"}
        )

    return {
        "created":       True,
        "file":          p.name,
        "path":          str(p),
        "generated_by":  "LLM (Groq)" if generated else "contenu direct",
        "lines_written": final_content.count("\n") + 1,
        "backup":        backup_path,
    }


@router.post("/modify")
async def modify_file(req: ModifyRequest):
    """
    Modifie un fichier existant selon une instruction en langage naturel.
    Le LLM lit le fichier, applique la modification, réécrit le tout.
    Backup automatique avant toute modification.
    """
    p = Path(req.file_path)
    if not p.exists():
        raise HTTPException(404, f"Fichier introuvable : {req.file_path}")
    if _is_protected(p):
        raise HTTPException(403, "Chemin protégé — opération refusée.")

    original = _read_file(p)
    if not original.strip():
        raise HTTPException(400, "Fichier vide ou illisible.")

    # Backup obligatoire
    backup_path = _backup(p) if req.backup else None

    # Chercher contexte lié dans la mémoire
    related = await search_memory(req.instruction, n_results=3)
    ctx = "\n".join([m["content"][:200] for m in related if m["distance"] < 0.75])

    prompt = (
        f"Voici un fichier ({p.name}) à modifier.\n\n"
        f"Instruction : {req.instruction}\n\n"
        f"Fichier original :\n"
        f"```{p.suffix.lstrip('.')}\n{original[:6000]}\n```\n\n"
        f"Retourne UNIQUEMENT le fichier modifié complet, sans explication, "
        f"sans balises markdown, juste le code brut."
    )

    modified = await groq_generate(prompt, ctx)

    # Nettoyer les balises markdown
    import re
    modified = re.sub(r"^```[\w]*\n?", "", modified.strip())
    modified = re.sub(r"\n?```$", "", modified.strip())

    _write_file(p, modified)

    # Ingérer la modification en mémoire
    await add_memory(
        f"[MODIFIÉ] {p.name} — {req.instruction}",
        metadata={"source": str(p), "title": p.name, "tags": "agent,modify", "type": "agent_modify"}
    )

    return {
        "modified":      True,
        "file":          p.name,
        "instruction":   req.instruction,
        "backup":        backup_path,
        "lines_before":  original.count("\n") + 1,
        "lines_after":   modified.count("\n") + 1,
        "diff_lines":    (modified.count("\n") + 1) - (original.count("\n") + 1),
    }


@router.post("/suggest")
async def suggest(req: SuggestRequest):
    """
    Analyse un fichier et propose des améliorations SANS le modifier.
    100% safe — lecture seule + suggestions LLM.
    """
    p = Path(req.file_path)
    if not p.exists():
        raise HTTPException(404, f"Fichier introuvable : {req.file_path}")

    content = _read_file(p)
    if not content.strip():
        raise HTTPException(400, "Fichier vide ou illisible.")

    # Contexte mémoire + graphe
    related = await search_memory(p.name + " " + (req.goal or ""), n_results=5)
    ctx = "\n".join([m["content"][:200] for m in related if m["distance"] < 0.75])

    try:
        from core.graph import neuron_graph
        from core.extractor import extract_fast
        concepts = extract_fast(content)
        for c in concepts[:3]:
            r = neuron_graph.explore(c.get("name",""), depth=1)
            if r.get("found"):
                neighbors = list(r.get("neighbors",{}).keys())[:4]
                if neighbors:
                    ctx += f"\nConcepts liés dans le graphe : {', '.join(neighbors)}"
    except Exception:
        pass

    goal_str = f"\nObjectif spécifique : {req.goal}" if req.goal else ""
    prompt = (
        f"Analyse ce fichier ({p.name}) et propose des améliorations concrètes.{goal_str}\n\n"
        f"Donne :\n"
        f"1. Résumé du fichier actuel\n"
        f"2. Problèmes identifiés (sécurité, performance, lisibilité)\n"
        f"3. Améliorations recommandées (avec exemples de code si pertinent)\n"
        f"4. Priorité : ce qu'il faut faire en premier\n\n"
        f"```{p.suffix.lstrip('.')}\n{content[:4000]}\n```"
    )

    suggestions = await groq_generate(prompt, ctx)

    return {
        "file":        p.name,
        "path":        str(p),
        "goal":        req.goal or "amélioration générale",
        "suggestions": suggestions,
        "modified":    False,
        "safe":        True,
    }


@router.delete("/delete")
async def delete_file(req: DeleteRequest):
    """Supprime un fichier. confirm=True obligatoire."""
    if not req.confirm:
        return {
            "deleted": False,
            "message": "Ajoute confirm=true pour confirmer la suppression.",
            "file": req.file_path,
        }
    p = Path(req.file_path)
    if not p.exists():
        raise HTTPException(404, "Fichier introuvable.")
    if _is_protected(p):
        raise HTTPException(403, "Chemin protégé — opération refusée.")
    backup = _backup(p)
    p.unlink()
    return {
        "deleted": True,
        "file":    p.name,
        "backup":  backup,
        "message": f"Fichier supprimé. Backup sauvegardé : {backup}",
    }


@router.post("/explore")
async def explore_folder(req: ExploreRequest):
    """
    Liste et analyse un dossier.
    Avec analyze=True, le cerveau donne une vue d'ensemble du projet.
    """
    folder = Path(req.folder_path)
    if not folder.exists():
        raise HTTPException(404, f"Dossier introuvable : {req.folder_path}")

    SKIP = {".git", "node_modules", ".venv", "__pycache__", "build", "dist", "target", ".idea"}
    EXT  = {".java", ".ts", ".tsx", ".js", ".jsx", ".py", ".md", ".sql", ".yml", ".yaml", ".json"}

    files_info = []
    total_lines = 0

    for f in sorted(folder.rglob("*")):
        if not f.is_file():
            continue
        if any(d in f.parts for d in SKIP):
            continue
        if f.suffix not in EXT:
            continue
        lines = 0
        try:
            lines = _read_file(f).count("\n") + 1
        except Exception:
            pass
        total_lines += lines
        files_info.append({
            "file": f.name,
            "path": str(f.relative_to(folder)),
            "ext":  f.suffix,
            "lines": lines,
        })

    by_ext: dict[str, int] = {}
    for fi in files_info:
        by_ext[fi["ext"]] = by_ext.get(fi["ext"], 0) + 1

    overview = None
    if req.analyze and files_info:
        file_list = "\n".join([f"- {fi['path']} ({fi['lines']} lignes)" for fi in files_info[:30]])
        prompt = (
            f"Voici la structure du dossier '{folder.name}' :\n\n{file_list}\n\n"
            f"Donne-moi :\n"
            f"1. Ce que fait ce projet/module en 2-3 phrases\n"
            f"2. L'organisation des fichiers (points forts)\n"
            f"3. Ce qui manque ou pourrait être amélioré"
        )
        overview = await groq_generate(prompt)

    return {
        "folder":       folder.name,
        "total_files":  len(files_info),
        "total_lines":  total_lines,
        "by_extension": by_ext,
        "files":        files_info[:50],
        "overview":     overview,
    }

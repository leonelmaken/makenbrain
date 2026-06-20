"""
Router Analysis — Phase 4 avancée
- Analyse complète backend + frontend séparément
- Détection ciblée des erreurs React/TypeScript
- Workflow de validation : propose → MAKEN valide → cerveau agit
- Toutes les actions sont documentées dans le changelog
"""
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.audit import audit_event
from core.permissions import permission_manager
from core.providers import groq_generate
from core.changelog import (
    create_proposal, get_proposals, get_proposal,
    update_proposal_status, log_action, get_changelog,
)

router = APIRouter()

SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__", "build",
             "dist", "target", ".idea", ".expo", "coverage"}

JAVA_EXT  = {".java", ".xml", ".yml", ".yaml", ".properties", ".sql"}
FRONT_EXT = {".ts", ".tsx", ".js", ".jsx", ".json", ".css", ".md"}


def _require_analysis_path(path: str, *, endpoint: str) -> Path:
    try:
        return permission_manager.require(path, action="analysis.read", endpoint=endpoint)
    except PermissionError as exc:
        audit_event(action="analysis.read", tool="analysis", endpoint=endpoint, file_path=path, result=str(exc), success=False)
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        audit_event(action="analysis.read", tool="analysis", endpoint=endpoint, file_path=path, result=str(exc), success=False)
        raise HTTPException(status_code=403, detail=str(exc))


# ── Schemas ───────────────────────────────────────────────────────────────────

class AnalyzeProjectRequest(BaseModel):
    backend_path:  Optional[str] = None
    frontend_path: Optional[str] = None
    mobile_path:   Optional[str] = None

    model_config = {"json_schema_extra": {"example": {
        "backend_path":  "C:/Users/MAKEN/Documents/Projet/SmartBudget Africa/backend",
        "frontend_path": "C:/Users/MAKEN/Documents/Projet/SmartBudget Africa/frontend",
    }}}


class FrontendErrorRequest(BaseModel):
    frontend_path: str
    include_mobile: bool = False

    model_config = {"json_schema_extra": {"example": {
        "frontend_path": "C:/Users/MAKEN/Documents/Projet/SmartBudget Africa/frontend/src",
        "include_mobile": False
    }}}


class ProposeRequest(BaseModel):
    title:       str
    description: str
    file_path:   Optional[str] = None
    action_type: str = "modify"        # modify | create | delete | refactor
    instruction: Optional[str] = None
    risks:       Optional[list[str]] = None

    model_config = {"json_schema_extra": {"example": {
        "title":       "Améliorer la gestion d'erreurs dans AuthService",
        "description": "Ajouter des messages d'erreur clairs pour JWT expiré et token invalide",
        "file_path":   "C:/Users/MAKEN/Documents/Projet/SmartBudget Africa/backend/src/main/java/com/smartbudget/security/JwtFilter.java",
        "action_type": "modify",
        "instruction": "Ajouter try-catch avec messages d'erreur explicites pour chaque type d'exception JWT"
    }}}


class ApproveRequest(BaseModel):
    proposal_id: str
    comment:     Optional[str] = ""

    model_config = {"json_schema_extra": {"example": {
        "proposal_id": "a3f9c2d1",
        "comment":     "OK, vas-y mais garde le backup"
    }}}


class RejectRequest(BaseModel):
    proposal_id: str
    reason:      Optional[str] = ""


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _read(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return ""


def _collect_files(folder: Path, extensions: set[str], max_files: int = 60) -> list[Path]:
    files = []
    for f in sorted(folder.rglob("*")):
        if len(files) >= max_files:
            break
        if not f.is_file():
            continue
        if any(d in f.parts for d in SKIP_DIRS):
            continue
        if f.suffix in extensions:
            files.append(f)
    return files


def _detect_frontend_issues(files: list[Path]) -> list[dict]:
    """
    Détecte les erreurs et mauvaises pratiques dans les fichiers React/TypeScript
    par analyse statique (sans LLM, ultra-rapide).
    """
    issues = []

    for f in files:
        content = _read(f)
        if not content:
            continue
        lines = content.splitlines()
        name  = f.name
        rel   = str(f)

        # 1. Console.log oublié en production
        for i, line in enumerate(lines, 1):
            if "console.log" in line and "// ok" not in line.lower():
                issues.append({
                    "file": name, "path": rel, "line": i,
                    "type": "debug", "severity": "low",
                    "message": f"console.log() oublié (ligne {i})",
                    "snippet": line.strip()[:80],
                })

        # 2. Type any explicite
        any_matches = [(i+1, l) for i, l in enumerate(lines) if ": any" in l or "as any" in l]
        for ln, snippet in any_matches[:3]:
            issues.append({
                "file": name, "path": rel, "line": ln,
                "type": "typescript", "severity": "medium",
                "message": f"Type 'any' utilisé (ligne {ln}) — perd la sécurité TypeScript",
                "snippet": snippet.strip()[:80],
            })

        # 3. TODO / FIXME non résolus
        todos = [(i+1, l) for i, l in enumerate(lines)
                 if re.search(r'\b(TODO|FIXME|HACK|XXX)\b', l, re.I)]
        for ln, snippet in todos[:3]:
            issues.append({
                "file": name, "path": rel, "line": ln,
                "type": "todo", "severity": "info",
                "message": f"TODO/FIXME non résolu (ligne {ln})",
                "snippet": snippet.strip()[:80],
            })

        # 4. Imports inutilisés (heuristique simple)
        import_lines = [(i+1, l) for i, l in enumerate(lines) if l.strip().startswith("import ")]
        for ln, imp in import_lines:
            match = re.search(r'import\s+\{([^}]+)\}', imp)
            if match:
                names = [n.strip() for n in match.group(1).split(",")]
                for n in names:
                    if n and content.count(n) <= 1:
                        issues.append({
                            "file": name, "path": rel, "line": ln,
                            "type": "import", "severity": "low",
                            "message": f"Import potentiellement inutilisé : '{n}'",
                            "snippet": imp.strip()[:80],
                        })

        # 5. Fichiers trop longs (> 300 lignes)
        if len(lines) > 300:
            issues.append({
                "file": name, "path": rel, "line": 0,
                "type": "architecture", "severity": "medium",
                "message": f"Fichier trop long ({len(lines)} lignes) — envisage de le découper",
                "snippet": "",
            })

        # 6. URLs hardcodées
        hardcoded = [(i+1, l) for i, l in enumerate(lines)
                     if re.search(r'https?://localhost', l) or
                        re.search(r"['\"]http://\d", l)]
        for ln, snippet in hardcoded[:2]:
            issues.append({
                "file": name, "path": rel, "line": ln,
                "type": "config", "severity": "high",
                "message": f"URL hardcodée détectée (ligne {ln}) — doit être en variable d'env",
                "snippet": snippet.strip()[:80],
            })

        # 7. useEffect sans dépendances
        effect_no_deps = [(i+1, l) for i, l in enumerate(lines)
                          if "useEffect(" in l and i+2 < len(lines)
                          and "[]" not in "".join(lines[i:i+3])
                          and "}" not in l]
        for ln, _ in effect_no_deps[:2]:
            issues.append({
                "file": name, "path": rel, "line": ln,
                "type": "react", "severity": "high",
                "message": f"useEffect sans tableau de dépendances (ligne {ln}) — risque de boucle infinie",
                "snippet": lines[ln-1].strip()[:80] if ln <= len(lines) else "",
            })

    # Trier par sévérité
    severity_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    return sorted(issues, key=lambda x: severity_order.get(x["severity"], 4))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/analyze-project")
async def analyze_project(req: AnalyzeProjectRequest):
    """
    Analyse complète du projet SmartBudget Africa.
    Backend et frontend analysés séparément par Groq 70B.
    Génère des propositions avec IDs pour validation ultérieure.
    Tout est documenté dans le changelog.
    """
    report = {"analyses": {}, "proposals_created": [], "timestamp": ""}
    from datetime import datetime
    report["timestamp"] = datetime.now().isoformat()

    # ── BACKEND ──────────────────────────────────────────────────────────────
    if req.backend_path:
        bk = _require_analysis_path(req.backend_path, endpoint="/analysis/analyze-project")
        if bk.exists():
            files = _collect_files(bk, JAVA_EXT, max_files=40)
            file_list = "\n".join([
                f"- {f.relative_to(bk)} ({_read(f).count(chr(10))+1} lignes)"
                for f in files
            ])
            # Lire quelques fichiers clés pour analyse profonde
            key_files_content = ""
            key_patterns = ["Service", "Controller", "Security", "Config", "Entity"]
            for f in files:
                if any(p in f.name for p in key_patterns) and len(key_files_content) < 5000:
                    content = _read(f)[:800]
                    key_files_content += f"\n\n--- {f.name} ---\n{content}"

            bk_prompt = (
                f"Analyse le backend Spring Boot SmartBudget Africa.\n\n"
                f"Structure ({len(files)} fichiers) :\n{file_list}\n\n"
                f"Fichiers clés :\n{key_files_content}\n\n"
                f"Donne-moi :\n"
                f"1. Architecture générale (forces et faiblesses)\n"
                f"2. Problèmes de sécurité détectés\n"
                f"3. Problèmes de performance potentiels\n"
                f"4. Top 3 améliorations prioritaires (avec justification)\n"
                f"5. Ce qui est bien fait (pour ne pas toucher)\n\n"
                f"Format : sections claires, concis, actionnable."
            )
            bk_analysis = await groq_generate(bk_prompt)
            report["analyses"]["backend"] = {
                "path":       str(bk),
                "files":      len(files),
                "analysis":   bk_analysis,
            }
            log_action("analyze", str(bk), "Analyse complète backend SmartBudget", "done")

    # ── FRONTEND ─────────────────────────────────────────────────────────────
    if req.frontend_path:
        fr = _require_analysis_path(req.frontend_path, endpoint="/analysis/analyze-project")
        if fr.exists():
            files = _collect_files(fr, FRONT_EXT, max_files=50)

            # Analyse statique des erreurs
            issues = _detect_frontend_issues(files)

            # Analyse LLM sur les fichiers clés
            key_content = ""
            key_patterns = ["App.", "auth", "axios", "context", "service", "hook"]
            for f in files:
                if any(p.lower() in f.name.lower() for p in key_patterns):
                    key_content += f"\n\n--- {f.name} ---\n{_read(f)[:600]}"
                    if len(key_content) > 5000:
                        break

            fr_prompt = (
                f"Analyse le frontend React/TypeScript SmartBudget Africa.\n\n"
                f"Structure : {len(files)} fichiers analysés\n"
                f"Erreurs détectées automatiquement : {len(issues)} issues\n\n"
                f"Fichiers clés :\n{key_content[:4000]}\n\n"
                f"Donne-moi :\n"
                f"1. Architecture frontend (points forts, problèmes)\n"
                f"2. Erreurs probables basées sur le code vu\n"
                f"3. Problèmes d'état/data flow\n"
                f"4. Sécurité frontend (tokens, localStorage, etc.)\n"
                f"5. Top 3 corrections prioritaires\n\n"
                f"Sois précis et cite les fichiers concernés."
            )
            fr_analysis = await groq_generate(fr_prompt)

            # Grouper les issues par type
            by_severity = {"high": [], "medium": [], "low": [], "info": []}
            for issue in issues:
                by_severity[issue["severity"]].append(issue)

            report["analyses"]["frontend"] = {
                "path":              str(fr),
                "files":             len(files),
                "static_issues":     len(issues),
                "issues_by_severity": {k: len(v) for k, v in by_severity.items()},
                "high_priority":     by_severity["high"],
                "medium_priority":   by_severity["medium"][:5],
                "analysis":          fr_analysis,
            }
            log_action("analyze", str(fr), f"Analyse frontend — {len(issues)} issues détectées", "done")

            # Créer des propositions automatiques pour les issues critiques
            high_issues = by_severity["high"]
            if high_issues:
                files_concerned = list({i["file"] for i in high_issues})
                proposal = create_proposal(
                    title=f"Corriger {len(high_issues)} erreur(s) critique(s) frontend",
                    description=(
                        f"Issues haute priorité détectées dans : {', '.join(files_concerned[:3])}\n"
                        + "\n".join([f"• {i['message']}" for i in high_issues[:5]])
                    ),
                    action_type="modify",
                    file_path=req.frontend_path,
                    action_data={"issues": high_issues},
                    risks=["Modification de fichiers frontend", "Backup automatique avant toute action"],
                )
                report["proposals_created"].append({
                    "id":      proposal["id"],
                    "title":   proposal["title"],
                    "action":  "En attente de ta validation → POST /analysis/approve",
                })

    return report


@router.post("/frontend-errors")
async def frontend_errors(req: FrontendErrorRequest):
    """
    Détection ciblée des erreurs frontend React/TypeScript.
    Analyse statique (instantanée) + analyse LLM (Groq).
    Aucune modification — lecture seule.
    """
    fr = _require_analysis_path(req.frontend_path, endpoint="/analysis/frontend-errors")
    if not fr.exists():
        raise HTTPException(404, f"Dossier introuvable : {req.frontend_path}")

    ext = FRONT_EXT | ({".tsx", ".ts"} if req.include_mobile else set())
    files = _collect_files(fr, ext, max_files=60)

    if not files:
        return {"message": "Aucun fichier frontend trouvé.", "issues": []}

    # Analyse statique
    issues = _detect_frontend_issues(files)

    # Analyse LLM — couvrir un maximum de fichiers pertinents
    KEY_PATTERNS = [
        "auth", "token", "axios", "api", "hook", "context",
        "service", "store", "screen", "page", "layout",
        "navigation", "router", "guard", "interceptor",
    ]
    key_files   = [f for f in files if any(p in f.name.lower() for p in KEY_PATTERNS)]
    other_files = [f for f in files if f not in key_files]
    ordered     = (key_files + other_files)[:15]

    llm_findings = ""
    if ordered:
        sections = []
        total_chars = 0
        for f in ordered:
            content = _read(f)
            if not content.strip():
                continue
            snippet = content[:1500]
            section = f"=== {f.name} ({len(content.splitlines())} lignes) ===\n{snippet}"
            if total_chars + len(section) > 12000:
                break
            sections.append(section)
            total_chars += len(section)

        combined  = "\n\n".join(sections)
        file_names = [f.name for f in ordered[:len(sections)]]

        prompt = (
            f"Tu es un expert React Native / TypeScript / Expo.\n"
            f"Analyse ces {len(sections)} fichiers du projet SmartBudget Africa "
            f"({', '.join(file_names)}).\n\n"
            f"Identifie TOUS les problèmes :\n"
            f"1. Erreurs TypeScript (types manquants, any implicite, types incorrects)\n"
            f"2. Bugs React/React Native (hooks mal utilisés, re-renders inutiles, memory leaks)\n"
            f"3. Problèmes de sécurité (tokens exposés, données sensibles en clair)\n"
            f"4. Architecture (couplage fort, responsabilités mélangées)\n"
            f"5. Performance (appels API inutiles, listes non optimisées, images lourdes)\n\n"
            f"Pour chaque problème : cite le fichier exact, la ligne si possible, "
            f"et propose une correction concrète.\n\n"
            f"--- FICHIERS ---\n\n{combined}"
        )
        llm_findings = await groq_generate(prompt)

    # Stats
    by_type: dict[str, int] = {}
    for issue in issues:
        by_type[issue["type"]] = by_type.get(issue["type"], 0) + 1

    log_action("analyze", req.frontend_path, f"Détection erreurs frontend — {len(issues)} issues", "done")

    return {
        "folder":        req.frontend_path,
        "files_analyzed": len(files),
        "total_issues":  len(issues),
        "by_type":       by_type,
        "critical":      [i for i in issues if i["severity"] == "high"],
        "medium":        [i for i in issues if i["severity"] == "medium"],
        "low":           [i for i in issues if i["severity"] in ("low", "info")][:10],
        "llm_findings":  llm_findings,
        "note":          "Aucune modification effectuée — analyse seule.",
    }


@router.post("/propose")
async def propose(req: ProposeRequest):
    """
    Crée une proposition en attente de validation.
    Le cerveau analyse, propose et attend ton approbation.
    Tu valides → POST /analysis/approve/{id}
    Tu refuses → POST /analysis/reject/{id}
    """
    # Analyser le fichier si fourni
    analysis = ""
    if req.file_path:
        p = _require_analysis_path(req.file_path, endpoint="/analysis/propose")
        if p.exists():
            content = p.read_text(encoding="utf-8", errors="ignore")[:3000]
            related = []
            try:
                from core.memory import search_memory
                related = await search_memory(req.title, n_results=3)
            except Exception:
                pass
            ctx = "\n".join([m["content"][:200] for m in related if m.get("distance", 1) < 0.75])

            analysis_prompt = (
                f"Analyse ce fichier ({p.name}) par rapport à cette tâche : {req.description}\n\n"
                f"Instruction prévue : {req.instruction or 'voir description'}\n\n"
                f"```\n{content}\n```\n\n"
                f"Dis-moi :\n"
                f"1. Ce que tu vas faire exactement\n"
                f"2. Les risques potentiels\n"
                f"3. Ce que tu ne toucheras PAS\n"
                f"4. Résultat attendu après modification"
            )
            analysis = await groq_generate(analysis_prompt, ctx)

    proposal = create_proposal(
        title=req.title,
        description=req.description,
        action_type=req.action_type,
        file_path=req.file_path or "",
        action_data={"instruction": req.instruction or ""},
        risks=req.risks or ["Backup automatique avant toute action"],
    )

    log_action("propose", req.file_path or "", req.title, "pending", proposal["id"])

    return {
        "proposal_id":   proposal["id"],
        "title":         proposal["title"],
        "status":        "En attente de ta validation",
        "analysis":      analysis,
        "next_steps": {
            "approve": f"POST /analysis/approve  {{\"proposal_id\": \"{proposal['id']}\"}}",
            "reject":  f"POST /analysis/reject   {{\"proposal_id\": \"{proposal['id']}\"}}",
            "list":    "GET /analysis/proposals",
        },
    }


@router.post("/approve")
async def approve(req: ApproveRequest):
    """
    Approuve et exécute une proposition.
    C'est TOI qui déclenches l'action — jamais le cerveau seul.
    """
    proposal = get_proposal(req.proposal_id)
    if not proposal:
        raise HTTPException(404, f"Proposition {req.proposal_id} introuvable.")
    if proposal["status"] != "pending":
        raise HTTPException(400, f"Proposition déjà {proposal['status']}.")

    update_proposal_status(req.proposal_id, "approved")

    # Exécuter l'action selon le type
    result = {"executed": False, "message": ""}

    if proposal["action_type"] == "modify" and proposal.get("file_path"):
        instruction = proposal["action_data"].get("instruction", "")
        if instruction:
            from routers.agent import ModifyRequest, modify_file
            mod_req = ModifyRequest(
                file_path=proposal["file_path"],
                instruction=instruction,
                provider="groq",
                backup=True,
                dry_run=False,
                apply_changes=True,
            )
            mod_result = await modify_file(mod_req)
            result = {"executed": True, **mod_result}

    update_proposal_status(req.proposal_id, "done")
    log_action(
        "approve", proposal.get("file_path", ""), proposal["title"],
        "done", req.proposal_id,
        {"comment": req.comment, "result": result}
    )

    return {
        "proposal_id": req.proposal_id,
        "title":       proposal["title"],
        "approved":    True,
        "executed":    result.get("executed", False),
        "result":      result,
        "logged":      True,
    }


@router.post("/reject")
async def reject(req: RejectRequest):
    """Rejette une proposition sans aucune action."""
    proposal = get_proposal(req.proposal_id)
    if not proposal:
        raise HTTPException(404, f"Proposition {req.proposal_id} introuvable.")

    update_proposal_status(req.proposal_id, "rejected")
    log_action("reject", "", proposal["title"], "rejected", req.proposal_id,
               {"reason": req.reason})

    return {
        "proposal_id": req.proposal_id,
        "rejected":    True,
        "reason":      req.reason,
        "message":     "Proposition rejetée. Aucune modification effectuée.",
    }


@router.get("/proposals")
async def list_proposals(status: Optional[str] = None):
    """Liste toutes les propositions (pending | approved | rejected | done)."""
    proposals = get_proposals(status)
    return {
        "total":     len(proposals),
        "filter":    status or "all",
        "proposals": proposals,
    }


@router.get("/changelog")
async def changelog(limit: int = 50):
    """Historique complet de toutes les actions du cerveau."""
    entries = get_changelog(limit)
    return {
        "total":   len(entries),
        "entries": entries,
    }

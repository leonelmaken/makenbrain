"""Endpoints identité et projets — isolés par rôle utilisateur.

Le profil local (user_profile) et les projets locaux (project_manager) sont
des singletons appartenant au SuperAdmin (Leonel / MAKEN).
Les utilisateurs non-SuperAdmin reçoivent un profil générique et une liste
de projets vide — aucune donnée SuperAdmin n'est jamais exposée.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from core.auth import require_chat_user
from core.user_profile import user_profile
from core.project_memory import project_manager
from models.user import User, UserRole

router = APIRouter()


class BioUpdateRequest(BaseModel):
    bio: str


class GoalRequest(BaseModel):
    description: str
    priority: int = 3


class ProjectCreateRequest(BaseModel):
    name: str
    description: str
    tags: Optional[List[str]] = []


# ── Identité ──────────────────────────────────────────────────────────────────

@router.get("/profile")
async def get_profile(current_user: User = Depends(require_chat_user)):
    """Retourne le profil de l'utilisateur connecté.

    SuperAdmin → profil complet stocké localement.
    Autres     → profil minimal dérivé du JWT (rôle uniquement).
    """
    if current_user.role == UserRole.SUPERADMIN:
        return user_profile.profile
    return {
        "role": current_user.role,
        "core_values": [],
        "goals": [],
        "bio": "",
    }


@router.post("/profile/bio")
async def update_bio(req: BioUpdateRequest, current_user: User = Depends(require_chat_user)):
    if current_user.role != UserRole.SUPERADMIN:
        raise HTTPException(status_code=403, detail="Accès réservé au SuperAdmin.")
    user_profile.update_bio(req.bio)
    return {"message": "Bio mise à jour."}


@router.post("/profile/goal")
async def add_goal(req: GoalRequest, current_user: User = Depends(require_chat_user)):
    if current_user.role != UserRole.SUPERADMIN:
        raise HTTPException(status_code=403, detail="Accès réservé au SuperAdmin.")
    user_profile.add_goal(req.description, req.priority)
    return {"message": "Objectif ajouté."}


# ── Projets ───────────────────────────────────────────────────────────────────

@router.get("/projects")
async def list_projects(current_user: User = Depends(require_chat_user)):
    """Retourne les projets actifs.

    SuperAdmin → liste complète des projets locaux.
    Autres     → liste vide (les projets locaux appartiennent au SuperAdmin).
    """
    if current_user.role == UserRole.SUPERADMIN:
        return project_manager.list_active_projects()
    return []


@router.post("/projects")
async def create_project(req: ProjectCreateRequest, current_user: User = Depends(require_chat_user)):
    if current_user.role != UserRole.SUPERADMIN:
        raise HTTPException(status_code=403, detail="Accès réservé au SuperAdmin.")
    pid = project_manager.create_project(req.name, req.description, req.tags)
    return {"message": "Projet créé.", "project_id": pid}


@router.get("/summary")
async def get_total_summary(current_user: User = Depends(require_chat_user)):
    if current_user.role == UserRole.SUPERADMIN:
        return {
            "user": user_profile.get_summary(),
            "projects": project_manager.get_projects_summary(),
        }
    return {
        "user": f"Utilisateur : {current_user.full_name or current_user.email or current_user.id}",
        "projects": "Aucun projet actif.",
    }

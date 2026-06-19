from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from core.user_profile import user_profile
from core.project_memory import project_manager

router = APIRouter()

# --- Identité ---

class BioUpdateRequest(BaseModel):
    bio: str

class GoalRequest(BaseModel):
    description: str
    priority: int = 3

@router.get("/profile")
async def get_profile():
    return user_profile.profile

@router.post("/profile/bio")
async def update_bio(req: BioUpdateRequest):
    user_profile.update_bio(req.bio)
    return {"message": "Bio mise à jour."}

@router.post("/profile/goal")
async def add_goal(req: GoalRequest):
    user_profile.add_goal(req.description, req.priority)
    return {"message": "Objectif ajouté."}

# --- Projets ---

class ProjectCreateRequest(BaseModel):
    name: str
    description: str
    tags: Optional[List[str]] = []

@router.get("/projects")
async def list_projects():
    return project_manager.list_active_projects()

@router.post("/projects")
async def create_project(req: ProjectCreateRequest):
    pid = project_manager.create_project(req.name, req.description, req.tags)
    return {"message": "Projet créé.", "project_id": pid}

@router.get("/summary")
async def get_total_summary():
    return {
        "user": user_profile.get_summary(),
        "projects": project_manager.get_projects_summary()
    }

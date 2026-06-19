import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from models.schemas import ProjectContext

PROJECTS_FILE = Path("brain_data/projects.json")

class ProjectManager:
    def __init__(self):
        self.projects = self._load_projects()

    def _load_projects(self) -> dict:
        if PROJECTS_FILE.exists():
            try:
                data = json.loads(PROJECTS_FILE.read_text(encoding="utf-8"))
                return {pid: ProjectContext(**pdata) for pid, pdata in data.items()}
            except Exception as e:
                print(f"⚠️ Erreur chargement projets: {e}")
        return {}

    def _save_projects(self):
        PROJECTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {pid: p.model_dump() for pid, p in self.projects.items()}
        PROJECTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def create_project(self, name: str, description: str, tags: List[str] = None) -> str:
        pid = str(uuid.uuid4())[:8]
        project = ProjectContext(
            id=pid,
            name=name,
            description=description,
            tags=tags or [],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        self.projects[pid] = project
        self._save_projects()
        return pid

    def get_project(self, pid: str) -> Optional[ProjectContext]:
        return self.projects.get(pid)

    def list_active_projects(self) -> List[ProjectContext]:
        return [p for p in self.projects.values() if p.status == "active"]

    def add_file_to_project(self, pid: str, file_path: str):
        if pid in self.projects:
            if file_path not in self.projects[pid].files_context:
                self.projects[pid].files_context.append(file_path)
                self.projects[pid].updated_at = datetime.now().isoformat()
                self._save_projects()

    def get_projects_summary(self) -> str:
        active = self.list_active_projects()
        if not active: return "Aucun projet actif."
        
        lines = ["Projets en cours :"]
        for p in active:
            lines.append(f"- {p.name} : {p.description} (Tags: {', '.join(p.tags)})")
        return "\n".join(lines)

# Singleton
project_manager = ProjectManager()

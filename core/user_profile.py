import json
from pathlib import Path
from datetime import datetime
from models.schemas import UserContext, UserGoal

USER_PROFILE_FILE = Path("brain_data/user_profile.json")

class UserProfileManager:
    def __init__(self):
        self.profile = self._load_profile()

    def _load_profile(self) -> UserContext:
        if USER_PROFILE_FILE.exists():
            try:
                data = json.loads(USER_PROFILE_FILE.read_text(encoding="utf-8"))
                return UserContext(**data)
            except Exception as e:
                print(f"⚠️ Erreur chargement profil utilisateur: {e}")
        
        # Profil par défaut si inexistant
        default_profile = UserContext(
            name="MAKEN",
            role="Ingénieur Full-stack",
            core_values=["Excellence", "Innovation", "Afrique"],
            preferences={"language": "fr", "theme": "ironman"}
        )
        self._save_profile(default_profile)
        return default_profile

    def _save_profile(self, profile: UserContext):
        USER_PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
        profile.updated_at = datetime.now().isoformat()
        USER_PROFILE_FILE.write_text(
            profile.model_dump_json(indent=2), encoding="utf-8"
        )

    def update_bio(self, bio: str):
        self.profile.bio = bio
        self._save_profile(self.profile)

    def add_goal(self, description: str, priority: int = 3):
        import uuid
        goal = UserGoal(id=str(uuid.uuid4())[:8], description=description, priority=priority)
        self.profile.goals.append(goal)
        self._save_profile(self.profile)

    def get_summary(self) -> str:
        """Retourne un résumé textuel pour le contexte du LLM."""
        goals_str = "\n".join([f"- {g.description} (Priorité: {g.priority})" for g in self.profile.goals if g.status == "active"])
        return (
            f"Utilisateur: {self.profile.name}\n"
            f"Rôle: {self.profile.role}\n"
            f"Valeurs: {', '.join(self.profile.core_values)}\n"
            f"Objectifs actifs:\n{goals_str}"
        )

# Singleton
user_profile = UserProfileManager()

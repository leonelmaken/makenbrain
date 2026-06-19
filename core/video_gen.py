import asyncio
import random
from pathlib import Path

VIDEO_DIR = Path("brain_data/outputs/videos")

async def generate_video(prompt: str, model: str = "wan-2.1") -> dict:
    """
    Génère une vidéo à partir d'un prompt.
    Supporte Wan 2.1 et Hunyuan Video (via API externe).
    Pour l'instant, simule la génération avec un suivi de progression.
    """
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Simulation de la latence de génération
    # En production, on appellerait ici Replicate, Fal.ai ou une instance locale
    print(f"🎬 Démarrage de la génération vidéo ({model}) : {prompt}")
    
    steps = ["Initialisation du modèle", "Traitement du prompt", "Génération des frames", "Encodage", "Finalisation"]
    for i, step in enumerate(steps):
        print(f"[{i*25}%] {step}...")
        await asyncio.sleep(2)

    # Simulation d'un fichier généré
    video_id = random.randint(1000, 9999)
    video_path = VIDEO_DIR / f"video_{video_id}.mp4"
    
    # On crée un fichier vide pour simuler
    video_path.touch()

    return {
        "status": "success",
        "video_url": f"/static/outputs/videos/{video_path.name}",
        "path": str(video_path),
        "prompt": prompt,
        "model": model
    }

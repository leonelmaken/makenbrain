from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from core.video_gen import generate_video

router = APIRouter()

class VideoRequest(BaseModel):
    prompt: str
    model: str = "wan-2.1"

@router.post("/generate")
async def create_video(req: VideoRequest):
    """Lance la génération d'une vidéo."""
    result = await generate_video(req.prompt, req.model)
    return result

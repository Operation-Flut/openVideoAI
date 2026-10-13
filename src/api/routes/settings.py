"""
Settings API endpoints for openVideoAI
Handles persisting and retrieving application configuration
"""
from fastapi import APIRouter, HTTPException
from src.schemas import SettingsModel
from src.api.state import read_saved_settings, write_saved_settings
from src.models.initialize import get_model_manager
from pathlib import Path

router = APIRouter()

@router.get("/settings")
async def get_settings():
    """Get persisted settings; clients layer defaults over this partial object."""
    return read_saved_settings()


@router.post("/settings")
async def update_settings(request: SettingsModel):
    """Update settings"""
    current = read_saved_settings()
    update_data = request.model_dump(exclude_unset=True)
    current.update(update_data)
    write_saved_settings(current)
    
    return {"message": "Settings updated", "settings": current}


@router.post("/settings/model-dir")
async def update_model_dir(request: dict):
    """Update model directory"""
    path = request.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="Path required")
    
    # Update settings
    current = read_saved_settings()
    current["modelDir"] = path
    write_saved_settings(current)
    
    # Update model manager
    manager = get_model_manager()
    manager.downloader.models_dir = Path(path)
    manager.downloader.models_dir.mkdir(parents=True, exist_ok=True)
    
    return {"message": "Model directory updated", "path": path}


@router.post("/settings/export-dir")
async def update_export_dir(request: dict):
    """Update standard export directory"""
    path = request.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="Path required")
    
    current = read_saved_settings()
    current["exportDir"] = path
    write_saved_settings(current)
    
    Path(path).mkdir(parents=True, exist_ok=True)
    return {"message": "Export directory updated", "path": path}

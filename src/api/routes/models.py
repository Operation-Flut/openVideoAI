"""
Model management API endpoints for openVideoAI
Handles model listing, downloading, progress tracking, and status
"""
from fastapi import HTTPException, BackgroundTasks, APIRouter
from fastapi.responses import FileResponse, HTMLResponse
from typing import List, Dict, Any
import asyncio
import json
from pathlib import Path

from src.models.initialize import (
    get_model_manager, initialize_models, get_model_status,
    download_model, is_model_downloaded, get_model_by_name,
    get_global_features, DownloadProgress
)
from src.schemas import (
    ModelResponse,
    DownloadRequest,
    DownloadStatus,
)
from src import system_ressources as system_resources
from src.api.state import (
    download_progress, download_tasks, SETTINGS_FILE, TEMP_PROCESSING_DIR,
)
from src.core.paths import ui_dir

router = APIRouter()


def _build_compatibility(model: Dict[str, Any]) -> Dict[str, Any]:
    """Compute platform compatibility for a model dict from the catalog."""
    compat = system_resources.check_model_compatibility(
        requires_cuda=bool(model.get('requires_cuda', model.get('requiresCuda', False))),
        requires_metal=bool(model.get('requires_metal', model.get('requiresMetal', False))),
        can_run_on_cpu=bool(model.get('can_run_on_cpu', model.get('canRunOnCpu', True))),
    )
    return {
        "compatible": compat["compatible"],
        "compatibility_reason": compat["reason"] or None,
    }


@router.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    def progress_cb(msg: str, progress: float):
        print(f"[Init] {msg} ({progress*100:.0f}%)")
    
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, initialize_models, progress_cb)


@router.get("/models", response_model=List[ModelResponse])
async def list_models():
    """Get all models with their download status"""
    status = get_model_status()
    return [
        ModelResponse(
            name=m['name'],
            description=m['description'],
            size=m['size'],
            weight=m['weight'],
            downloaded=m['downloaded'],
            features=m['features'],
            is_ai=m.get('is_ai', True),
            is_builtin=m.get('is_builtin', False),
            requires_cuda=m.get('requires_cuda', False),
            requires_metal=m.get('requires_metal', False),
            can_run_on_cpu=m.get('can_run_on_cpu', True),
            has_download_url=m.get('has_download_url', False),
            compatible=_build_compatibility(m)["compatible"],
            compatibility_reason=_build_compatibility(m)["compatibility_reason"],
        )
        for m in status['models']
    ]


@router.get("/features")
async def get_features():
    """Get all globally defined features from models.json"""
    features = get_global_features()
    return [
        {"name": f['name'], "description": f['description']}
        for f in features
    ]


@router.get("/modules")
async def get_modules():
    """Get configured processing modules from models.json"""
    catalog_path = get_model_manager().downloader.models_json
    with open(catalog_path, 'r', encoding='utf-8') as catalog_file:
        data = json.load(catalog_file)
    return data.get('modules', [])


@router.get("/features/implemented")
async def get_implemented_features():
    """Get features that have UI implementations in the frontend"""
    implemented = {
        "Upscale": {"key": "base", "required": True, "hasToggle": False, "hasStrength": False},
        "Face Recovery": {"key": "face", "required": False, "hasToggle": True, "hasStrength": False},
        "Sharpen / Deblur": {"key": "sharpen", "required": False, "hasToggle": True, "hasStrength": True},
        "Denoise": {"key": "denoise", "required": False, "hasToggle": True, "hasStrength": True},
        "Frame Interpolation": {"key": "frame", "required": False, "hasToggle": True, "hasStrength": False},
        "Match Color": {"key": "color", "required": False, "hasToggle": True, "hasStrength": True, "requiresReferenceImage": True},
        "Log Restoration": {"key": "log", "required": False, "hasToggle": True, "hasStrength": False},
    }
    return implemented


@router.get("/models/{model_name}", response_model=ModelResponse)
async def get_model(model_name: str):
    """Get details for a specific model"""
    model = get_model_by_name(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    downloaded = is_model_downloaded(model_name)
    compatibility = _build_compatibility({
        'requires_cuda': model.requires_cuda,
        'requires_metal': model.requires_metal,
    })
    return ModelResponse(
        name=model.name,
        description=model.description_short,
        size=model.size,
        weight=model.weight,
        downloaded=downloaded,
        features=[f['name'] for f in model.features],
        is_ai=model.is_ai,
        is_builtin=model.is_builtin,
        requires_cuda=model.requires_cuda,
        requires_metal=model.requires_metal,
        can_run_on_cpu=model.can_run_on_cpu,
        has_download_url=bool(model.download_url),
        compatible=compatibility["compatible"],
        compatibility_reason=compatibility["compatibility_reason"],
    )


@router.post("/models/download")
async def start_download(request: DownloadRequest, background_tasks: BackgroundTasks):
    """Start downloading a model"""
    model_name = request.model_name
    
    if is_model_downloaded(model_name):
        raise HTTPException(status_code=400, detail="Model already downloaded")
    
    if model_name in download_tasks and not download_tasks[model_name].done():
        raise HTTPException(status_code=400, detail="Download already in progress")
    
    model = get_model_by_name(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Hardware compatibility gate
    compatibility = _build_compatibility({
        'requires_cuda': model.requires_cuda,
        'requires_metal': model.requires_metal,
    })
    if not compatibility["compatible"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot download '{model_name}': {compatibility['compatibility_reason']}",
        )
    
    # Create progress tracking
    progress = DownloadProgress(
        model_name=model_name,
        total_bytes=0,
        downloaded_bytes=0,
        speed_bps=0,
        eta_seconds=0,
        status='starting'
    )
    download_progress[model_name] = progress
    
    # Start download in background
    async def download_task():
        def progress_callback(p: DownloadProgress):
            download_progress[model_name] = p
        
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None, download_model, model_name, progress_callback
        )
        
        if success:
            download_progress[model_name].status = 'completed'
        else:
            download_progress[model_name].status = 'failed'
    
    task = asyncio.create_task(download_task())
    download_tasks[model_name] = task
    
    return {"message": "Download started", "model_name": model_name}


@router.get("/models/{model_name}/progress", response_model=DownloadStatus)
async def get_download_progress(model_name: str):
    """Get download progress for a model"""
    if model_name not in download_progress:
        raise HTTPException(status_code=404, detail="No download in progress")
    
    p = download_progress[model_name]
    progress_pct = 0.0
    if p.total_bytes > 0:
        progress_pct = (p.downloaded_bytes / p.total_bytes) * 100
    
    return DownloadStatus(
        model_name=p.model_name,
        total_bytes=p.total_bytes,
        downloaded_bytes=p.downloaded_bytes,
        speed_bps=p.speed_bps,
        eta_seconds=p.eta_seconds,
        status=p.status,
        error=p.error,
        progress_percent=progress_pct
    )


@router.post("/models/{model_name}/cancel")
async def cancel_download(model_name: str):
    """Cancel an ongoing download"""
    manager = get_model_manager()
    manager.downloader.cancel_download(model_name)
    
    if model_name in download_tasks:
        download_tasks[model_name].cancel()
        del download_tasks[model_name]
    
    if model_name in download_progress:
        download_progress[model_name].status = 'cancelled'
    
    return {"message": "Download cancelled"}


@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a downloaded model"""
    manager = get_model_manager()
    if manager.downloader.delete_model(model_name):
        return {"message": "Model deleted"}
    raise HTTPException(status_code=404, detail="Model not found")


@router.get("/status")
async def get_status():
    """Get overall system status"""
    status = get_model_status()
    return {
        "total_models": status['total'],
        "downloaded": status['downloaded'],
        "available": status['available'],
        "models_dir": str(Path("models").absolute())
    }

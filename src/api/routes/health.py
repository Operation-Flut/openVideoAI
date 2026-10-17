"""
Health check and system info API routes.
"""
from fastapi import APIRouter, Depends
from typing import Dict, Any

from ...api.schemas import HealthResponse, SystemStatus
from ...api.dependencies import get_model_service, get_settings_service
from ...services.model_service import ModelService
from ...services.settings_service import SettingsService
from ...core.config import __version__
from ...utils.system_resources import get_system_info
from ...api.startup import get_startup_progress
import time

router = APIRouter(prefix="/health", tags=["health"])

_start_time = time.time()


@router.get("", response_model=HealthResponse)
async def health_check(
    model_service: ModelService = Depends(get_model_service),
    settings_service: SettingsService = Depends(get_settings_service),
):
    """Health check endpoint."""
    status = model_service.get_system_status()
    sys_info = get_system_info()

    return HealthResponse(
        status="healthy",
        version=__version__,
        uptime_seconds=time.time() - _start_time,
    )


@router.get("/ready")
async def readiness_check(
    model_service: ModelService = Depends(get_model_service),
):
    """Readiness check - returns 200 if ready to serve requests."""
    # Check if model service is initialized
    try:
        model_service.get_all_models()
        return {"status": "ready"}
    except Exception:
        return {"status": "not_ready"}, 503


@router.get("/live")
async def liveness_check():
    """Liveness check - returns 200 if process is alive."""
    return {"status": "alive"}


@router.get("/system", response_model=SystemStatus)
async def system_info(
    model_service: ModelService = Depends(get_model_service),
):
    """Get detailed system information."""
    status = model_service.get_system_status()
    sys_info = get_system_info()

    return SystemStatus(
        total_models=status["total_models"],
        downloaded=status["downloaded"],
        available=status["available"],
        models_dir=status["models_dir"],
        gpu_info=sys_info.get("gpu"),
        cpu_threads=sys_info.get("cpu_threads", 0),
        ram_gb=sys_info.get("ram_gb", 0),
        vram_gb=sys_info.get("vram_gb"),
    )


@router.get("/version")
async def version_info():
    """Get version information."""
    return {
        "version": __version__,
        "name": "openVideoAI",
        "description": "Local AI Video Enhancement Toolkit",
    }


@router.get("/startup-progress", include_in_schema=False)
async def startup_progress():
    """Get startup progress for loading screen."""
    return get_startup_progress()
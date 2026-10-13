"""
Health and system diagnostics API endpoints for openVideoAI
"""
from fastapi import APIRouter
from src import system_ressources as system_resources

router = APIRouter()

@router.get("/system/info")
async def get_system_info():
    """Get system hardware information"""
    return system_resources.get_system_hardware_info()


@router.get("/system/gpus")
async def get_gpus():
    """Get list of available GPUs"""
    return {"gpus": system_resources.get_hardware_gpus()}


@router.get("/system/capabilities")
async def get_system_capabilities():
    """Get platform capabilities for model compatibility"""
    return system_resources.get_platform_capabilities()

"""
Settings API routes.
"""
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pathlib import Path
import tempfile

from ...api.schemas import SettingsModel, SettingsResponse, ErrorResponse
from ...api.dependencies import get_settings_service
from ...services.settings_service import SettingsService
from ...utils.system_resources import get_available_compute_backends

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=SettingsResponse)
async def get_settings(
    settings_service: SettingsService = Depends(get_settings_service),
):
    """Get current application settings."""
    settings = settings_service.get_settings()
    max_settings = settings_service.get_max_settings()
    return SettingsResponse(
        settings=settings,
        max_values=max_settings,
        available_backends=get_available_compute_backends(),
    )


@router.patch("", response_model=SettingsModel)
async def update_settings(
    updates: Dict[str, Any],
    settings_service: SettingsService = Depends(get_settings_service),
):
    """Update application settings."""
    try:
        # Validate first
        errors = settings_service.validate_settings(updates)
        if errors:
            raise HTTPException(status_code=400, detail={"errors": errors})

        updated = settings_service.update_settings(updates)
        return updated
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset", response_model=SettingsModel)
async def reset_settings(
    settings_service: SettingsService = Depends(get_settings_service),
):
    """Reset settings to defaults."""
    return settings_service.reset_to_defaults()


@router.get("/{key}")
async def get_setting(
    key: str,
    settings_service: SettingsService = Depends(get_settings_service),
):
    """Get a single setting by key (dot notation)."""
    value = settings_service.get_setting(key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"Setting not found: {key}")
    return {"key": key, "value": value}


@router.put("/{key}")
async def set_setting(
    key: str,
    value: Any,
    settings_service: SettingsService = Depends(get_settings_service),
):
    """Set a single setting by key (dot notation)."""
    try:
        settings_service.set_setting(key, value)
        return {"key": key, "value": value}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/export/file")
async def export_settings_file(
    settings_service: SettingsService = Depends(get_settings_service),
):
    """Export settings to a downloadable JSON file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = Path(f.name)

    try:
        success = settings_service.export_settings(temp_path)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to export settings")

        return FileResponse(
            temp_path,
            media_type="application/json",
            filename="openvideoai_settings.json",
            # Delete the temporary file once the download has been sent.
            background=BackgroundTask(temp_path.unlink, missing_ok=True),
        )
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import/file")
async def import_settings_file(
    file: UploadFile = File(...),
    settings_service: SettingsService = Depends(get_settings_service),
):
    """Import settings from a JSON file."""
    if not (file.filename or "").lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="File must be JSON")

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as f:
        temp_path = Path(f.name)
        content = await file.read()
        f.write(content)

    try:
        success = settings_service.import_settings(temp_path)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to import settings")
        return {"message": "Settings imported successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        temp_path.unlink(missing_ok=True)
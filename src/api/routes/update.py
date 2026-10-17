"""
Update service API routes.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from ...api.schemas import (
    UpdateCheckResponse,
    PreReleaseCheckResponse,
    UpdateInstallRequest,
    UpdateInstallResponse,
)
from ...services.update import check_for_update, latest_pre_release, install_update

router = APIRouter(prefix="/update", tags=["update"])


@router.get("/check", response_model=UpdateCheckResponse)
async def check_update():
    """Check for a new stable release on GitHub."""
    result = check_for_update()
    if "error" in result:
        raise HTTPException(status_code=502, detail=result)
    return UpdateCheckResponse(**result)


@router.get("/pre-release", response_model=PreReleaseCheckResponse)
async def check_pre_release():
    """Check for a new pre-release on GitHub."""
    result = latest_pre_release()
    if "error" in result:
        raise HTTPException(status_code=502, detail=result)
    return PreReleaseCheckResponse(**result)


@router.post("/install", response_model=UpdateInstallResponse)
async def install_update_endpoint(request: UpdateInstallRequest):
    """Download the release asset for this platform.

    Returns the local path to the downloaded installer/archive. The file is
    not executed; the user must run it manually.
    """
    try:
        path = install_update(request.release_tag)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Download failed: {error}")
    return UpdateInstallResponse(asset_path=str(path), asset_name=path.name)
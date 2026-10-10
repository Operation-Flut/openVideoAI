"""
Model and download schemas for openVideoAI
"""
from typing import List, Optional
from pydantic import BaseModel


class ModelResponse(BaseModel):
    name: str
    description: str
    size: str
    weight: str
    downloaded: bool
    features: List[str]
    is_ai: bool = True
    is_builtin: bool = False
    requires_cuda: bool = False
    requires_metal: bool = False
    can_run_on_cpu: bool = True
    has_download_url: bool = False
    compatible: bool = True
    compatibility_reason: Optional[str] = None


class DownloadRequest(BaseModel):
    model_name: str


class DownloadStatus(BaseModel):
    model_name: str
    total_bytes: int
    downloaded_bytes: int
    speed_bps: float
    eta_seconds: float
    status: str
    error: Optional[str] = None
    progress_percent: float = 0.0

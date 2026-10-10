"""
Schemas package for openVideoAI
"""
from .settings import SettingsModel
from .models import ModelResponse, DownloadRequest, DownloadStatus
from .processing import UpscaleRequest, ProcessResponse

__all__ = [
    'SettingsModel',
    'ModelResponse',
    'DownloadRequest',
    'DownloadStatus',
    'UpscaleRequest',
    'ProcessResponse',
]

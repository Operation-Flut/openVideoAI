"""
openVideoAI - Main Package
"""
from dist.openVideoAI._internal.torch._vendor.packaging.version import Version

from .models import (
    get_model_manager,
    initialize_models,
    get_model_status,
    get_downloaded_models,
    get_available_models,
    download_model,
    ensure_model,
    is_model_downloaded,
    get_model_by_name,
    get_model_files,
    get_all_models,
    ModelInfo,
    DownloadProgress
)

from core.config import __version__ as version

__version__ = version
__all__ = [
    'get_model_manager',
    'initialize_models',
    'get_model_status',
    'get_downloaded_models',
    'get_available_models',
    'download_model',
    'ensure_model',
    'is_model_downloaded',
    'get_model_by_name',
    'get_model_files',
    'get_all_models',
    'ModelInfo',
    'DownloadProgress'
]
"""
openVideoAI - Main Package
"""
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

__version__ = "0.2.0-alpha"
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
"""
Model initialization and discovery for openVideoAI
"""
import os
from pathlib import Path
from typing import List, Dict, Optional, Callable
from .downloader import ModelManager, ModelInfo, DownloadProgress


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get or create the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def initialize_models(progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict:
    """Initialize model system - create directories, verify existing models"""
    manager = get_model_manager()
    return manager.initialize(progress_callback)


def get_model_status() -> Dict:
    """Get current status of all models"""
    manager = get_model_manager()
    return manager.get_model_status()


def get_downloaded_models() -> List[ModelInfo]:
    """Get list of downloaded model objects"""
    manager = get_model_manager()
    return manager.downloader.get_downloaded_models()


def get_available_models() -> List[ModelInfo]:
    """Get list of models available for download"""
    manager = get_model_manager()
    return manager.downloader.get_available_models()


def download_model(model_name: str, progress_callback: Optional[Callable[[DownloadProgress], None]] = None) -> bool:
    """Download a specific model by name"""
    manager = get_model_manager()
    return manager.download_model(model_name, progress_callback)


def ensure_model(model_name: str, progress_callback: Optional[Callable[[DownloadProgress], None]] = None) -> bool:
    """Ensure a model is available locally, downloading if necessary"""
    manager = get_model_manager()
    return manager.ensure_model(model_name, progress_callback)


def is_model_downloaded(model_name: str) -> bool:
    """Check if a model is already downloaded"""
    manager = get_model_manager()
    return manager.downloader.is_model_downloaded(model_name)


def get_model_by_name(name: str) -> Optional[ModelInfo]:
    """Get model info by name"""
    manager = get_model_manager()
    return manager.downloader.get_model_by_name(name)


def get_model_files() -> List[str]:
    """Legacy function - returns list of model file paths"""
    models_dir = Path("models")
    if not models_dir.exists():
        return ["models"]
    
    files = ["models"]
    for filepath in models_dir.rglob("*"):
        if filepath.is_file():
            files.append(str(filepath))
    return files


def get_all_models() -> List[ModelInfo]:
    """Get all models from catalog"""
    manager = get_model_manager()
    return manager.downloader.load_models_catalog()


def get_global_features() -> List[Dict]:
    """Get all globally defined features from models.json"""
    manager = get_model_manager()
    # Access the raw models.json data through the downloader
    import json
    with open(manager.downloader.models_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('features', [])
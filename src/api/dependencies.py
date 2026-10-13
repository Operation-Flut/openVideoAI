"""
FastAPI dependency injection for openVideoAI
Provides shared application services and utilities
"""
from fastapi import Depends
from typing import AsyncGenerator
import asyncio

# Application services - will be initialized by startup event
_services = {
    'model_manager': None,
    'processing_available': False,
    '_upscale_image': None,
    '_upscale_video': None,
    '_get_upscale_model': None,
    '_list_available_upscale_models': None,
    '_UpscalePipeline': None,
}


def get_model_manager():
    """Get the initialized model manager"""
    if _services['model_manager'] is None:
        from src.models.initialize import get_model_manager as _get_mm
        _services['model_manager'] = _get_mm()
    return _services['model_manager']


def load_processing():
    """Lazy load processing modules after distutils monkey patch"""
    if not _services['processing_available']:
        try:
            from src.processing.upscale import (
                upscale_image, upscale_video, get_upscale_model,
                list_available_upscale_models, UpscalePipeline
            )
            _services['_upscale_image'] = upscale_image
            _services['_upscale_video'] = upscale_video
            _services['_get_upscale_model'] = get_upscale_model
            _services['_list_available_upscale_models'] = list_available_upscale_models
            _services['_UpscalePipeline'] = UpscalePipeline
            _services['processing_available'] = True
        except ImportError as e:
            print(f"Processing modules not available: {e}")
            _services['processing_available'] = False
    
    return _services


def get_upscale_image():
    """Get the upscale_image function"""
    load_processing()
    if not _services['processing_available']:
        raise RuntimeError("Processing modules not available")
    if _services['_upscale_image'] is None:
        raise RuntimeError("Upscale image function not loaded")
    return _services['_upscale_image']


def get_upscale_video():
    """Get the upscale_video function"""
    load_processing()
    if not _services['processing_available']:
        raise RuntimeError("Processing modules not available")
    if _services['_upscale_video'] is None:
        raise RuntimeError("Upscale video function not loaded")
    return _services['_upscale_video']


def get_upscale_model():
    """Get the upscale model lookup function"""
    load_processing()
    if not _services['processing_available']:
        raise RuntimeError("Processing modules not available")
    if _services['_get_upscale_model'] is None:
        raise RuntimeError("Upscale model lookup function not loaded")
    return _services['_get_upscale_model']


def get_list_available_upscale_models():
    """Get the list available upscale models function"""
    load_processing()
    if not _services['processing_available']:
        raise RuntimeError("Processing modules not available")
    if _services['_list_available_upscale_models'] is None:
        raise RuntimeError("List available upscale models function not loaded")
    return _services['_list_available_upscale_models']


def get_upscale_pipeline_class():
    """Get the UpscalePipeline class"""
    load_processing()
    if not _services['processing_available']:
        raise RuntimeError("Processing modules not available")
    if _services['_UpscalePipeline'] is None:
        raise RuntimeError("Upscale pipeline class not loaded")
    return _services['_UpscalePipeline']


async def get_event_loop():
    """Get the current event loop"""
    return asyncio.get_event_loop()


# Global state for shutdown tracking
_shutdown_requested = False


def request_processing_shutdown() -> None:
    """Signal running processing tasks to abort (called on window close)."""
    global _shutdown_requested
    _shutdown_requested = True


def has_active_processing() -> bool:
    """True when at least one processing task is still running."""
    # This will be implemented with proper state management
    return False


# Re-export for convenience
__all__ = [
    'get_model_manager',
    'load_processing',
    'get_upscale_image',
    'get_upscale_video',
    'get_upscale_model',
    'get_list_available_upscale_models',
    'get_upscale_pipeline_class',
    'get_event_loop',
    'request_processing_shutdown',
    'has_active_processing',
]
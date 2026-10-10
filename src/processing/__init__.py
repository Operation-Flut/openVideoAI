"""
openVideoAI Processing Package
AI inference pipeline for image/video enhancement

The package itself is intentionally dependency-light: the heavy submodules
(cv2/numpy-based upscaling) are imported lazily on attribute access so that
importing the package just for exception types (e.g. ProcessingAborted) never
pulls in the full processing stack.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .upscale import (  # noqa: F401
        upscale_image,
        upscale_video,
        get_upscale_model,
        list_available_upscale_models,
    )
    from .base import EnhancementPipeline, ProcessingError  # noqa: F401

_PUBLIC_ATTRS = (
    'upscale_image',
    'upscale_video',
    'get_upscale_model',
    'list_available_upscale_models',
    'EnhancementPipeline',
    'ProcessingError',
    'ProcessingAborted',
)


def __getattr__(name):
    if name == 'upscale_image':
        from .upscale import upscale_image
        return upscale_image
    if name == 'upscale_video':
        from .upscale import upscale_video
        return upscale_video
    if name == 'get_upscale_model':
        from .upscale import get_upscale_model
        return get_upscale_model
    if name == 'list_available_upscale_models':
        from .upscale import list_available_upscale_models
        return list_available_upscale_models
    if name == 'EnhancementPipeline':
        from .base import EnhancementPipeline
        return EnhancementPipeline
    if name == 'ProcessingError':
        from .base import ProcessingError
        return ProcessingError
    if name == 'ProcessingAborted':
        from .base import ProcessingAborted
        return ProcessingAborted
    raise AttributeError(f"module 'src.processing' has no attribute '{name}'")


def __dir__():
    return sorted(_PUBLIC_ATTRS)
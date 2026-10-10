"""
Upscaling Feature Module for openVideoAI
Provides upscale pipelines and model configurations.
"""
from ...processing.upscale import (
    UpscalePipeline,
    upscale_image,
    upscale_video,
    get_upscale_model,
    list_available_upscale_models,
    UPSCALE_MODELS
)

__all__ = [
    'UpscalePipeline',
    'upscale_image',
    'upscale_video',
    'get_upscale_model',
    'list_available_upscale_models',
    'UPSCALE_MODELS'
]

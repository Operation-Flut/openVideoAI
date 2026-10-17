"""
Processing package - Unified enhancement pipelines.

This package provides all video/image enhancement pipelines organized by feature.
Each pipeline is self-contained and can be chained together.
"""
from .pipeline.base import (
    EnhancementPipeline,
    PipelineChain,
    PipelineConfig,
    ProcessingContext,
    ProcessingResult,
    PipelineError,
    ModelLoadError,
    InferenceError,
    UnsupportedFormatError,
    ProcessingAborted,
)
from .pipeline.registry import PipelineRegistry, get_pipeline_registry, register_pipeline

# Import all pipelines to trigger registration
from .upscale import pipeline as upscale_pipeline
from .face import pipeline as face_pipeline
from .denoise import pipeline as denoise_pipeline
from .sharpen import pipeline as sharpen_pipeline
from .interpolation import pipeline as interpolation_pipeline
from .color import pipeline as color_pipeline
from .video import pipeline as video_pipeline
from .low_light import pipeline as low_light_pipeline
from .sdr_to_hdr import pipeline as sdr_to_hdr_pipeline
from .reencoding import pipeline as reencoding_pipeline

__all__ = [
    # Base classes
    "EnhancementPipeline",
    "PipelineChain",
    "PipelineConfig",
    "ProcessingContext",
    "ProcessingResult",
    "PipelineError",
    "ModelLoadError",
    "InferenceError",
    "UnsupportedFormatError",
    "ProcessingAborted",
    # Registry
    "PipelineRegistry",
    "get_pipeline_registry",
    "register_pipeline",
]
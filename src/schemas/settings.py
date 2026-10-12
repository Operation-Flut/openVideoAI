"""
Settings schemas for openVideoAI
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field


class SettingsModel(BaseModel):
    # System Resources (maxRamUsage / maxVramUsage as percentages, cpuThreads as thread count)
    maxRamUsage: Optional[int] = Field(None, ge=1, le=100)
    maxVramUsage: Optional[int] = Field(None, ge=0, le=100)
    cpuThreads: Optional[int] = Field(None, ge=1, le=128)
    tileSize: Optional[int] = Field(None, ge=128, le=2048)

    # GPU Acceleration
    computeBackend: Optional[Literal["auto", "cuda", "amd", "hip", "dml", "metal", "cpu"]] = None
    gpuDevice: Optional[str] = None
    fp16Enabled: Optional[bool] = None
    tensorrtEnabled: Optional[bool] = None

    # Standard Export & Processing Defaults
    defaultUpscale: Optional[Literal[2, 4, 8]] = None
    defaultTileSize: Optional[Literal[256, 400, 512, 768]] = None
    outputFormat: Optional[Literal["source", "jpg", "png", "webp", "tiff", "mp4"]] = None
    jpgQuality: Optional[int] = Field(None, ge=1, le=100)
    # Tile overlap (px) used by the overlap-add stitcher for
    # sharpen/deblur on large images. Larger overlap blends tile seams more
    # deeply at the cost of more model calls.
    sharpenOverlap: Optional[int] = Field(None, ge=32, le=256)
    # Subtle unsharp-mask finishing applied after deblurring. 0 disables it;
    # NAFNet leaves defocus/compression-blurred content rather soft.
    sharpenUnsharpAmount: Optional[float] = Field(None, ge=0, le=2)
    exportDir: Optional[str] = None
    exportNaming: Optional[str] = None
    overwriteExisting: Optional[bool] = None
    videoCodec: Optional[Literal["libx264", "libx265", "prores", "auto"]] = None
    videoCrf: Optional[int] = Field(None, ge=0, le=51)

    # Model Management
    modelDir: Optional[str] = None
    autoDownloadModels: Optional[bool] = None
    verifyModels: Optional[bool] = None
    # Only show models compatible with this system in the Model Manager
    hideIncompatibleModels: Optional[bool] = None
    # Favor AI models over built-in math algorithms in recommendations
    preferAi: Optional[bool] = None
    # Show every individual model in the Model Manager instead of the curated view
    advancedMode: Optional[bool] = None

    # Appearance
    theme: Optional[Literal["dark", "light", "system"]] = None
    accentColor: Optional[str] = None
    fontSize: Optional[Literal["small", "medium", "large"]] = None
    density: Optional[Literal["compact", "comfortable", "spacious"]] = None

    # Advanced
    debugLogging: Optional[bool] = None
    showMetrics: Optional[bool] = None
    experimentalFeatures: Optional[bool] = None

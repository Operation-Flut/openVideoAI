"""
API Schemas - Pydantic models for request/response validation.
"""
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from uuid import UUID


# Enums
class ComputeBackend(str, Enum):
    AUTO = "auto"
    CUDA = "cuda"
    AMD = "amd"
    HIP = "hip"
    DML = "dml"
    METAL = "metal"
    CPU = "cpu"


class OutputFormat(str, Enum):
    SOURCE = "source"
    JPG = "jpg"
    PNG = "png"
    WEBP = "webp"
    TIFF = "tiff"
    MP4 = "mp4"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class ModelStatus(str, Enum):
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    FAILED = "failed"


# Base Models
class BaseResponse(BaseModel):
    success: bool = True
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None


# Model Schemas
class ModelFeature(BaseModel):
    name: str
    description: str


class ModelInfo(BaseModel):
    name: str
    description: str
    description_short: str
    size: str
    weight: str
    url: str
    download_url: Optional[str] = None
    filename: Optional[str] = None
    features: List[ModelFeature] = []
    is_ai: bool = True
    requires_cuda: bool = False
    requires_metal: bool = False
    can_run_on_cpu: bool = True
    is_builtin: bool = False
    module: Optional[str] = None


class ModelResponse(BaseModel):
    name: str
    description: str
    size: str
    weight: str
    downloaded: bool
    # False for catalog entries that declare requirements the build cannot meet
    # (``arch: null`` metadata-only entries or missing optional modules). Such
    # models stay visible and downloadable but are never offered as pipeline
    # steps.
    implemented: bool = True
    features: List[str]
    is_ai: bool = True
    is_builtin: bool = False
    requires_cuda: bool = False
    requires_metal: bool = False
    can_run_on_cpu: bool = True
    has_download_url: bool = False
    compatible: bool = True
    compatibility_reason: Optional[str] = None
    local_path: Optional[str] = None

    # What the model applies to ("image" / "video" / "both") plus the label to
    # show for it. The label is produced here so the UI never phrases the
    # catalog's media vocabulary itself.
    media: str = "both"
    media_label: str = "Images and video"

    # Extended detail fields (surfaced by the model details popup).
    full_description: Optional[str] = None
    # Catalog `bestFor` note: when to reach for this model, and which sibling
    # model to prefer instead.
    best_for: Optional[str] = None
    filename: Optional[str] = None
    module: Optional[str] = None
    source_url: Optional[str] = None
    download_url: Optional[str] = None
    verified: bool = False
    error: Optional[str] = None
    additional_paths: Dict[str, str] = Field(default_factory=dict)


class ExcludedModel(BaseModel):
    """A candidate Smart Select ruled out, with the reason why."""
    model: str
    reason: str


class ModelRecommendation(BaseModel):
    """The model Smart Select would use for one feature, and why.

    Produced by ``src/services/model_recommender.py``; the wording of ``reason``
    and ``hint`` is part of the payload so the UI never has to phrase it.
    """
    feature: str
    model: str
    reason: str
    media: str = "both"
    # Best match by fit alone, which may not be installed yet (see `hint`).
    ideal_model: Optional[str] = None
    ideal_installed: bool = False
    needs_download: bool = False
    fallback: bool = False
    hint: Optional[str] = None
    candidates: List[str] = Field(default_factory=list)
    excluded: List[ExcludedModel] = Field(default_factory=list)


class ModelRecommendationsResponse(BaseModel):
    media: str = "both"
    smart_select: bool = False
    recommendations: Dict[str, ModelRecommendation] = Field(default_factory=dict)


class DownloadRequest(BaseModel):
    model_name: str
    force: bool = False


class DownloadProgress(BaseModel):
    model_name: str
    total_bytes: int = 0
    downloaded_bytes: int = 0
    speed_bps: float = 0.0
    eta_seconds: float = 0.0
    status: str = "starting"
    error: Optional[str] = None
    progress_percent: float = 0.0


class DownloadStatus(BaseModel):
    model_name: str
    total_bytes: int
    downloaded_bytes: int
    speed_bps: float
    eta_seconds: float
    status: str
    error: Optional[str] = None
    progress_percent: float


# Processing Schemas
class ProcessingParameters(BaseModel):
    """Base processing parameters."""
    model: str
    device: Optional[str] = "auto"
    task_id: Optional[str] = None


class UpscaleParameters(ProcessingParameters):
    scale: int = Field(default=4, ge=1, le=8)
    tile_size: int = Field(default=400, ge=64, le=1024)
    fp16: bool = True
    jpg_quality: int = Field(default=95, ge=1, le=100)


class FaceRecoveryParameters(ProcessingParameters):
    fidelity_weight: float = Field(default=0.5, ge=0, le=1)
    face_upscale: bool = True


class SharpenParameters(ProcessingParameters):
    strength: float = Field(default=1.0, ge=0, le=2)
    overlap: int = Field(default=128, ge=0, le=256)
    unsharp_amount: float = Field(default=0.5, ge=0, le=2)


class DenoiseParameters(ProcessingParameters):
    strength: float = Field(default=0.5, ge=0, le=1)
    tile_size: int = Field(default=256, ge=64, le=1024)


class FrameInterpolationParameters(ProcessingParameters):
    multiplier: int = Field(default=2, ge=2, le=8)
    slowmo_factor: float = Field(default=1.0, ge=0.1, le=1.0)
    scene_threshold: float = Field(default=0.3, ge=0, le=1)
    use_scene_detection: bool = True


class MatchColorParameters(ProcessingParameters):
    strength: float = Field(default=1.0, ge=0, le=1)
    preserve_luminance: bool = True
    reference_path: str


class LowLightParameters(ProcessingParameters):
    strength: float = Field(default=0.7, ge=0, le=1)


class SDRToHDRParameters(ProcessingParameters):
    target_format: str = Field(default="HDR10", pattern="^(HDR10|HDR10\\+|HLG)$")
    strength: float = Field(default=1.0, ge=0, le=1)


class ReencodingParameters(ProcessingParameters):
    engine: str = Field(default="ffmpeg", pattern="^(ffmpeg|handbrake)$")
    video_codec: str = Field(default="libx264")
    preset: str = Field(default="medium")
    crf: int = Field(default=23, ge=0, le=51)
    audio_codec: str = Field(default="aac")
    audio_bitrate: str = Field(default="128k")
    scale: str = ""
    fps: float = 0


class PipelineStep(BaseModel):
    feature: str
    model: str
    parameters: Dict[str, Any] = {}


class ProcessRequest(BaseModel):
    """Unified processing request. Use 'steps' format (preferred). Legacy single-feature fields are deprecated."""
    input_path: str
    output_path: Optional[str] = None
    steps: List[PipelineStep] = []
    # Legacy single-feature parameters (deprecated)
    upscale: Optional[UpscaleParameters] = None
    face_recovery: Optional[FaceRecoveryParameters] = None
    sharpen: Optional[SharpenParameters] = None
    denoise: Optional[DenoiseParameters] = None
    frame_interpolation: Optional[FrameInterpolationParameters] = None
    match_color: Optional[MatchColorParameters] = None
    low_light: Optional[LowLightParameters] = None
    sdr_to_hdr: Optional[SDRToHDRParameters] = None
    reencoding: Optional[ReencodingParameters] = None
    # Global settings
    output_format: OutputFormat = OutputFormat.SOURCE
    quality: str = "medium"  # high, medium, low
    # Preview runs process a short sample of the input (see
    # ``ProcessingService._prepare_preview_input``) so a pipeline can be judged in
    # seconds instead of after a full export.
    preview: bool = False
    preview_duration: float = 2.0


class ProcessingTask(BaseModel):
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    current_step: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    output_path: Optional[str] = None


class ProcessResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: float
    message: str
    current_step: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    output_path: Optional[str] = None


# Settings Schemas
class SystemResources(BaseModel):
    max_ram_usage: int = Field(default=85, ge=10, le=100)
    max_vram_usage: int = Field(default=95, ge=10, le=100)
    cpu_threads: int = Field(default=8, ge=1, le=128)
    tile_size: int = Field(default=400, ge=64, le=1024)


class GPUSettings(BaseModel):
    compute_backend: ComputeBackend = ComputeBackend.AUTO
    gpu_device: str = "auto"
    fp16_enabled: bool = True
    tensorrt_enabled: bool = False


class ExportSettings(BaseModel):
    default_upscale: int = Field(default=4, ge=1, le=8)
    default_tile_size: int = Field(default=400, ge=64, le=1024)
    output_format: OutputFormat = OutputFormat.JPG
    jpg_quality: int = Field(default=95, ge=1, le=100)
    sharpen_overlap: int = Field(default=128, ge=0, le=256)
    sharpen_unsharp_amount: float = Field(default=0.5, ge=0, le=2)
    export_dir: str = "./exports"
    export_naming: str = "{name}_enhanced_{scale}x"
    overwrite_existing: bool = False
    video_codec: str = "libx264"
    video_crf: int = Field(default=18, ge=0, le=51)


class ModelSettings(BaseModel):
    model_dir: str = "./models"
    auto_download_models: bool = True
    verify_models: bool = True
    hide_incompatible_models: bool = False
    prefer_ai: bool = True
    advanced_mode: bool = False
    # Pick the model that suits the loaded file when a filter is enabled.
    smart_select: bool = False


class AppearanceSettings(BaseModel):
    theme: str = Field(default="dark", pattern="^(light|dark|system)$")
    accent_color: str = "#3b82f6"
    font_size: str = Field(default="medium", pattern="^(small|medium|large)$")
    density: str = Field(default="comfortable", pattern="^(compact|comfortable|spacious)$")


class AdvancedSettings(BaseModel):
    debug_logging: bool = False
    show_metrics: bool = True
    experimental_features: bool = False


class SettingsModel(BaseModel):
    system: SystemResources = Field(default_factory=SystemResources)
    gpu: GPUSettings = Field(default_factory=GPUSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    appearance: AppearanceSettings = Field(default_factory=AppearanceSettings)
    advanced: AdvancedSettings = Field(default_factory=AdvancedSettings)


class ComputeBackendInfo(BaseModel):
    """A compute backend the Settings UI can offer for this machine."""
    value: ComputeBackend
    label: str
    available: bool = True
    recommended: bool = False
    reason: Optional[str] = None


class SettingsResponse(BaseModel):
    settings: SettingsModel
    max_values: Dict[str, Any] = {}
    # Machine-derived: which compute backends this install can actually use.
    available_backends: List[ComputeBackendInfo] = []


# Feature Schemas
class FeatureInfo(BaseModel):
    name: str
    description: str
    implemented: bool = True
    available_models: List[str] = []
    default_model: Optional[str] = None
    parameters_schema: Dict[str, Any] = {}


class ProcessingFeaturesResponse(BaseModel):
    features: Dict[str, FeatureInfo]


# System Status
class SystemStatus(BaseModel):
    total_models: int
    downloaded: int
    available: int
    models_dir: str
    gpu_info: Optional[Dict[str, Any]] = None
    cpu_threads: int
    ram_gb: float
    vram_gb: Optional[float] = None


# Health Check
class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)
    uptime_seconds: float
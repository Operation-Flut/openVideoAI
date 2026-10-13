"""
Processing schemas for openVideoAI
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class UpscaleRequest(BaseModel):
    model_name: str = "Real-ESRGAN"
    scale: int = 4
    device: str = "auto"
    output_path: Optional[str] = None


class FaceRecoveryRequest(BaseModel):
    model_name: str = "CodeFormer"
    fidelity_weight: float = 0.5
    device: str = "auto"


class SharpenRequest(BaseModel):
    model_name: str = "NAFNet"
    strength: float = 1.0
    device: str = "auto"
    overlap: Optional[int] = Field(None, ge=32, le=256)
    unsharp_amount: Optional[float] = Field(None, ge=0, le=2)


class DenoiseRequest(BaseModel):
    model_name: str = "SCUNet"
    strength: float = 0.5
    device: str = "auto"


class FrameInterpolationRequest(BaseModel):
    model_name: str = "RIFE"
    multiplier: int = 2
    device: str = "auto"


class SlowMotionRequest(BaseModel):
    model_name: str = "RIFE-SloMo"
    slowdown_factor: float = 4.0
    device: str = "auto"


class MatchColorRequest(BaseModel):
    model_name: str = "Reinhard Transfer"
    reference_image_path: Optional[str] = None
    strength: float = 1.0
    preserve_luminance: bool = True


class LogRestorationRequest(BaseModel):
    model_name: str = "ITMLUT"
    color_space_target: str = "Rec.709"
    denoise_strength: float = 0.6
    shadow_recovery: float = 0.8


class LowLightRequest(BaseModel):
    model_name: str = "Zero-DCE"
    strength: float = 0.7
    device: str = "auto"


class DistortionRequest(BaseModel):
    model_name: str = "LensCorrection"


class SDRtoHDRRequest(BaseModel):
    model_name: str = "HDRNet"
    target_format: str = "HDR10"


class ObjectRemovalRequest(BaseModel):
    model_name: str = "LaMa"
    mask_path: Optional[str] = None


class FocusFixRequest(BaseModel):
    model_name: str = "DeblurGAN"


class OverexposureFixRequest(BaseModel):
    model_name: str = "AutoEnhance"


class FlatProfileRequest(BaseModel):
    model_name: str = "ColorGrade"


class ReencodeRequest(BaseModel):
    engine: str = "ffmpeg"
    preset: str = "h264_fast"
    custom_params: Optional[Dict[str, Any]] = None


class MultiprocessRequest(BaseModel):
    """Pipeline request for multiple enhancement steps"""
    features: List[str] = ["upscale"]
    upscale_model: str = "Real-ESRGAN"
    upscale_scale: int = 4
    face_recovery_enabled: bool = False
    face_recovery_model: str = "CodeFormer"
    face_fidelity: float = 0.5
    sharpen_enabled: bool = False
    sharpen_model: str = "NAFNet"
    sharpen_strength: float = 0.5
    denoise_enabled: bool = False
    denoise_model: str = "SCUNet"
    denoise_strength: float = 0.5
    frame_interp_enabled: bool = False
    frame_interp_model: str = "RIFE"
    frame_interp_multiplier: int = 2
    device: str = "auto"


class PipelineRequest(BaseModel):
    """Request schema for multi-step pipeline processing"""
    # Required: List of features to execute in order
    features: List[str] = Field(
        default=["upscale"],
        description="Ordered list of processing features to apply (e.g., ['denoise', 'upscale', 'face-recovery'])"
    )
    
    # Feature-specific configurations
    upscale_model: str = Field("Real-ESRGAN", description="Upscale model (e.g., Real-ESRGAN, RealBasicVSR)")
    upscale_scale: int = Field(4, description="Upscaling factor (2, 4)")
    
    # Face recovery
    face_recovery_model: str = Field("CodeFormer", description="Face recovery model (CodeFormer, GFPGAN)")
    face_fidelity_weight: float = Field(0.5, ge=0.1, le=1.0, description="Face restoration fidelity weight")
    
    # Sharpening/deblurring
    sharpen_model: str = Field("NAFNet", description="Sharpening model (NAFNet, MPRNet)")
    sharpen_strength: float = Field(1.0, ge=0.1, le=5.0, description="Sharpening strength")
    sharpen_overlap: int = Field(128, ge=32, le=256, description="Processing window overlap")
    sharpen_unsharp_amount: float = Field(0.5, ge=0, le=2.0, description="Unsharp mask amount")
    
    # Denoising
    denoise_model: str = Field("SCUNet", description="Denoising model (SCUNet, NAFNet)")
    denoise_strength: float = Field(0.5, ge=0.1, le=1.0, description="Denoising strength")
    
    # Frame interpolation
    frame_interp_model: str = Field("RIFE", description="Frame interpolation model")
    frame_interp_multiplier: int = Field(2, ge=1, description="Frame multiplier (2 = 60fps→120fps)")
    
    # Slow motion
    slow_motion_model: str = Field("RIFE-SloMo", description="Slow motion model")
    slow_motion_slowdown: float = Field(4.0, ge=1.0, description="Slowdown factor (e.g., 4.0 = 30fps→120fps)")
    
    # Color matching
    match_color_model: str = Field("Reinhard Transfer", description="Color matching model")
    match_color_strength: float = Field(1.0, ge=0.1, le=3.0, description="Color transfer strength")
    match_color_preserve_luminance: bool = Field(True, description="Preserve luminance during color transfer")
    
    # Low light enhancement
    low_light_model: str = Field("Zero-DCE", description="Low light model")
    low_light_strength: float = Field(0.7, ge=0.1, le=1.0, description="Low light enhancement strength")
    
    # Global settings
    device: str = Field("auto", description="Processing device (auto, cuda, cpu, metal, hip)")
    output_path: Optional[str] = Field(None, description="Custom output file path")
    
    # Advanced options
    enable_progress_tracking: bool = Field(True, description="Enable real-time progress reporting")
    max_retries: int = Field(3, ge=0, description="Maximum retry attempts for failed features")
    skip_failed_features: bool = Field(True, description="Continue with remaining features if one fails")


class ProcessResponse(BaseModel):
    success: bool
    input_path: str
    output_path: str
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    scale_factor: Optional[int] = None
    model: Optional[str] = None
    frames_processed: Optional[int] = None
    frames_failed: Optional[int] = None
    total_frames: Optional[int] = None
    input_resolution: Optional[str] = None
    output_resolution: Optional[str] = None
    fps: Optional[float] = None
    error: Optional[str] = None
    feature: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

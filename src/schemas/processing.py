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

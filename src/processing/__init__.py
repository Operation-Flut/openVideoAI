"""
Processing package for openVideoAI
Comprehensive video/image enhancement pipelines
"""
from .base import (
    EnhancementPipeline,
    ModelLoadError,
    InferenceError,
    UnsupportedFormatError,
    ProcessingError,
    ProcessingAborted,
)

# Upscale
from .upscale import (
    UpscalePipeline,
    upscale_image,
    upscale_video,
    get_upscale_model,
    list_available_upscale_models,
    UPSCALE_MODELS,
)

# Face Recovery
from .face_recovery import (
    FaceRecoveryPipeline,
    recover_face,
    get_face_model,
    list_face_models,
    FACE_MODELS,
)

# Sharpen / Deblur
from .sharpening import (
    SharpenPipeline,
    sharpen_image,
    get_sharpen_model,
    list_sharpen_models,
    SHARPEN_MODELS,
)

# Denoise
from .denoising import (
    DenoisePipeline,
    denoise_image,
    get_denoise_model,
    list_denoise_models,
    DENOISE_MODELS,
)

# Frame Interpolation
from .frame_interpolation import (
    FrameInterpolationPipeline,
    interpolate_video,
    get_frame_interp_model,
    list_frame_interp_models,
    FRAME_INTERP_MODELS,
)

# Slow Motion
from .slow_motion import (
    SlowMotionPipeline,
    create_slowmo,
    SLOWMO_MODELS,
)

# Restoration (Frame Recovery, Multipurpose)
from .restoration import (
    FrameRecoveryPipeline,
    MultipurposeRestorationPipeline,
    recover_frames,
    multipurpose_restore,
    get_multipurpose_model,
    list_multipurpose_models,
    FRAME_RECOVERY_MODELS,
    MULTIPURPOSE_MODELS,
    DIFFUSION_RESTORERS,
)

# Color Transfer (Colorization, Match Color)
from .color_transfer import (
    ColorizationPipeline,
    MatchColorPipeline,
    colorize_image,
    match_color,
    get_match_color_model,
    MATCH_COLOR_MODELS,
)

# Low Light, Distortion, LOG Restoration
from .low_light import (
    LowLightEnhancementPipeline,
    DistortionCorrectionPipeline,
    LogRestorationPipeline,
    enhance_low_light,
    correct_distortion,
    restore_log,
)

# Video Enhancements (SDR->HDR, Object Removal, Focus Fix, Overexposure, Flat Profile)
from .video_enhancements import (
    SDRtoHDRPipeline,
    ObjectRemovalPipeline,
    FocusFixPipeline,
    OverexposureFixPipeline,
    FlatColorProfilePipeline,
    convert_sdr_to_hdr,
    remove_objects,
    fix_focus,
    fix_overexposure,
    apply_flat_profile,
)

# Reencoding
from .reencoding import (
    ReencodingPipeline,
    reencode_video,
    list_encoding_presets,
    get_video_metadata,
    REENCODING_PRESETS,
    HANDBRAKE_PRESETS,
)

__all__ = [
    # Base
    'EnhancementPipeline',
    'ModelLoadError',
    'InferenceError',
    'UnsupportedFormatError',
    'ProcessingError',
    'ProcessingAborted',
    
    # Upscale
    'UpscalePipeline',
    'upscale_image',
    'upscale_video',
    'get_upscale_model',
    'list_available_upscale_models',
    'UPSCALE_MODELS',
    
    # Face Recovery
    'FaceRecoveryPipeline',
    'recover_face',
    'get_face_model',
    'list_face_models',
    'FACE_MODELS',
    
    # Sharpen
    'SharpenPipeline',
    'sharpen_image',
    'get_sharpen_model',
    'list_sharpen_models',
    'SHARPEN_MODELS',
    
    # Denoise
    'DenoisePipeline',
    'denoise_image',
    'get_denoise_model',
    'list_denoise_models',
    'DENOISE_MODELS',
    
    # Frame Interpolation
    'FrameInterpolationPipeline',
    'interpolate_video',
    'get_frame_interp_model',
    'list_frame_interp_models',
    'FRAME_INTERP_MODELS',
    
    # Slow Motion
    'SlowMotionPipeline',
    'create_slowmo',
    'SLOWMO_MODELS',
    
    # Restoration
    'FrameRecoveryPipeline',
    'MultipurposeRestorationPipeline',
    'recover_frames',
    'multipurpose_restore',
    'get_multipurpose_model',
    'list_multipurpose_models',
    'FRAME_RECOVERY_MODELS',
    'MULTIPURPOSE_MODELS',
    'DIFFUSION_RESTORERS',
    
    # Color Transfer
    'ColorizationPipeline',
    'MatchColorPipeline',
    'colorize_image',
    'match_color',
    'get_match_color_model',
    'MATCH_COLOR_MODELS',
    
    # Low Light & Distortion
    'LowLightEnhancementPipeline',
    'DistortionCorrectionPipeline',
    'LogRestorationPipeline',
    'enhance_low_light',
    'correct_distortion',
    'restore_log',
    
    # Video Enhancements
    'SDRtoHDRPipeline',
    'ObjectRemovalPipeline',
    'FocusFixPipeline',
    'OverexposureFixPipeline',
    'FlatColorProfilePipeline',
    'convert_sdr_to_hdr',
    'remove_objects',
    'fix_focus',
    'fix_overexposure',
    'apply_flat_profile',
    
    # Reencoding
    'ReencodingPipeline',
    'reencode_video',
    'list_encoding_presets',
    'get_video_metadata',
    'REENCODING_PRESETS',
    'HANDBRAKE_PRESETS',
]

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

# Pipeline modules
from .upscale import (
    UpscalePipeline,
    upscale_image,
    upscale_video,
    get_upscale_model,
    list_available_upscale_models,
)

from .face_recovery import (
    FaceRecoveryPipeline,
    recover_face,
    get_face_model,
    list_face_models,
)

from .sharpening import (
    SharpenPipeline,
    sharpen_image,
    get_sharpen_model,
    list_sharpen_models,
)

from .denoising import (
    DenoisePipeline,
    denoise_image,
    get_denoise_model,
    list_denoise_models,
)

from .frame_interpolation import (
    FrameInterpolationPipeline,
    interpolate_video,
    get_frame_interp_model,
    list_frame_interp_models,
)

from .slow_motion import (
    SlowMotionPipeline,
    create_slowmo,
)

from .restoration import (
    FrameRecoveryPipeline,
    MultipurposeRestorationPipeline,
    recover_frames,
    multipurpose_restore,
    get_multipurpose_model,
    list_multipurpose_models,
)

from .color_transfer import (
    ColorizationPipeline,
    MatchColorPipeline,
    colorize_image,
    match_color,
    get_match_color_model,
)

from .low_light import (
    LowLightEnhancementPipeline,
    DistortionCorrectionPipeline,
    LogRestorationPipeline,
    enhance_low_light,
    correct_distortion,
    restore_log,
)

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

from .reencoding import (
    ReencodingPipeline,
    reencode_video,
    list_encoding_presets,
    get_video_metadata,
    REENCODING_PRESETS,
    HANDBRAKE_PRESETS,
)

# Pipeline Chaining
from .pipeline import PipelineChain

# Unified Model Registry
from src.models.unified_registry import (
    get_unified_registry,
    get_models_by_feature,
    get_model_by_feature_and_name,
    get_all_features,
    get_global_features,
    MODEL_REGISTRY_EXPORT,
    GLOBAL_FEATURES_EXPORT,
)

# Backwards compatibility - expose individual model dictionaries
UPSCALE_MODELS = {}
FACE_MODELS = {}
SHARPEN_MODELS = {}
DENOISE_MODELS = {}
FRAME_INTERP_MODELS = {}
SLOWMO_MODELS = {}
FRAME_RECOVERY_MODELS = {}
MULTIPURPOSE_MODELS = {}
DIFFUSION_RESTORERS = {}
MATCH_COLOR_MODELS = {}

# Initialize backwards compatibility
from src.models.unified_registry import _init_backward_compatibility
_init_backward_compatibility()

__all__ = [
    # Base classes
    'EnhancementPipeline',
    'ModelLoadError',
    'InferenceError',
    'UnsupportedFormatError',
    'ProcessingError',
    'ProcessingAborted',
    
    # Pipeline classes
    'UpscalePipeline',
    'FaceRecoveryPipeline', 
    'SharpenPipeline',
    'DenoisePipeline',
    'FrameInterpolationPipeline',
    'SlowMotionPipeline',
    'FrameRecoveryPipeline',
    'MultipurposeRestorationPipeline',
    'ColorizationPipeline',
    'MatchColorPipeline',
    'LowLightEnhancementPipeline',
    'DistortionCorrectionPipeline',
    'LogRestorationPipeline',
    'SDRtoHDRPipeline',
    'ObjectRemovalPipeline',
    'FocusFixPipeline',
    'OverexposureFixPipeline',
    'FlatColorProfilePipeline',
    'ReencodingPipeline',
    'PipelineChain',
    
    # Processing functions
    'upscale_image',
    'upscale_video',
    'recover_face',
    'sharpen_image',
    'denoise_image',
    'interpolate_video',
    'create_slowmo',
    'recover_frames',
    'multipurpose_restore',
    'colorize_image',
    'match_color',
    'enhance_low_light',
    'correct_distortion',
    'restore_log',
    'convert_sdr_to_hdr',
    'remove_objects',
    'fix_focus',
    'fix_overexposure',
    'apply_flat_profile',
    'reencode_video',
    
    # Model access functions
    'get_unified_registry',
    'get_models_by_feature',
    'get_model_by_feature_and_name',
    'get_all_features',
    'get_global_features',
    
    # Unified registry exports
    'MODEL_REGISTRY_EXPORT',
    'GLOBAL_FEATURES_EXPORT',
    
    # Backward compatibility - individual model dictionaries
    'UPSCALE_MODELS',
    'FACE_MODELS',
    'SHARPEN_MODELS',
    'DENOISE_MODELS',
    'FRAME_INTERP_MODELS',
    'SLOWMO_MODELS',
    'FRAME_RECOVERY_MODELS',
    'MULTIPURPOSE_MODELS',
    'DIFFUSION_RESTORERS',
    'MATCH_COLOR_MODELS',
]
"""
Unified Model Registry for openVideoAI
Replaces hardcoded model dictionaries with a dynamic catalog-driven registry
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from .downloader import ModelInfo
class UnifiedModelRegistry:
    """
    Unified model registry that combines models.json catalog with hardcoded models
    for backward compatibility
    """
    
    def __init__(self):
        self._catalog_cache = None
        self._hardcoded_cache = None
        self._registry_cache = None
        
    def _load_catalog(self) -> Dict[str, Any]:
        """Load models.json catalog"""
        if self._catalog_cache is not None:
            return self._catalog_cache
            
        catalog_path = Path("models.json")
        if catalog_path.exists():
            with open(catalog_path, 'r', encoding='utf-8') as f:
                self._catalog_cache = json.load(f)
        else:
            self._catalog_cache = {"models": []}
            
        return self._catalog_cache
    
    def _load_hardcoded_models(self) -> Dict[str, Dict]:
        """Load hardcoded model dictionaries from processing modules"""
        if self._hardcoded_cache is not None:
            return self._hardcoded_cache
            
        hardcoded = {}
        
        try:
            from src.processing.upscale import UPSCALE_MODELS
            hardcoded.update({
                "Upscale": {k: {**v, "name": k, "is_builtin": True} 
                           for k, v in UPSCALE_MODELS.items()}
            })
        except ImportError:
            pass
            
        try:
            from src.processing.face_recovery import FACE_MODELS
            hardcoded.update({
                "Face Recovery": {k: {**v, "name": k, "is_builtin": True}
                                for k, v in FACE_MODELS.items()}
            })
        except ImportError:
            pass
            
        try:
            from src.processing.sharpening import SHARPEN_MODELS
            hardcoded.update({
                "Sharpen / Deblur": {k: {**v, "name": k, "is_builtin": True}
                                    for k, v in SHARPEN_MODELS.items()}
            })
        except ImportError:
            pass
            
        try:
            from src.processing.denoising import DENOISE_MODELS
            hardcoded.update({
                "Denoise": {k: {**v, "name": k, "is_builtin": True}
                           for k, v in DENOISE_MODELS.items()}
            })
        except ImportError:
            pass
            
        try:
            from src.processing.frame_interpolation import FRAME_INTERP_MODELS
            hardcoded.update({
                "Frame Interpolation": {k: {**v, "name": k, "is_builtin": True}
                                       for k, v in FRAME_INTERP_MODELS.items()}
            })
        except ImportError:
            pass
            
        try:
            from src.processing.slow_motion import SLOWMO_MODELS
            hardcoded.update({
                "SloMo": {k: {**v, "name": k, "is_builtin": True}
                         for k, v in SLOWMO_MODELS.items()}
            })
        except ImportError:
            pass
            
        try:
            from src.processing.restoration import FRAME_RECOVERY_MODELS, MULTIPURPOSE_MODELS
            hardcoded.update({
                "Multipurpose Restoration": {**FRAME_RECOVERY_MODELS, **MULTIPURPOSE_MODELS},
                "Frame Recovery": {**FRAME_RECOVERY_MODELS, **MULTIPURPOSE_MODELS}
            })
        except ImportError:
            pass
            
        try:
            from src.processing.color_transfer import MATCH_COLOR_MODELS
            hardcoded.update({
                "Match Color": {k: {**v, "name": k, "is_builtin": True}
                               for k, v in MATCH_COLOR_MODELS.items()}
            })
        except ImportError:
            pass
            
        try:
            from src.processing.low_light import get_low_light_models
            if callable(get_low_light_models):
                low_light_models = get_low_light_models()
                hardcoded.update({
                    "Low Light": low_light_models
                })
        except ImportError:
            pass
            
        self._hardcoded_cache = hardcoded
        return hardcoded
    
    def _merge_catalog_and_hardcoded(self) -> Dict[str, List[Dict]]:
        """Merge catalog and hardcoded models"""
        if self._registry_cache is not None:
            return self._registry_cache
            
        catalog = self._load_catalog()
        hardcoded = self._load_hardcoded_models()
        
        # Map catalog features to processing keys
        feature_to_key = {
            "Upscale": "Upscale",
            "Face Recovery": "Face Recovery", 
            "Sharpen / Deblur": "Sharpen / Deblur",
            "Denoise": "Denoise",
            "Frame Interpolation": "Frame Interpolation",
            "SloMo": "SloMo",
            "Log Restoration": "Low Light",
            "Match Color": "Match Color",
            "Multipurpose Restoration": "Multipurpose Restoration",
            "Multipurpose Pipeline": "Multipurpose Restoration",
            "Frame Recovery": "Multipurpose Restoration",
            "Focus Fix": "Multipurpose Restoration",
            "Overexposure Fix": "Multipurpose Restoration", 
            "Flat Color Profile": "Multipurpose Restoration",
            "Object Removal": "Multipurpose Restoration",
            "SDR -> HDR": "Multipurpose Restoration",
            "Low Light": "Low Light",
            "Distortion Correction": "Low Light",
        }
        
        registry = {}
        
        # Add catalog models
        for catalog_model in catalog.get("models", []):
            model_name = catalog_model["name"]
            model_features = [f["name"] for f in catalog_model["features"]]
            
            for feature_name in model_features:
                registry_key = feature_to_key.get(feature_name, feature_name)
                if registry_key not in registry:
                    registry[registry_key] = []
                
                # Check if this catalog model already exists
                for existing_model in registry[registry_key]:
                    if existing_model["name"] == model_name:
                        break
                else:
                    # Add new catalog model
                    registry[registry_key].append({
                        "name": model_name,
                        "description": catalog_model["description"],
                        "description_short": catalog_model.get("descriptionShort", catalog_model["description"]),
                        "size": catalog_model["size"],
                        "weight": catalog_model["weight"],
                        "url": catalog_model.get("url"),
                        "downloadUrl": catalog_model.get("downloadUrl"),
                        "requires_cuda": catalog_model.get("requiresCuda", False),
                        "requires_metal": catalog_model.get("requiresMetal", False),
                        "can_run_on_cpu": catalog_model.get("canRunOnCpu", True),
                        "is_ai": catalog_model.get("isAi", True),
                        "is_builtin": False,
                        "features": [f["name"] for f in catalog_model["features"]],
                    })
        
        # Add hardcoded models (with priority)
        for feature_key, models in hardcoded.items():
            if feature_key not in registry:
                registry[feature_key] = []
                
            for model_name, model_info in models.items():
                # Check if this hardcoded model already exists
                for existing_model in registry[feature_key]:
                    if existing_model["name"] == model_name:
                        # Update existing model with hardcoded info
                        for key, value in model_info.items():
                            if key != "name" and key not in ["features"]:
                                existing_model[key] = value
                        break
                else:
                    # Add new hardcoded model
                    registry[feature_key].append({
                        "name": model_name,
                        "description": model_info.get("description", ""),
                        "size": "~??MB",
                        "weight": "Unknown",
                        "is_builtin": True,
                        **{k: v for k, v in model_info.items() if k not in ["name", "is_builtin"]}
                    })
        
        self._registry_cache = registry
        return registry
    
    def get_registry(self) -> Dict[str, List[Dict]]:
        """Get unified model registry"""
        return self._merge_catalog_and_hardcoded()
    
    def get_models_by_feature(self, feature_name: str) -> List[Dict]:
        """Get models for a specific feature"""
        registry = self.get_registry()
        return registry.get(feature_name, [])
    
    def get_model_by_feature_and_name(self, feature_name: str, model_name: str) -> Optional[Dict]:
        """Get a specific model by feature and name"""
        models = self.get_models_by_feature(feature_name)
        for model in models:
            if model["name"] == model_name:
                return model
        return None
    
    def get_all_features(self) -> List[str]:
        """Get all available processing features"""
        registry = self.get_registry()
        return list(registry.keys())
    
    def get_global_features(self) -> List[Dict]:
        """Get global features from catalog"""
        catalog = self._load_catalog()
        return catalog.get("features", [])
# Global registry instance
_registry_instance = None

def get_unified_registry() -> UnifiedModelRegistry:
    """Get or create the global unified registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = UnifiedModelRegistry()
    return _registry_instance

def get_models_by_feature(feature_name: str) -> List[Dict]:
    """Get models for a specific feature (convenience function)"""
    registry = get_unified_registry()
    return registry.get_models_by_feature(feature_name)

def get_model_by_feature_and_name(feature_name: str, model_name: str) -> Optional[Dict]:
    """Get a specific model by feature and name (convenience function)"""
    registry = get_unified_registry()
    return registry.get_model_by_feature_and_name(feature_name, model_name)

def get_all_features() -> List[str]:
    """Get all available processing features (convenience function)"""
    registry = get_unified_registry()
    return registry.get_all_features()

def get_global_features() -> List[Dict]:
    """Get global features from catalog (convenience function)"""
    registry = get_unified_registry()
    return registry.get_global_features()

# Initialize hardcoded model dictionaries for backward compatibility
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

def _init_backward_compatibility():
    """Initialize backwards compatibility model dictionaries"""
    registry = get_unified_registry()
    
    # Upscale
    for model in registry.get_models_by_feature("Upscale"):
        if model.get("is_builtin", False):
            UPSCALE_MODELS[model["name"]] = {
                "name": model["name"],
                "description": model.get("description", ""),
                "requires_cuda": model.get("requires_cuda", False),
                "requires_metal": model.get("requires_metal", False),
                "can_run_on_cpu": model.get("can_run_on_cpu", True),
            }
    
    # Face Recovery
    for model in registry.get_models_by_feature("Face Recovery"):
        if model.get("is_builtin", False):
            FACE_MODELS[model["name"]] = {
                "name": model["name"],
                "description": model.get("description", ""),
                "requires_cuda": model.get("requires_cuda", False),
                "requires_metal": model.get("requires_metal", False),
                "can_run_on_cpu": model.get("can_run_on_cpu", True),
            }
    
    # Sharpen / Deblur
    for model in registry.get_models_by_feature("Sharpen / Deblur"):
        if model.get("is_builtin", False):
            SHARPEN_MODELS[model["name"]] = {
                "name": model["name"],
                "description": model.get("description", ""),
                "requires_cuda": model.get("requires_cuda", False),
                "requires_metal": model.get("requires_metal", False),
                "can_run_on_cpu": model.get("can_run_on_cpu", True),
            }
    
    # Denoise
    for model in registry.get_models_by_feature("Denoise"):
        if model.get("is_builtin", False):
            DENOISE_MODELS[model["name"]] = {
                "name": model["name"],
                "description": model.get("description", ""),
                "requires_cuda": model.get("requires_cuda", False),
                "requires_metal": model.get("requires_metal", False),
                "can_run_on_cpu": model.get("can_run_on_cpu", True),
            }
    
    # Frame Interpolation
    for model in registry.get_models_by_feature("Frame Interpolation"):
        if model.get("is_builtin", False):
            FRAME_INTERP_MODELS[model["name"]] = {
                "name": model["name"],
                "description": model.get("description", ""),
                "requires_cuda": model.get("requires_cuda", False),
                "requires_metal": model.get("requires_metal", False),
                "can_run_on_cpu": model.get("can_run_on_cpu", True),
            }
    
    # SloMo
    for model in registry.get_models_by_feature("SloMo"):
        if model.get("is_builtin", False):
            SLOWMO_MODELS[model["name"]] = {
                "name": model["name"],
                "description": model.get("description", ""),
                "requires_cuda": model.get("requires_cuda", False),
                "requires_metal": model.get("requires_metal", False),
                "can_run_on_cpu": model.get("can_run_on_cpu", True),
            }
    
    # Match Color
    for model in registry.get_models_by_feature("Match Color"):
        if model.get("is_builtin", False):
            MATCH_COLOR_MODELS[model["name"]] = {
                "name": model["name"],
                "description": model.get("description", ""),
                "requires_cuda": model.get("requires_cuda", False),
                "requires_metal": model.get("requires_metal", False),
                "can_run_on_cpu": model.get("can_run_on_cpu", True),
            }
    
    # Multipurpose Restoration
    for model in registry.get_models_by_feature("Multipurpose Restoration"):
        if model.get("is_builtin", False):
            FRAME_RECOVERY_MODELS[model["name"]] = {
                "name": model["name"],
                "description": model.get("description", ""),
                "requires_cuda": model.get("requires_cuda", False),
                "requires_metal": model.get("requires_metal", False),
                "can_run_on_cpu": model.get("can_run_on_cpu", True),
            }
            MULTIPURPOSE_MODELS[model["name"]] = {
                "name": model["name"],
                "description": model.get("description", ""),
                "requires_cuda": model.get("requires_cuda", False),
                "requires_metal": model.get("requires_metal", False),
                "can_run_on_cpu": model.get("can_run_on_cpu", True),
            }

# Initialize backwards compatibility
_init_backward_compatibility()

# Export unified registry for API
MODEL_REGISTRY_EXPORT = get_unified_registry().get_registry()
GLOBAL_FEATURES_EXPORT = get_unified_registry().get_global_features()
__all__ = [
    "UnifiedModelRegistry",
    "get_unified_registry",
    "get_models_by_feature",
    "get_model_by_feature_and_name",
    "get_all_features",
    "get_global_features",
    "MODEL_REGISTRY_EXPORT",
    "GLOBAL_FEATURES_EXPORT",
    # Backwards compatibility
    "UPSCALE_MODELS",
    "FACE_MODELS",
    "SHARPEN_MODELS",
    "DENOISE_MODELS",
    "FRAME_INTERP_MODELS",
    "SLOWMO_MODELS",
    "FRAME_RECOVERY_MODELS",
    "MULTIPURPOSE_MODELS",
    "DIFFUSION_RESTORERS",
    "MATCH_COLOR_MODELS",
]
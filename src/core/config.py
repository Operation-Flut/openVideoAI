"""
Core configuration and application paths for openVideoAI
"""
from pathlib import Path
import json
from typing import Dict, Any
from ..utils import system_resources
from . import paths as _paths

__version__ = "1.0.0"

# These are evaluated at import time so they stay stable for the process
# lifetime, but they already call the right resolver (dev vs. frozen).
PROJECT_ROOT       = _paths.project_root()
SETTINGS_FILE      = _paths.settings_file()
TEMP_PROCESSING_DIR = _paths.temp_processing_dir()
MODELS_DIR         = _paths.models_dir()
EXPORTS_DIR        = _paths.exports_dir()

# Defaults use the same nested shape as ``SettingsModel`` (and as the
# settings.json written to disk), so ``SettingsModel(**DEFAULT_SETTINGS)``
# actually applies them. The previous flat camelCase shape was silently
# ignored by pydantic, which meant e.g. the machine-derived thread count never
# took effect.
DEFAULT_SETTINGS: Dict[str, Any] = {
    # System Resources (RAM/VRAM are percentages of system max, CPU is thread count)
    "system": {
        "max_ram_usage": 85,
        "max_vram_usage": 95,
        "cpu_threads": max(1, min(128, system_resources.cpu_threads - 1)),
        "tile_size": 400,
    },
    # GPU Acceleration
    "gpu": {
        "compute_backend": "auto",  # auto, cuda, amd, hip, dml, metal, cpu
        "gpu_device": "auto",
        "fp16_enabled": True,
        "tensorrt_enabled": False,
    },
    # Standard Export & Processing Defaults
    "export": {
        "default_upscale": 4,
        "default_tile_size": 400,
        "output_format": "jpg",
        "jpg_quality": 95,
        "sharpen_overlap": 128,
        "sharpen_unsharp_amount": 0.5,
        "export_dir": "./exports",
        "export_naming": "{name}_enhanced_{scale}x",
        "overwrite_existing": False,
        "video_codec": "libx264",
        "video_crf": 18,
    },
    # Model Management
    "models": {
        "model_dir": "./models",
        "auto_download_models": True,
        "verify_models": True,
        "hide_incompatible_models": False,
        "prefer_ai": True,
        "advanced_mode": False,
    },
    # Appearance
    "appearance": {
        "theme": "dark",
        "accent_color": "#3b82f6",
        "font_size": "medium",
        "density": "comfortable",
    },
    # Advanced
    "advanced": {
        "debug_logging": False,
        "show_metrics": True,
        "experimental_features": False,
    },
}

# Flat camelCase on purpose: ``GET /api/settings`` returns this dict as
# ``max_values`` and the UI reads ``max_values.cpuThreads`` / ``max_values.tileSize``
# to bound its sliders. The service maps these keys onto the nested settings
# fields when clamping/validating (see ``settings_service``).
MAX_SETTINGS: Dict[str, Any] = {
    "maxRamUsage": 100,
    "maxVramUsage": 100,
    "cpuThreads": system_resources.cpu_threads,
    "tileSize": 1024,
    "sharpenOverlap": 256,
    "sharpenUnsharpAmount": 2,
}


def read_saved_settings(settings_file: Path = SETTINGS_FILE) -> Dict[str, Any]:
    """Read saved settings from disk"""
    if not settings_file.exists():
        return {}
    try:
        data = json.loads(settings_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return data
    except (OSError, json.JSONDecodeError):
        return {}


def write_saved_settings(settings: Dict[str, Any], settings_file: Path = SETTINGS_FILE) -> None:
    """Save settings to disk"""
    settings_file.write_text(json.dumps(settings, indent=2), encoding="utf-8")

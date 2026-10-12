"""
Core configuration and application paths for openVideoAI
"""
from pathlib import Path
import json
from typing import Dict, Any
from ..utils import system_resources
from . import paths as _paths

# These are evaluated at import time so they stay stable for the process
# lifetime, but they already call the right resolver (dev vs. frozen).
PROJECT_ROOT       = _paths.project_root()
SETTINGS_FILE      = _paths.settings_file()
TEMP_PROCESSING_DIR = _paths.temp_processing_dir()
MODELS_DIR         = _paths.models_dir()
EXPORTS_DIR        = _paths.exports_dir()

DEFAULT_SETTINGS: Dict[str, Any] = {
    # System Resources (RAM/VRAM are percentages of system max, CPU is thread count)
    "maxRamUsage": 85,
    "maxVramUsage": 95,
    "cpuThreads": max(1, system_resources.cpu_threads - 1),
    "tileSize": 400,

    # GPU Acceleration
    "computeBackend": "auto",  # auto, cuda, amd, hip, dml, metal, cpu
    "gpuDevice": "auto",
    "fp16Enabled": True,
    "tensorrtEnabled": False,

    # Standard Export & Processing Defaults
    "defaultUpscale": 4,
    "defaultTileSize": 400,
    "outputFormat": "jpg",
    "jpgQuality": 95,
    "sharpenOverlap": 128,
    "sharpenUnsharpAmount": 0.5,
    "exportDir": "./exports",
    "exportNaming": "{name}_enhanced_{scale}x",
    "overwriteExisting": False,
    "videoCodec": "libx264",
    "videoCrf": 18,

    # Model Management
    "modelDir": "./models",
    "autoDownloadModels": True,
    "verifyModels": True,
    "hideIncompatibleModels": False,
    "preferAi": True,
    "advancedMode": False,

    # Appearance
    "theme": "dark",
    "accentColor": "#3b82f6",
    "fontSize": "medium",
    "density": "comfortable",

    # Advanced
    "debugLogging": False,
    "showMetrics": True,
    "experimentalFeatures": False
}

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

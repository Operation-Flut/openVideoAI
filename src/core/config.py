"""
Core configuration and application paths for openVideoAI
"""
from pathlib import Path
import json
from typing import Dict, Any
from ..utils import system_resources

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SETTINGS_FILE = PROJECT_ROOT / "settings.json"
TEMP_PROCESSING_DIR = PROJECT_ROOT / "temp_processing"
MODELS_DIR = PROJECT_ROOT / "models"
EXPORTS_DIR = PROJECT_ROOT / "exports"

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

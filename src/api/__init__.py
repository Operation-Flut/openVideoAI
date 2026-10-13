"""
FastAPI backend for openVideoAI
Serves the UI and provides REST API for model management

Backward-compatibility aliases for existing consumers:
  - main.py  : ``from src.api import app``
  - tests/   : ``from src import api``
  - desktop mode: ``api.request_processing_shutdown()`` / ``api.has_active_processing()``
"""
from .main import app
from .state import (
    download_progress,
    download_tasks,
    process_progress,
    process_tasks,
    _shutdown_requested,
    request_processing_shutdown,
    is_shutdown_requested,
    has_active_processing,
    PROJECT_ROOT,
    SETTINGS_FILE,
    TEMP_PROCESSING_DIR,
    read_saved_settings as _read_saved_settings,
    write_saved_settings as _write_saved_settings,
)

# Re-export so ``tests/test_settings_api.py`` can do ``api.update_settings(...)``,
# ``api.SettingsModel(...)``, ``api.get_settings()``, etc.
from .routes import models as _models
from .routes import settings as _settings
from .routes import processing as _processing
from .routes import health as _health

# Functions
list_models = _models.list_models
get_features = _models.get_features
get_modules = _models.get_modules
get_implemented_features = _models.get_implemented_features
get_model = _models.get_model
start_download = _models.start_download
get_download_progress = _models.get_download_progress
cancel_download = _models.cancel_download
delete_model = _models.delete_model
get_status = _models.get_status

get_settings = _settings.get_settings
update_settings = _settings.update_settings
update_model_dir = _settings.update_model_dir
update_export_dir = _settings.update_export_dir

get_system_info = _health.get_system_info
get_gpus = _health.get_gpus
get_system_capabilities = _health.get_system_capabilities

# Processing
process_upscale_image = _processing.process_upscale_image
process_upscale_file = _processing.process_upscale_file
get_processing_status = _processing.get_processing_status
get_processing_result = _processing.get_processing_result
get_processing_features = _processing.get_processing_features
get_upscale_models = _processing.get_upscale_models
process_face_recovery_file = _processing.process_face_recovery_file
process_sharpen_file = _processing.process_sharpen_file
process_denoise_file = _processing.process_denoise_file
process_frame_interpolation_file = _processing.process_frame_interpolation_file
process_slow_motion_file = _processing.process_slow_motion_file
process_match_color_file = _processing.process_match_color_file
process_low_light_file = _processing.process_low_light_file
process_log_restoration_file = _processing.process_log_restoration_file
process_sdr_to_hdr_file = _processing.process_sdr_to_hdr_file
process_overexposure_fix_file = _processing.process_overexposure_fix_file
process_flat_profile_file = _processing.process_flat_profile_file
process_reencode_file = _processing.process_reencode_file

# Schemas
from src.schemas import SettingsModel

# Startup event (for tests that trigger it)
startup_event = _models.startup_event

__all__ = [
    "app",
    "download_progress",
    "download_tasks",
    "process_progress",
    "process_tasks",
    "request_processing_shutdown",
    "has_active_processing",
    "SETTINGS_FILE",
    "TEMP_PROCESSING_DIR",
    "SettingsModel",
    "list_models",
    "get_features",
    "get_modules",
    "get_implemented_features",
    "get_model",
    "start_download",
    "get_download_progress",
    "cancel_download",
    "delete_model",
    "get_status",
    "get_settings",
    "update_settings",
    "update_model_dir",
    "update_export_dir",
    "get_system_info",
    "get_gpus",
    "get_system_capabilities",
    "process_upscale_image",
    "process_upscale_file",
    "get_processing_status",
    "get_processing_result",
    "get_processing_features",
    "get_upscale_models",
    "process_face_recovery_file",
    "process_sharpen_file",
    "process_denoise_file",
    "process_frame_interpolation_file",
    "process_slow_motion_file",
    "process_match_color_file",
    "process_low_light_file",
    "process_log_restoration_file",
    "process_sdr_to_hdr_file",
    "process_overexposure_fix_file",
    "process_flat_profile_file",
    "process_reencode_file",
]

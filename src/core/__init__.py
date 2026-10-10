"""
Core package for openVideoAI
"""
from .config import (
    PROJECT_ROOT,
    SETTINGS_FILE,
    TEMP_PROCESSING_DIR,
    MODELS_DIR,
    EXPORTS_DIR,
    DEFAULT_SETTINGS,
    MAX_SETTINGS,
    read_saved_settings,
    write_saved_settings,
)

__all__ = [
    'PROJECT_ROOT',
    'SETTINGS_FILE',
    'TEMP_PROCESSING_DIR',
    'MODELS_DIR',
    'EXPORTS_DIR',
    'DEFAULT_SETTINGS',
    'MAX_SETTINGS',
    'read_saved_settings',
    'write_saved_settings',
]

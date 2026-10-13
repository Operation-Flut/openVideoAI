"""
Shared mutable state for the openVideoAI API.

Extracted from the former monolithic src/api.py so that route modules
(models, processing, settings, health) can share progress tracking and
the cooperative-shutdown flag without circular imports.

Tests can monkeypatch ``SETTINGS_FILE`` on this module to redirect
settings persistence (see tests/test_settings_api.py).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from fastapi import HTTPException

from src.core import paths as _paths
from src.models.initialize import DownloadProgress

PROJECT_ROOT = _paths.project_root()
SETTINGS_FILE = _paths.settings_file()
TEMP_PROCESSING_DIR = _paths.temp_processing_dir()

# Progress tracking for active downloads
download_progress: Dict[str, DownloadProgress] = {}
download_tasks: Dict[str, asyncio.Task] = {}

# Progress tracking for active processing
process_progress: Dict[str, Dict[str, Any]] = {}
process_tasks: Dict[str, Any] = {}

# Cooperative shutdown flag: set when the desktop window closes so any in-flight
# processing task aborts instead of silently continuing in the background
# (which previously left a zombie server holding the port).
_shutdown_requested = False


def request_processing_shutdown() -> None:
    """Signal running processing tasks to abort (called on window close)."""
    global _shutdown_requested
    _shutdown_requested = True


def is_shutdown_requested() -> bool:
    """True once a cooperative shutdown has been requested."""
    return _shutdown_requested


def has_active_processing() -> bool:
    """True when at least one processing task is still running."""
    return any(p.get('status') == 'processing' for p in process_progress.values())


# ---------------------------------------------------------------------------
# Settings persistence helpers (raise HTTPException on I/O errors, as before)
# ---------------------------------------------------------------------------

def read_saved_settings() -> Dict[str, Any]:
    if not SETTINGS_FILE.exists():
        return {}
    try:
        settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read settings.json: {exc}") from exc
    if not isinstance(settings, dict):
        raise HTTPException(status_code=500, detail="settings.json must contain a JSON object")
    return settings


def write_saved_settings(settings: Dict[str, Any]) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not save settings.json: {exc}") from exc

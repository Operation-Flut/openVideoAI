"""
Path resolution for openVideoAI.

Dev mode   – running from source (python main.py, pytest, etc.)
            All paths are relative to the project root, exactly as before.

Distributed – running as a frozen PyInstaller executable
            • Settings + models  → %APPDATA%/openVideoAI   (persistent, user-writable)
            • Temp processing    → %TEMP%/openVideoAI       (cleared on reboot)
            • Log file           → %APPDATA%/openVideoAI/logs/openVideoAI.log
            • Read-only assets   → sys._MEIPASS (the PyInstaller bundle)

Detection: PyInstaller sets sys.frozen = True and sys._MEIPASS at startup.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def is_frozen() -> bool:
    """True when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def bundle_dir() -> Path:
    """
    The directory that contains the extracted PyInstaller bundle (sys._MEIPASS)
    when frozen, or the project root when running from source.
    """
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# User-data roots
# ---------------------------------------------------------------------------

def _appdata_root() -> Path:
    """
    %APPDATA%/openVideoAI on Windows, ~/.config/openVideoAI elsewhere.
    Created on first access.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    root = base / "openVideoAI"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _temp_root() -> Path:
    """
    %TEMP%/openVideoAI on Windows, /tmp/openVideoAI elsewhere.
    Created on first access.
    """
    base = Path(os.environ.get("TEMP", os.environ.get("TMP", "/tmp")))
    root = base / "openVideoAI"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Public path accessors
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """
    Project root in dev mode; the PyInstaller bundle dir when frozen.
    Use only for read-only bundled assets (UI, models.json, icon …).
    """
    return bundle_dir()


def settings_file() -> Path:
    if is_frozen():
        return _appdata_root() / "settings.json"
    return bundle_dir() / "settings.json"


def models_dir() -> Path:
    if is_frozen():
        d = _appdata_root() / "models"
    else:
        d = bundle_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_json() -> Path:
    """
    models.json is a read-only asset bundled with the application.
    Both dev and frozen point at the bundle/project root.
    """
    return bundle_dir() / "models.json"


def luts_dir() -> Path:
    """
    Cubes (``.cube`` 3D LUTs) the Export dialog can apply without typing a path.

    User content, so it lives next to the models when frozen and in the project
    root in dev mode.
    """
    if is_frozen():
        d = _appdata_root() / "luts"
    else:
        d = bundle_dir() / "luts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def temp_processing_dir() -> Path:
    if is_frozen():
        d = _temp_root() / "processing"
    else:
        d = bundle_dir() / "temp_processing"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    """Log directory – only meaningful in distributed mode."""
    if is_frozen():
        d = _appdata_root() / "logs"
    else:
        d = bundle_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_file() -> Path:
    return logs_dir() / "openVideoAI.log"


def exports_dir() -> Path:
    if is_frozen():
        # Default to Documents/openVideoAI in distributed builds so users can
        # actually find their output files.
        docs = Path.home() / "Documents"
        d = docs / "openVideoAI" / "exports"
    else:
        d = bundle_dir() / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ui_dir() -> Path:
    """The UI/ directory – always bundled, never written to."""
    return bundle_dir() / "UI"


def icon_path() -> Path:
    """Window icon path - always returns .ico for Windows."""
    icon_dir = bundle_dir() / "assets" / "icon"
    
    # On Windows, use .ico format for pywebview
    if sys.platform == "win32":
        icon_file = icon_dir / "black_glass.ico"
        # Fallback to PNG if .ico doesn't exist (will be filtered in main.py)
        return icon_file if icon_file.exists() else icon_dir / "black_glass.png"
    
    # Non-Windows platforms
    icon_file = icon_dir / "black_glass.png"
    return icon_file if icon_file.exists() else icon_dir / "black_glass.ico"

"""
Settings Service - Manages application settings persistence.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from threading import Lock

from ..core.config import read_saved_settings, write_saved_settings, DEFAULT_SETTINGS, MAX_SETTINGS, SETTINGS_FILE
from ..api.schemas import SettingsModel, SettingsResponse
from ..utils.system_resources import get_available_devices, BACKEND_DEVICE_TYPES

logger = logging.getLogger(__name__)

# ``MAX_SETTINGS`` is flat camelCase because it doubles as the UI's
# ``max_values`` payload (bounded sliders). The settings themselves are nested,
# so the flat policy keys are mapped onto their nested fields here; without
# this mapping the clamping/validation compared flat keys against nested ones
# and silently did nothing.
_MAX_SETTINGS_KEY_MAP = {
    "maxRamUsage": "system.max_ram_usage",
    "maxVramUsage": "system.max_vram_usage",
    "cpuThreads": "system.cpu_threads",
    "tileSize": "system.tile_size",
    "sharpenOverlap": "export.sharpen_overlap",
    "sharpenUnsharpAmount": "export.sharpen_unsharp_amount",
}


class SettingsService:
    """Service for managing application settings."""

    def __init__(self, settings_file: Optional[Path] = None):
        self._lock = Lock()
        self._cached_settings: Optional[SettingsModel] = None
        # Injectable so tests (and future multi-profile support) can point the
        # service at another settings.json without touching module globals.
        self._settings_file = Path(settings_file) if settings_file is not None else SETTINGS_FILE

    def get_settings(self) -> SettingsModel:
        """Get current settings (merged with defaults)."""
        with self._lock:
            return self._get_settings_locked().model_copy()

    def _get_settings_locked(self) -> SettingsModel:
        """Return the cached settings, loading them on first use.

        Must be called while ``self._lock`` is held. It never acquires the lock
        itself, so locked callers such as ``update_settings`` cannot deadlock on
        the non-reentrant ``Lock``.
        """
        if self._cached_settings is None:
            self._load_settings()
        return self._cached_settings

    def _load_settings(self):
        """Load settings from disk."""
        saved = read_saved_settings(self._settings_file)
        merged = {**DEFAULT_SETTINGS, **saved}
        self._cached_settings = SettingsModel(**merged)

    def get_raw_settings(self) -> Dict[str, Any]:
        """Get raw merged settings as dict."""
        return self.get_settings().model_dump()

    def update_settings(self, updates: Dict[str, Any]) -> SettingsModel:
        """Update settings with new values."""
        with self._lock:
            # Use the unlocked accessor: this method already holds the lock and
            # ``threading.Lock`` is not reentrant.
            current = self._get_settings_locked().model_dump()

            # Deep merge updates
            def deep_merge(base: Dict, update: Dict) -> Dict:
                result = base.copy()
                for key, value in update.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = deep_merge(result[key], value)
                    else:
                        result[key] = value
                return result

            merged = deep_merge(current, updates)

            # Validate against max values
            merged = self._clamp_to_max(merged)

            # Save to disk
            write_saved_settings(merged, self._settings_file)

            # Update cache
            self._cached_settings = SettingsModel(**merged)
            logger.info("Settings updated and saved")

            return self._cached_settings

    def _max_bounds(self) -> Dict[str, Any]:
        """Maximum values keyed by dotted settings field (e.g. ``system.tile_size``)."""
        return {
            _MAX_SETTINGS_KEY_MAP[key]: value
            for key, value in MAX_SETTINGS.items()
            if key in _MAX_SETTINGS_KEY_MAP
        }

    def _clamp_to_max(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Clamp numeric settings to maximum values."""
        max_flat = self._max_bounds()
        settings_flat = self._flatten_dict(settings)

        for key, max_val in max_flat.items():
            if key in settings_flat and isinstance(settings_flat[key], (int, float)):
                settings_flat[key] = min(settings_flat[key], max_val)

        return self._unflatten_dict(settings_flat)

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
        """Flatten nested dict with dot notation."""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key))
            else:
                items[new_key] = v
        return items

    def _unflatten_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Unflatten dict with dot notation."""
        result = {}
        for key, value in d.items():
            parts = key.split(".")
            target = result
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        return result

    def reset_to_defaults(self) -> SettingsModel:
        """Reset all settings to defaults."""
        with self._lock:
            write_saved_settings({}, self._settings_file)
            self._cached_settings = SettingsModel(**DEFAULT_SETTINGS)
            logger.info("Settings reset to defaults")
            return self._cached_settings

    def get_setting(self, key: str) -> Any:
        """Get a single setting by dot-notation key (e.g., 'gpu.compute_backend')."""
        settings = self.get_settings().model_dump()
        parts = key.split(".")
        value = settings
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value

    def set_setting(self, key: str, value: Any) -> bool:
        """Set a single setting by dot-notation key."""
        updates = {}
        parts = key.split(".")
        target = updates
        for part in parts[:-1]:
            target[part] = {}
            target = target[part]
        target[parts[-1]] = value

        self.update_settings(updates)
        return True

    def export_settings(self, path: Path) -> bool:
        """Export settings to a file."""
        try:
            settings = self.get_settings().model_dump()
            path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
            return True
        except Exception as e:
            logger.error(f"Failed to export settings: {e}")
            return False

    def import_settings(self, path: Path) -> bool:
        """Import settings from a file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.update_settings(data)
                return True
        except Exception as e:
            logger.error(f"Failed to import settings: {e}")
        return False

    def get_max_settings(self) -> Dict[str, Any]:
        """Get maximum allowed values for settings."""
        return MAX_SETTINGS.copy()

    def validate_settings(self, settings: Dict[str, Any]) -> Dict[str, str]:
        """Validate settings and return any errors."""
        errors = {}

        # Check numeric ranges
        max_flat = self._max_bounds()
        settings_flat = self._flatten_dict(settings)

        for key, value in settings_flat.items():
            if key in max_flat and isinstance(value, (int, float)):
                max_val = max_flat[key]
                if value > max_val:
                    errors[key] = f"Value {value} exceeds maximum {max_val}"

        # Check enum values
        enums = {
            "gpu.compute_backend": ["auto", "cuda", "amd", "hip", "dml", "metal", "cpu"],
            "export.output_format": ["source", "jpg", "png", "webp", "tiff", "mp4"],
            "appearance.theme": ["light", "dark", "system"],
            "appearance.font_size": ["small", "medium", "large"],
            "appearance.density": ["compact", "comfortable", "spacious"],
        }

        for key, allowed in enums.items():
            value = settings_flat.get(key)
            if value is not None and value not in allowed:
                errors[key] = f"Invalid value '{value}', allowed: {allowed}"

        return errors


_settings_service_instance: Optional[SettingsService] = None


def get_settings_service() -> SettingsService:
    """Get or create SettingsService instance."""
    global _settings_service_instance
    if _settings_service_instance is None:
        _settings_service_instance = SettingsService()
    return _settings_service_instance


def experimental_features_enabled() -> bool:
    """Whether the user opted into experimental features (Settings -> Advanced).

    Read at request time rather than cached at import so toggling the setting
    takes effect immediately, without a restart. Anything the catalog marks as
    experimental is hidden from the UI and skipped during processing while this
    returns ``False``.
    """
    try:
        return bool(get_settings_service().get_settings().advanced.experimental_features)
    except Exception:
        logger.debug("Could not read experimental_features setting", exc_info=True)
        return False


def configured_compute_device() -> str:
    """The compute backend chosen in Settings, as a torch device string.

    ``auto`` is passed through so the pipeline resolves it to the best backend
    this machine has. Anything else is only returned when the device actually
    exists here; an unavailable choice (a settings file copied from another
    machine, or a GPU that has since disappeared) falls back to ``auto`` rather
    than handing torch a device type it would fail on.
    """
    try:
        raw = get_settings_service().get_setting("gpu.compute_backend") or "auto"
    except Exception:
        logger.debug("Could not read compute_backend setting", exc_info=True)
        return "auto"

    # get_setting returns the pydantic enum, whose str() is "ComputeBackend.DML".
    backend = str(getattr(raw, "value", raw)).strip().lower()
    if backend in ("", "auto"):
        return "auto"

    # "amd" predates the HIP/DirectML split; resolve it to whichever is usable.
    if backend == "amd":
        available = set(get_available_devices())
        backend = "hip" if "hip" in available else "dml"

    device_type = BACKEND_DEVICE_TYPES.get(backend)
    if not device_type or device_type == "auto":
        return "auto"
    if device_type == "cpu" or device_type in get_available_devices():
        return device_type
    logger.info(
        "Configured compute backend %r is not available on this machine; using auto",
        backend,
    )
    return "auto"
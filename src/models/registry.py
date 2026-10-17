"""
Unified Model Registry - Single source of truth for all models.

Replaces hardcoded dictionaries in processing modules with a dynamic,
catalog-driven registry that loads from models.json and local filesystem.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import importlib.util
import json
import logging
import re
from threading import Lock

from ..core.paths import models_dir, models_json

logger = logging.getLogger(__name__)

MEDIA_TYPES = ("image", "video", "both")

# Human-readable form of a `media` value. Served to the UI so the catalog's
# vocabulary is phrased in exactly one place.
MEDIA_LABELS = {
    "image": "Images only",
    "video": "Video only",
    "both": "Images and video",
}


def media_label(media: Any) -> str:
    """Display label for a catalog ``media`` value."""
    return MEDIA_LABELS[_normalise_media(media)]


def _normalise_media(value: Any) -> str:
    """Validate a catalog ``media`` value, defaulting to ``both``.

    Unknown or misspelled values are treated as ``both`` so a typo in the
    catalog can never silently hide a model from a picker.
    """
    media = str(value or "").strip().lower()
    return media if media in MEDIA_TYPES else "both"


@dataclass
class ModelConfig:
    """Configuration for a specific model from the catalog."""
    name: str
    description: str
    description_short: str
    size: str
    weight: str
    url: str
    download_url: Optional[str] = None
    filename: Optional[str] = None
    additional_downloads: List[Dict[str, str]] = field(default_factory=list)
    features: List[Dict[str, str]] = field(default_factory=list)
    # Longer, use-case oriented note shown in the model details popup ("Best used
    # for"). Optional: the popup falls back to the plain description.
    best_for: Optional[str] = None
    is_ai: bool = True
    requires_cuda: bool = False
    requires_metal: bool = False
    can_run_on_cpu: bool = True
    is_builtin: bool = False
    module: Optional[str] = None  # Processing module name
    arch: Optional[str] = None  # Runtime architecture key; None = not implemented
    runtime: Dict[str, Any] = field(default_factory=dict)  # Loader hyperparameters
    default_for: List[str] = field(default_factory=list)  # Features this model is the default for
    requires_module: Optional[str] = None  # Optional importable module gating availability
    experimental: bool = False  # Hidden unless "Experimental Features" is enabled in Settings
    # What the model can be applied to: "image", "video" or "both" (the default,
    # and what the catalog means by leaving it out). Smart Select uses it so a
    # video does not get a stills-only model and vice versa.
    media: str = "both"

    @property
    def feature_names(self) -> List[str]:
        return [f.get("name", "") for f in self.features]

    def has_feature(self, feature_name: str) -> bool:
        return feature_name in self.feature_names

    def supports_media(self, media: Optional[str]) -> bool:
        """Whether this model applies to ``image`` / ``video`` media.

        Unknown media (nothing loaded yet) matches everything, and a model
        marked ``both`` always matches.
        """
        if media not in ("image", "video"):
            return True
        return self.media == "both" or self.media == media

    @property
    def is_implemented(self) -> bool:
        """Whether a pipeline in this build can actually run this model.

        ``arch`` names the runtime architecture the loaders dispatch on; entries
        without one are catalog-only (planned/alternative runtimes such as MLX or
        ONNX). ``requiresModule`` additionally gates models that need an optional
        dependency, e.g. Dual TV-L1 needs ``cv2.optflow`` from OpenCV contrib.
        """
        if not self.arch:
            return False
        if self.requires_module:
            try:
                return importlib.util.find_spec(self.requires_module) is not None
            except (ImportError, ValueError):
                return False
        return True


@dataclass
class ModelStatus:
    """Runtime status of a model."""
    config: ModelConfig
    downloaded: bool = False
    local_path: Optional[Path] = None
    additional_paths: Dict[str, Path] = field(default_factory=dict)
    verified: bool = False
    error: Optional[str] = None


class ModelRegistry:
    """
    Unified model registry that combines catalog metadata with local filesystem state.

    This replaces:
    - Hardcoded UPSCALE_MODELS dict in upscale.py
    - Hardcoded model lists in face_recovery.py, denoising.py, etc.
    - Scattered model lookup logic across the codebase
    """

    _instance: Optional["ModelRegistry"] = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._models: Dict[str, ModelConfig] = {}
        self._status_cache: Dict[str, ModelStatus] = {}
        self._feature_index: Dict[str, List[str]] = {}  # feature -> model names, catalog order
        self._features: List[Dict[str, Any]] = []  # feature metadata declared in the catalog
        self._groups: List[Dict[str, Any]] = []  # Simple-Mode sidebar groups declared in the catalog
        self._initialized = True
        self._load_catalog()

    def _load_catalog(self) -> None:
        """Load model catalog from models.json.

        ``models_json()`` resolves the read-only bundled asset in both dev and
        frozen (PyInstaller) builds; the models directory is a user-writable
        AppData path when frozen and does not contain the catalog.
        """
        catalog_path = models_json()
        if not catalog_path.exists():
            # Fallback for source checkouts where the catalog sits next to models/.
            catalog_path = models_dir().parent / "models.json"
        if not catalog_path.exists():
            logger.warning(f"Catalog not found at {catalog_path}")
            return

        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for model_data in data.get("models", []):
                config = ModelConfig(
                    name=model_data["name"],
                    description=model_data.get("description", ""),
                    description_short=model_data.get("descriptionShort", ""),
                    size=model_data.get("size", "Unknown"),
                    weight=model_data.get("weight", "Unknown"),
                    url=model_data.get("url", ""),
                    download_url=model_data.get("downloadUrl"),
                    filename=model_data.get("filename"),
                    additional_downloads=model_data.get("additionalDownloads", []),
                    features=model_data.get("features", []),
                    best_for=model_data.get("bestFor"),
                    is_ai=model_data.get("isAi", True),
                    requires_cuda=model_data.get("requiresCuda", False),
                    requires_metal=model_data.get("requiresMetal", False),
                    can_run_on_cpu=model_data.get("canRunOnCpu", True),
                    is_builtin=model_data.get("is_builtin", False) or model_data.get("downloadUrl") is None,
                    module=model_data.get("module"),
                    arch=model_data.get("arch"),
                    runtime=model_data.get("runtime", {}),
                    default_for=model_data.get("defaultFor", []),
                    requires_module=model_data.get("requiresModule"),
                    experimental=bool(model_data.get("experimental", False)),
                    media=_normalise_media(model_data.get("media")),
                )
                self._models[config.name] = config

                # Build feature index, preserving catalog order
                for feature in config.feature_names:
                    bucket = self._feature_index.setdefault(feature, [])
                    if config.name not in bucket:
                        bucket.append(config.name)

            # Feature metadata and sidebar groups are declared in models.json too.
            # Keep every declared field (e.g. `alwaysOn`) and normalise the
            # required ones so consumers can rely on them existing.
            self._features = [
                {
                    **feature,
                    "name": feature.get("name", ""),
                    "displayName": feature.get("displayName", feature.get("name", "")),
                    "description": feature.get("description", ""),
                    "experimental": bool(feature.get("experimental", False)),
                }
                for feature in data.get("features", [])
            ]
            self._groups = list(data.get("groups", []))

            logger.info(f"Loaded {len(self._models)} models from catalog")

        except Exception as e:
            logger.exception(f"Failed to load model catalog: {e}")

    def refresh_status(self) -> None:
        """Refresh downloaded status by scanning filesystem."""
        models_path = models_dir()
        self._status_cache.clear()

        for name, config in self._models.items():
            status = ModelStatus(config=config)

            if config.is_builtin:
                status.downloaded = True
                status.local_path = models_path / "builtin"
                status.verified = True
            else:
                # Check main model file
                filename = config.filename or (config.download_url.split("/")[-1] if config.download_url else None)
                if filename:
                    main_path = models_path / filename
                    if main_path.exists():
                        status.downloaded = True
                        status.local_path = main_path

                # Check additional downloads
                for addl in config.additional_downloads:
                    addl_filename = addl.get("filename")
                    addl_url = addl.get("url")
                    if addl_filename:
                        addl_path = models_path / addl_filename
                        if addl_path.exists():
                            status.additional_paths[addl_filename] = addl_path

                # Model is downloaded if main file exists (additional are optional)
                if config.download_url and not status.downloaded:
                    status.downloaded = False

            self._status_cache[name] = status

    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        return self._models.get(name)

    def get_status(self, name: str) -> Optional[ModelStatus]:
        """Get model status (download state, local paths)."""
        return self._status_cache.get(name)

    def get_all_models(self) -> List[ModelConfig]:
        """Get all model configurations."""
        return list(self._models.values())

    def get_all_statuses(self) -> List[ModelStatus]:
        """Get all model statuses."""
        return list(self._status_cache.values())

    def get_downloaded_models(self) -> List[ModelStatus]:
        """Get only downloaded models."""
        return [s for s in self._status_cache.values() if s.downloaded]

    def get_models_for_feature(self, feature: str) -> List[ModelConfig]:
        """Get all models that provide a specific feature, in catalog order."""
        return [self._models[name]
                for name in self._feature_index.get(feature, [])
                if name in self._models]

    def get_implemented_for_feature(self, feature: str) -> List[ModelConfig]:
        """Models for a feature that this build can actually run."""
        return [config for config in self.get_models_for_feature(feature) if config.is_implemented]

    def get_available_model_names(self, feature: str) -> List[str]:
        """Runnable model names for a feature, in catalog order.

        This is the list pipelines advertise to the UI, so it always matches what
        the loaders can construct and what the downloader can resolve to a file.
        """
        return [config.name for config in self.get_implemented_for_feature(feature)]

    def get_default_model(self, feature: str) -> Optional[str]:
        """Default model for a feature as declared by ``defaultFor`` in the catalog."""
        for name in self._feature_index.get(feature, []):
            config = self._models.get(name)
            if config and feature in config.default_for and config.is_implemented:
                return config.name
        implemented = self.get_available_model_names(feature)
        return implemented[0] if implemented else None

    def get_feature_metadata(self) -> List[Dict[str, Any]]:
        """Feature metadata (name, displayName, description) from the catalog."""
        return list(self._features)

    def get_feature(self, name: str) -> Optional[Dict[str, Any]]:
        """Metadata for a single feature, or None when the catalog does not define it."""
        for feature in self._features:
            if feature.get("name") == name:
                return feature
        return None

    def get_feature_display_name(self, name: str) -> str:
        """User-facing feature label declared by the catalog (falls back to the key)."""
        feature = self.get_feature(name)
        return feature.get("displayName") or name if feature else name

    def get_groups(self) -> List[Dict[str, Any]]:
        """Sidebar groups (key, label, description, icon, features) from the catalog."""
        return list(self._groups)

    def is_feature_experimental(self, name: str) -> bool:
        """Whether the catalog marks a feature as experimental."""
        feature = self.get_feature(name)
        return bool(feature.get("experimental")) if feature else False

    def is_model_experimental(self, name: str) -> bool:
        """Whether the catalog marks a model as experimental."""
        config = self._models.get(name)
        return bool(config.experimental) if config else False

    def drop_experimental(self, names: List[str]) -> List[str]:
        """Remove experimental models from a picker list.

        Used when the "Experimental Features" setting is off; the models stay in
        the catalog (and on the Models page) so they can still be downloaded.
        """
        return [name for name in names if not self.is_model_experimental(name)]

    def get_experimental_items(self) -> List[Dict[str, Any]]:
        """Everything the catalog marks as experimental, in catalog order.

        Nothing is marked today, so this is normally empty — the UI shows an
        empty state. Marking an entry in ``models.json`` with
        ``"experimental": true`` (on a feature or a model) is all it takes for
        it to appear here.
        """
        items: List[Dict[str, Any]] = []

        for feature in self._features:
            if feature.get("experimental"):
                items.append({
                    "kind": "feature",
                    "name": feature["name"],
                    "displayName": feature.get("displayName") or feature["name"],
                    "description": feature.get("description", ""),
                    "features": [],
                })

        for name, config in self._models.items():
            if config.experimental:
                items.append({
                    "kind": "model",
                    "name": name,
                    "displayName": name,
                    "description": config.description_short,
                    "features": config.feature_names,
                })

        return items

    def get_model_names_by_arch(self, *archs: str) -> List[str]:
        """Catalog model names whose runtime architecture matches any of ``archs``.

        Used by pipelines that only handle a subset of a feature's models (for
        example the RealBasicVSR video pipeline versus the Real-ESRGAN upscaler,
        which share the "Upscale" feature).
        """
        wanted = {a for a in archs if a}
        return [name for name, cfg in self._models.items()
                if cfg.arch in wanted and cfg.is_implemented]

    def get_default_model_for_arch(self, feature: str, arch: str) -> Optional[str]:
        """Default model for ``feature`` among the models using ``arch``.

        Prefers the model the catalog marks with ``defaultFor``; otherwise uses
        the first catalog model of that architecture.
        """
        candidates = [n for n in self.get_model_names_by_arch(arch)
                      if feature in (self._models[n].default_for or [])]
        if candidates:
            return candidates[0]
        names = self.get_model_names_by_arch(arch)
        return names[0] if names else None

    def get_arch(self, name: str) -> Optional[str]:
        """Runtime architecture key for a model (None when not implemented)."""
        config = self._models.get(name)
        return config.arch if config else None

    def get_media(self, name: str) -> str:
        """Media a model applies to: "image", "video" or "both"."""
        config = self._models.get(name)
        return config.media if config else "both"

    def get_runtime(self, name: str) -> Dict[str, Any]:
        """Loader hyperparameters declared for a model in the catalog."""
        config = self._models.get(name)
        return config.runtime if config else {}

    def runtime_config(self, name: Optional[str], feature: Optional[str] = None) -> Dict[str, Any]:
        """Runtime loader config for a model: its architecture plus hyperparameters.

        Catalog keys are camelCase to stay consistent with the rest of
        ``models.json``; the pipeline loaders read snake_case, so keys are
        translated here and ``is_classical`` is derived from ``isAi`` (catalog
        non-AI entries are exactly the mathematical/classical algorithms).
        """
        entry = self.resolve(name, feature)
        if entry is None:
            return {}
        config: Dict[str, Any] = {
            "arch": entry.arch,
            "is_ai": entry.is_ai,
            "is_classical": not entry.is_ai,
            "model_file": entry.filename,
        }
        for key, value in entry.runtime.items():
            snake = re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()
            config[snake] = value
        return config

    def resolve(self, name: Optional[str], feature: Optional[str] = None) -> Optional[ModelConfig]:
        """Resolve a model name to its catalog entry.

        Unknown or missing names fall back to the feature default so a pipeline
        can always be constructed, matching the previous per-pipeline fallbacks.
        """
        if name and name in self._models:
            return self._models[name]
        if feature:
            default = self.get_default_model(feature)
            if default:
                return self._models[default]
        return None

    def get_downloaded_for_feature(self, feature: str) -> List[ModelStatus]:
        """Get downloaded models for a specific feature."""
        return [
            self._status_cache[name]
            for name in self._feature_index.get(feature, [])
            if name in self._status_cache and self._status_cache[name].downloaded
        ]

    def get_best_model_for_feature(
        self,
        feature: str,
        prefer_ai: bool = True,
        device_type: str = "auto"
    ) -> Optional[ModelStatus]:
        """
        Get the best available model for a feature based on preferences.

        Args:
            feature: Feature name (e.g., "Upscale", "Face Recovery")
            prefer_ai: Prefer AI models over classical algorithms
            device_type: "cuda", "dml", "hip", "metal", "cpu", "auto"
        """
        candidates = self.get_downloaded_for_feature(feature)
        if not candidates:
            return None

        # Filter by device compatibility
        compatible = []
        for status in candidates:
            config = status.config
            if device_type == "auto" or device_type == "cpu":
                if config.can_run_on_cpu:
                    compatible.append(status)
            elif device_type in ("cuda", "dml", "hip"):
                if config.requires_cuda or config.can_run_on_cpu:
                    compatible.append(status)
            elif device_type == "metal":
                if config.requires_metal or config.can_run_on_cpu:
                    compatible.append(status)

        if not compatible:
            return None

        # Sort: AI first (if preferred), then by weight (lighter first)
        weight_order = {
            "Ultra-Lightweight": 0,
            "Ultra-Lightweight (0 VRAM)": 0,
            "Lightweight": 1,
            "Medium": 2,
            "Heavy": 3,
            "Heavyweight": 4,
        }

        def sort_key(s: ModelStatus) -> tuple:
            is_ai = s.config.is_ai
            weight = weight_order.get(s.config.weight, 5)
            return (0 if (prefer_ai and is_ai) else 1, weight)

        compatible.sort(key=sort_key)
        return compatible[0]

    def get_model_path(self, name: str) -> Optional[Path]:
        """Get the local path for a model's main file."""
        status = self._status_cache.get(name)
        if status and status.downloaded:
            return status.local_path
        return None

    def get_all_model_paths(self, name: str) -> Dict[str, Path]:
        """Get all local paths for a model (main + additional)."""
        status = self._status_cache.get(name)
        if status and status.downloaded:
            paths = {}
            if status.local_path:
                paths["main"] = status.local_path
            paths.update(status.additional_paths)
            return paths
        return {}

    def is_downloaded(self, name: str) -> bool:
        """Check if a model is downloaded."""
        status = self._status_cache.get(name)
        return status.downloaded if status else False

    def get_features(self) -> List[str]:
        """Get all available feature names."""
        return list(self._feature_index.keys())

    def get_models_by_module(self, module: str) -> List[ModelConfig]:
        """Get models belonging to a processing module."""
        return [m for m in self._models.values() if m.module == module]


# Convenience functions
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def refresh_registry() -> None:
    """Refresh the global registry's status cache."""
    registry = get_registry()
    registry.refresh_status()


def default_model(feature: str, fallback: str = "") -> str:
    """Default model for a feature, as declared by ``defaultFor`` in models.json.

    ``fallback`` is only used for features with no runnable model in this build.
    """
    return get_registry().get_default_model(feature) or fallback


# Backward compatibility functions (to replace old imports)
def get_upscale_model(model_name: str, models_dir: Optional[str] = None) -> Optional[str]:
    """Backward compatible function for upscale model lookup."""
    registry = get_registry()
    path = registry.get_model_path(model_name)
    return str(path) if path else None


def list_available_upscale_models(models_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Backward compatible function for listing upscale models."""
    registry = get_registry()
    results = []
    for status in registry.get_all_statuses():
        if "Upscale" in status.config.feature_names:
            results.append({
                "name": status.config.name,
                "scale": 4,  # Default, could be extracted from config
                "description": status.config.description_short,
                "downloaded": status.downloaded,
                "path": str(status.local_path) if status.local_path else None,
            })
    return results
"""
Processing pipeline registry.
Auto-discovers and manages all available enhancement pipelines.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Set
import inspect
import logging
from threading import Lock

from .base import EnhancementPipeline, PipelineConfig, PipelineChain

logger = logging.getLogger(__name__)


def instantiate_pipeline(
    pipeline_class: Type[EnhancementPipeline],
    *,
    model_name: str,
    device: str = "auto",
    parameters: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
) -> EnhancementPipeline:
    """Build a pipeline instance, mapping flat JSON parameters onto its ``__init__``.

    The UI sends one flat ``parameters`` dict per step (driven by the pipeline's
    JSON schema), while each pipeline exposes its tunables as explicit constructor
    arguments (``scale``, ``strength``, ``multiplier``, ...). This binds matching
    keys by signature and only forwards ``parameters`` to classes that accept it,
    so construction never raises ``TypeError`` on the keyword itself.
    """
    params = dict(parameters or {})
    sig = inspect.signature(pipeline_class.__init__)

    kwargs: Dict[str, Any] = {
        "model_name": model_name,
        "device": device,
        "task_id": task_id,
    }

    for name in sig.parameters:
        if name in ("self", "model_name", "device", "task_id", "parameters"):
            continue
        if name in params:
            kwargs[name] = params[name]

    if "parameters" in sig.parameters:
        kwargs["parameters"] = params

    return pipeline_class(**kwargs)


@dataclass
class PipelineInfo:
    """Metadata about a registered pipeline.

    A single feature may be implemented by more than one pipeline class (for
    example ``Upscale`` is handled by the generic image/video upscaler *and* the
    multi-frame ``RealBasicVSR`` pipeline). ``pipeline_class``/``config`` describe
    the primary implementation, while ``model_pipeline_classes`` maps each
    available model to the class that actually handles it.
    """
    feature: str
    pipeline_class: Type[EnhancementPipeline]
    config: PipelineConfig
    available_models: List[str] = field(default_factory=list)
    default_model: Optional[str] = None
    model_pipeline_classes: Dict[str, Type[EnhancementPipeline]] = field(default_factory=dict)


class PipelineRegistry:
    """
    Registry for all enhancement pipelines.
    Auto-discovers pipelines from submodules.
    """

    _instance: Optional["PipelineRegistry"] = None
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
        self._pipelines: Dict[str, PipelineInfo] = {}  # feature -> PipelineInfo
        self._model_to_feature: Dict[str, str] = {}  # model_name -> feature
        self._initialized = True
        self._discover_pipelines()

    def _discover_pipelines(self) -> None:
        """Auto-discover pipeline classes from submodules."""
        # Import all pipeline modules to trigger registration
        try:
            from .. import upscale, face, denoise, sharpen, interpolation, color, video
            # The modules will register themselves via register_pipeline()
        except ImportError as e:
            logger.warning(f"Some pipeline modules not available: {e}")

    def register(
        self,
        feature: str,
        pipeline_class: Type[EnhancementPipeline],
        available_models: Optional[List[str]] = None,
        default_model: Optional[str] = None
    ) -> None:
        """Register a pipeline class for a feature.

        Multiple pipeline classes can share a feature (e.g. a feature-specific
        class plus a specialised one such as ``RealBasicVSR`` for ``Upscale``).
        Registrations are merged rather than overwriting each other so no model
        ever disappears from a feature.
        """
        config = pipeline_class.config
        models = list(available_models or [])

        if not models:
            # Nothing runnable to attach (e.g. every model of this
            # architecture is gated by a missing dependency in this build).
            logger.debug(
                "Skipped pipeline registration: %s -> %s (no available models)",
                feature, pipeline_class.__name__,
            )
            return

        existing = self._pipelines.get(feature)

        if existing is None:
            info = PipelineInfo(
                feature=feature,
                pipeline_class=pipeline_class,
                config=config,
                available_models=models,
                default_model=default_model,
                model_pipeline_classes={m: pipeline_class for m in models},
            )
            self._pipelines[feature] = info
        else:
            # Merge into the existing feature, keeping the first pipeline as the
            # primary (its config/schema is exposed for the feature).
            for model in models:
                if model not in existing.available_models:
                    existing.available_models.append(model)
                existing.model_pipeline_classes[model] = pipeline_class
            if existing.default_model is None:
                existing.default_model = default_model

        # Index models -> feature (first feature to claim a model wins)
        for model in models:
            self._model_to_feature.setdefault(model, feature)

        logger.debug(f"Registered pipeline: {feature} -> {pipeline_class.__name__} ({len(models)} models)")

    def get_pipeline_class(
        self,
        feature: str,
        model_name: Optional[str] = None,
    ) -> Optional[Type[EnhancementPipeline]]:
        """Get the pipeline class for a feature (optionally for a specific model)."""
        info = self._pipelines.get(feature)
        if not info:
            return None
        if model_name and model_name in info.model_pipeline_classes:
            return info.model_pipeline_classes[model_name]
        return info.pipeline_class

    def get_pipeline_info(self, feature: str) -> Optional[PipelineInfo]:
        """Get full pipeline info for a feature."""
        return self._pipelines.get(feature)

    def get_all_features(self) -> List[str]:
        """Get all registered features."""
        return list(self._pipelines.keys())

    def get_pipelines_for_models(self, model_names: List[str]) -> List[str]:
        """Get features that can use the given models."""
        features = set()
        for model in model_names:
            if model in self._model_to_feature:
                features.add(self._model_to_feature[model])
        return list(features)

    def create_pipeline(
        self,
        feature: str,
        model_name: str,
        device: str = "auto",
        parameters: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None
    ) -> Optional[EnhancementPipeline]:
        """Create a pipeline instance for a feature with a specific model."""
        pipeline_class = self.get_pipeline_class(feature, model_name)
        if not pipeline_class:
            return None

        # Validate model is available for this pipeline
        info = self._pipelines.get(feature)
        if info and info.available_models and model_name not in info.available_models:
            logger.warning(f"Model {model_name} not in available models for {feature}")

        return instantiate_pipeline(
            pipeline_class,
            model_name=model_name,
            device=device,
            parameters=parameters,
            task_id=task_id,
        )

    def create_chain(
        self,
        steps: List[Dict[str, Any]],
        device: str = "auto",
        task_id: Optional[str] = None
    ) -> Optional[PipelineChain]:
        """
        Create a pipeline chain from a list of step configurations.

        Steps format:
        [
            {"feature": "Upscale", "model": "Real-ESRGAN", "parameters": {"scale": 4}},
            {"feature": "Face Recovery", "model": "CodeFormer", "parameters": {"fidelity": 0.5}},
        ]
        """
        pipelines = []
        for step in steps:
            pipeline = self.create_pipeline(
                feature=step["feature"],
                model_name=step["model"],
                device=device,
                parameters=step.get("parameters"),
                task_id=task_id,
            )
            if pipeline:
                pipelines.append(pipeline)
            else:
                logger.error(f"Failed to create pipeline for {step}")

        if not pipelines:
            return None

        return PipelineChain(pipelines)

    def get_default_model(self, feature: str) -> Optional[str]:
        """Get the default model for a feature."""
        info = self._pipelines.get(feature)
        return info.default_model if info else None

    def get_available_models(self, feature: str) -> List[str]:
        """Get available models for a feature."""
        info = self._pipelines.get(feature)
        return info.available_models if info else []


# Global registry instance
_registry: Optional[PipelineRegistry] = None


def get_pipeline_registry() -> PipelineRegistry:
    """Get the global pipeline registry."""
    global _registry
    if _registry is None:
        _registry = PipelineRegistry()
    return _registry


def register_pipeline(
    feature: str,
    pipeline_class: Type[EnhancementPipeline],
    available_models: Optional[List[str]] = None,
    default_model: Optional[str] = None
) -> None:
    """Convenience function to register a pipeline."""
    get_pipeline_registry().register(feature, pipeline_class, available_models, default_model)
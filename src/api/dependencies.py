"""
API Dependencies - FastAPI dependency injection for services.
"""
from typing import Optional, Dict, Any
from functools import lru_cache
from fastapi import Depends, HTTPException, Header, Request
from pydantic import BaseModel

from ..core.config import read_saved_settings, write_saved_settings, DEFAULT_SETTINGS, MAX_SETTINGS
from ..core.paths import settings_file, models_dir, exports_dir, temp_processing_dir
from ..models.registry import ModelRegistry, get_registry
from ..processing.pipeline.registry import (
    PipelineRegistry,
    get_pipeline_registry as _get_pipeline_registry,
)
from ..services.model_service import ModelService
from ..services.processing_service import ProcessingService
from ..services.settings_service import SettingsService, get_settings_service as _get_settings_service
from ..events.bus import EventBus, get_event_bus


# Service singletons
_model_service: Optional[ModelService] = None
_processing_service: Optional[ProcessingService] = None


def get_model_service() -> ModelService:
    """Get or create ModelService instance."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


def get_processing_service() -> ProcessingService:
    """Get or create ProcessingService instance."""
    global _processing_service
    if _processing_service is None:
        _processing_service = ProcessingService()
    return _processing_service


def get_settings_service() -> SettingsService:
    """Get the shared SettingsService instance.

    Delegates to the service-layer singleton so every consumer observes the
    same cached settings. Keeping a second instance here meant updates made
    through the API were invisible to services (and vice versa) until restart.
    """
    return _get_settings_service()


# Dependency functions for FastAPI
async def get_current_settings() -> Dict[str, Any]:
    """Get current application settings."""
    settings = read_saved_settings(settings_file())
    return {**DEFAULT_SETTINGS, **settings}


async def get_model_registry() -> ModelRegistry:
    """Get model registry with fresh status."""
    registry = get_registry()
    registry.refresh_status()
    return registry


async def get_pipeline_registry() -> PipelineRegistry:
    """Get pipeline registry (FastAPI async dependency wrapper)."""
    return _get_pipeline_registry()


async def get_event_bus_dep() -> EventBus:
    """Get event bus."""
    return get_event_bus()


# Request validation dependencies
async def validate_model_exists(
    model_name: str,
    registry: ModelRegistry = Depends(get_model_registry)
) -> str:
    """Validate that a model exists in the catalog."""
    model = registry.get_model(model_name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    return model_name


async def validate_model_downloaded(
    model_name: str = Depends(validate_model_exists),
    registry: ModelRegistry = Depends(get_model_registry)
) -> str:
    """Validate that a model is downloaded."""
    if not registry.is_downloaded(model_name):
        raise HTTPException(
            status_code=400,
            detail=f"Model not downloaded: {model_name}. Please download it first."
        )
    return model_name


async def validate_feature_available(
    feature: str,
    pipeline_registry: PipelineRegistry = Depends(get_pipeline_registry)
) -> str:
    """Validate that a feature has registered pipelines."""
    # These validators are also invoked directly (not through FastAPI's Depends
    # machinery), in which case the argument still holds the unresolved
    # ``Depends(...)`` placeholder. Fall back to the singleton in that case.
    if not isinstance(pipeline_registry, PipelineRegistry):
        pipeline_registry = _get_pipeline_registry()
    if feature not in pipeline_registry.get_all_features():
        raise HTTPException(status_code=404, detail=f"Feature not available: {feature}")
    return feature


async def validate_pipeline_step(
    step: Dict[str, Any],
    pipeline_registry: PipelineRegistry = Depends(get_pipeline_registry),
    model_registry: ModelRegistry = Depends(get_model_registry)
) -> Dict[str, Any]:
    """Validate a pipeline step configuration."""
    # This helper is called directly from the processing routes rather than
    # being wired through FastAPI's dependency injection, so the defaults are
    # still ``Depends(...)`` placeholders here. Resolve them explicitly.
    if not isinstance(pipeline_registry, PipelineRegistry):
        pipeline_registry = _get_pipeline_registry()
    if not isinstance(model_registry, ModelRegistry):
        model_registry = get_registry()

    feature = step.get("feature")
    model = step.get("model")

    if not feature or not model:
        raise HTTPException(status_code=400, detail="Pipeline step must have 'feature' and 'model'")

    if feature not in pipeline_registry.get_all_features():
        raise HTTPException(status_code=400, detail=f"Unknown feature: {feature}")

    if not model_registry.get_model(model):
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    # Check if model provides this feature
    model_config = model_registry.get_model(model)
    if feature not in model_config.feature_names:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model} does not support feature {feature}"
        )

    return step


# Pagination
class PaginationParams(BaseModel):
    page: int = 1
    page_size: int = 50

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        return self.page_size


async def get_pagination(
    page: int = 1,
    page_size: int = 50
) -> PaginationParams:
    """Get pagination parameters."""
    return PaginationParams(page=max(1, page), page_size=min(100, max(1, page_size)))
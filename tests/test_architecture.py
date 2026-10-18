"""
Tests for the new openVideoAI architecture.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that all new modules can be imported."""
    from src.events.bus import EventBus, EventType, get_event_bus
    from src.models.registry import ModelRegistry, get_registry, ModelConfig, ModelStatus
    from src.processing.pipeline.base import (
        EnhancementPipeline,
        PipelineChain,
        PipelineConfig,
        ProcessingContext,
        ProcessingResult,
    )
    from src.processing.pipeline.registry import PipelineRegistry, get_pipeline_registry
    from src.api.main import create_app
    from src.api.schemas import (
        ModelResponse,
        ProcessRequest,
        PipelineStep,
        SettingsModel,
        TaskStatus,
    )
    from src.services.model_service import ModelService
    from src.services.processing_service import ProcessingService
    from src.services.settings_service import SettingsService

    print("All imports successful!")


def test_event_bus():
    """Test event bus functionality."""
    from src.events.bus import EventBus, EventType, ProcessingProgressEvent

    bus = EventBus()
    events = []

    def handler(event):
        events.append(event)

    bus.subscribe(EventType.PROCESSING_PROGRESS, handler)
    bus.publish(ProcessingProgressEvent(
        type=EventType.PROCESSING_PROGRESS,
        task_id="test",
        progress=50,
        message="Testing"
    ))

    assert len(events) == 1
    assert events[0].progress == 50
    assert events[0].task_id == "test"

    print("Event bus test passed!")


def test_model_registry():
    """Test model registry."""
    from src.models.registry import get_registry

    registry = get_registry()
    models = registry.get_all_models()

    # Should have loaded models from catalog
    assert len(models) > 0
    print(f"Loaded {len(models)} models from catalog")

    # Check a known model
    realesrgan = registry.get_model("Real-ESRGAN")
    assert realesrgan is not None
    assert "Upscale" in realesrgan.feature_names

    print("Model registry test passed!")


def test_pipeline_registry():
    """Test pipeline registry."""
    from src.processing.pipeline.registry import get_pipeline_registry

    registry = get_pipeline_registry()
    features = registry.get_all_features()

    # Should have registered pipelines
    assert len(features) > 0
    print(f"Registered features: {features}")

    # Check that Upscale is registered
    assert "Upscale" in features
    pipeline_class = registry.get_pipeline_class("Upscale")
    assert pipeline_class is not None

    print("Pipeline registry test passed!")


def test_api_schemas():
    """Test API schemas."""
    from src.api.schemas import ProcessRequest, PipelineStep, SettingsModel, TaskStatus

    # Test ProcessRequest
    request = ProcessRequest(
        input_path="test.mp4",
        steps=[
            PipelineStep(feature="Upscale", model="Real-ESRGAN", parameters={"scale": 4}),
        ],
    )
    assert request.input_path == "test.mp4"
    assert len(request.steps) == 1

    # Test SettingsModel
    settings = SettingsModel()
    assert settings.system.max_ram_usage == 85
    assert settings.gpu.compute_backend.value == "auto"

    print("API schemas test passed!")


def test_fastapi_app():
    """Test FastAPI app creation."""
    from src.api.main import create_app
    from src.core.config import __version__

    app = create_app()
    assert app.title == "openVideoAI"
    # The version has a single source of truth in src/core/config.py.
    assert app.version == __version__

    print("FastAPI app test passed!")


if __name__ == "__main__":
    test_imports()
    test_event_bus()
    test_model_registry()
    test_pipeline_registry()
    test_api_schemas()
    test_fastapi_app()
    print("\n✅ All tests passed!")
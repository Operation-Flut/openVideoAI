"""
Base processing pipeline classes.
Provides the foundation for all enhancement pipelines with unified interface,
progress reporting, device management, and error handling.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import logging
import time
from uuid import uuid4

from ...events.bus import EventBus, EventType, ProcessingProgressEvent, get_event_bus

logger = logging.getLogger(__name__)


@dataclass
class ProcessingContext:
    """Context passed through the processing pipeline."""
    input_path: Path
    output_path: Path
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    progress_callback: Optional[Callable[[float, str], None]] = None
    abort_flag: bool = False

    def report_progress(self, progress: float, message: str = "") -> None:
        """Report progress via callback and event bus."""
        if self.progress_callback:
            self.progress_callback(progress, message)
        # Also publish to event bus for SSE/WebSocket
        event_bus = get_event_bus()
        event = ProcessingProgressEvent(
            type=EventType.PROCESSING_PROGRESS,
            task_id=self.metadata.get("task_id", "unknown"),
            progress=progress,
            message=message,
        )
        event_bus.publish(event)


@dataclass
class ProcessingResult:
    """Result of a processing operation."""
    success: bool
    input_path: Path
    output_path: Optional[Path] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    model_used: Optional[str] = None
    device_used: Optional[str] = None


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ModelLoadError(PipelineError):
    """Failed to load model."""
    pass


class InferenceError(PipelineError):
    """Inference failed."""
    pass


class UnsupportedFormatError(PipelineError):
    """Unsupported file format."""
    pass


class ProcessingAborted(PipelineError):
    """Processing was aborted (e.g., shutdown)."""
    pass


@dataclass
class PipelineConfig:
    """Configuration for a pipeline."""
    name: str
    feature: str  # e.g., "Upscale", "Face Recovery"
    description: str
    supported_formats: List[str] = field(default_factory=list)
    supports_video: bool = False
    supports_image: bool = True
    requires_gpu: bool = False
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    parameter_schema: Dict[str, Any] = field(default_factory=dict)  # JSON Schema


class EnhancementPipeline(ABC):
    """
    Abstract base class for all enhancement pipelines.

    Each pipeline implements a single enhancement feature (upscale, face recovery, etc.)
    and can be chained together in a PipelineChain.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        parameters: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device
        self.parameters = parameters or {}
        self.task_id = task_id or str(uuid4())
        self._model = None
        self._loaded = False
        self._config = self._get_config()

    @property
    @abstractmethod
    def config(self) -> PipelineConfig:
        """Return pipeline configuration."""
        pass

    def _get_config(self) -> PipelineConfig:
        return self.config

    @abstractmethod
    def load_model(self, model_path: Union[str, Path, Dict[str, Path]]) -> bool:
        """
        Load the model from disk.

        Args:
            model_path: Path to model file(s). Can be a single path or dict for multi-file models.

        Returns:
            True if loaded successfully
        """
        pass

    @abstractmethod
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """
        Process the input file.

        Args:
            context: Processing context with input/output paths and parameters

        Returns:
            ProcessingResult with success status and metadata
        """
        pass

    def preprocess(self, context: ProcessingContext) -> ProcessingContext:
        """Pre-process input before inference. Override if needed."""
        return context

    def postprocess(self, context: ProcessingContext, result: ProcessingResult) -> ProcessingResult:
        """Post-process output after inference. Override if needed."""
        return result

    def validate_input(self, context: ProcessingContext) -> bool:
        """Validate input file format and existence."""
        if not context.input_path.exists():
            raise UnsupportedFormatError(f"Input file not found: {context.input_path}")

        suffix = context.input_path.suffix.lower()
        if self.config.supported_formats and suffix not in self.config.supported_formats:
            raise UnsupportedFormatError(
                f"Unsupported format: {suffix}. Supported: {self.config.supported_formats}"
            )
        return True

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        """Unload model to free memory."""
        self._model = None
        self._loaded = False

    def get_device(self) -> str:
        """Resolve device string."""
        from ...utils.system_resources import resolve_default_device
        return resolve_default_device(self.device)

    def run(self, context: ProcessingContext) -> ProcessingResult:
        """
        Run the full pipeline: validate -> preprocess -> load -> process -> postprocess.
        """
        start_time = time.time()

        try:
            # Validate
            self.validate_input(context)

            # Ensure model is loaded
            if not self._loaded:
                model_paths = self._resolve_model_paths()
                if not model_paths:
                    raise ModelLoadError(f"Model not found: {self.model_name}")
                self.load_model(model_paths)

            # Preprocess
            context = self.preprocess(context)

            # Check abort flag
            if context.abort_flag:
                raise ProcessingAborted("Processing aborted before inference")

            # Process
            result = self.process(context)

            # Postprocess
            result = self.postprocess(context, result)

            result.processing_time = time.time() - start_time
            result.model_used = self.model_name
            result.device_used = self.get_device()

            return result

        except ProcessingAborted:
            raise
        except PipelineError:
            raise
        except Exception as e:
            logger.exception(f"Pipeline {self.model_name} failed: {e}")
            raise InferenceError(f"Processing failed: {e}")

    def _resolve_model_paths(self) -> Optional[Union[str, Path, Dict[str, Path]]]:
        """Resolve model paths from registry."""
        from ...models.registry import get_registry
        registry = get_registry()

        if registry.is_downloaded(self.model_name):
            paths = registry.get_all_model_paths(self.model_name)
            if paths:
                return paths if len(paths) > 1 else paths.get("main")

        # Try to find by feature
        feature = self.config.feature
        best = registry.get_best_model_for_feature(feature, device_type=self.device)
        if best:
            paths = registry.get_all_model_paths(best.config.name)
            if paths:
                return paths if len(paths) > 1 else paths.get("main")

        return None


class PipelineChain:
    """
    Chains multiple pipelines together for multi-stage processing.

    Example:
        chain = PipelineChain([
            UpscalePipeline("Real-ESRGAN", scale=4),
            FaceRecoveryPipeline("CodeFormer"),
            DenoisePipeline("SCUNet"),
        ])
        result = chain.run(context)
    """

    def __init__(self, pipelines: List[EnhancementPipeline]):
        self.pipelines = pipelines
        self.task_id = str(uuid4())

    def run(self, context: ProcessingContext) -> ProcessingResult:
        """Run all pipelines in sequence, passing output of one as input to next."""
        context.metadata["task_id"] = self.task_id

        current_input = context.input_path
        final_result = None

        for i, pipeline in enumerate(self.pipelines):
            # Check abort
            if context.abort_flag:
                raise ProcessingAborted(f"Aborted at pipeline {i}: {pipeline.model_name}")

            # Report pipeline start
            context.report_progress(
                (i / len(self.pipelines)) * 100,
                f"Starting {pipeline.config.feature} ({pipeline.model_name})"
            )

            # Create sub-context for this pipeline
            pipeline_context = ProcessingContext(
                input_path=current_input,
                output_path=context.output_path if i == len(self.pipelines) - 1
                else context.output_path.parent / f"{context.output_path.stem}_stage{i}{context.output_path.suffix}",
                parameters={**context.parameters, **pipeline.parameters},
                metadata=context.metadata,
                progress_callback=lambda p, m: context.report_progress(
                    (i / len(self.pipelines)) * 100 + (p / 100) * (100 / len(self.pipelines)),
                    m
                ),
                abort_flag=context.abort_flag,
            )

            try:
                result = pipeline.run(pipeline_context)
            except ProcessingAborted:
                raise
            except Exception as e:
                logger.exception(f"Pipeline {pipeline.model_name} failed: {e}")
                return ProcessingResult(
                    success=False,
                    input_path=context.input_path,
                    output_path=None,
                    error=f"{pipeline.config.feature} failed: {e}",
                    metadata={"failed_pipeline": pipeline.model_name},
                )

            if not result.success:
                return ProcessingResult(
                    success=False,
                    input_path=context.input_path,
                    output_path=None,
                    error=result.error,
                    metadata={"failed_pipeline": pipeline.model_name},
                )

            # Next pipeline uses this output as input
            current_input = result.output_path
            final_result = result

        if final_result:
            final_result.processing_time = sum(p.processing_time for p in self.pipelines if hasattr(p, 'processing_time'))
            final_result.metadata["pipelines"] = [p.model_name for p in self.pipelines]

        return final_result or ProcessingResult(
            success=False,
            input_path=context.input_path,
            output_path=None,
            error="No pipelines executed",
        )

    def add_pipeline(self, pipeline: EnhancementPipeline) -> "PipelineChain":
        """Add a pipeline to the chain (fluent interface)."""
        self.pipelines.append(pipeline)
        return self
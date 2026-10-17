"""
Processing Service - Orchestrates enhancement pipelines.
"""
import asyncio
import logging
import shutil
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

from ..processing.pipeline.base import (
    EnhancementPipeline,
    PipelineChain,
    PipelineConfig,
    ProcessingContext,
    ProcessingResult,
    PipelineError,
    ModelLoadError,
    InferenceError,
    UnsupportedFormatError,
    ProcessingAborted,
)
from ..processing.pipeline.registry import get_pipeline_registry, instantiate_pipeline
from ..models.registry import get_registry
from ..services.settings_service import configured_compute_device
from ..core.paths import temp_processing_dir, exports_dir
from ..events.bus import EventBus, EventType, ProcessingProgressEvent, get_event_bus
from ..api.schemas import ProcessRequest, PipelineStep, TaskStatus, ProcessingTask, ProcessResponse, TaskStatusResponse

logger = logging.getLogger(__name__)

# Suffixes that make a file a video (an image is written by a pipeline itself,
# a video goes through the encoder).
_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v", ".ts", ".mpg", ".mpeg", ".wmv"}


class ProcessingService:
    """Service for managing and executing processing tasks."""

    def __init__(self):
        self._pipeline_registry = get_pipeline_registry()
        self._model_registry = get_registry()
        self._event_bus = get_event_bus()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._tasks = {}
        self._active_task = None
        self._shutdown_requested = False
        self._tasks_lock = threading.RLock()

    def create_task(self, request: ProcessRequest) -> ProcessingTask:
        """Create a new processing task."""
        task_id = str(uuid.uuid4())

        # Resolve the destination up front: an unwritable folder is reported at
        # request time instead of minutes into a render.
        if not request.output_path:
            try:
                self.resolve_output_path(request, Path(request.input_path), task_id)
            except Exception as error:  # pragma: no cover - reported when writing
                logger.warning("Could not resolve the output path yet: %s", error)

        task = ProcessingTask(
            task_id=task_id,
            status=TaskStatus.PENDING,
            progress=0.0,
            message="Task created",
        )
        with self._tasks_lock:
            self._tasks[task_id] = task
        return task

    def _get_output_suffix(self, output_format: str, input_suffix: str) -> str:
        """Get output file suffix based on format."""
        if output_format == "source":
            return input_suffix
        format_map = {
            "jpg": ".jpg",
            "png": ".png",
            "webp": ".webp",
            "tiff": ".tiff",
            "mp4": ".mp4",
        }
        return format_map.get(output_format, input_suffix)

    # -- Export dialog ----------------------------------------------------
    def export_options(self, request: ProcessRequest, input_path: Path) -> Tuple[Dict[str, Any], List[str]]:
        """The dialog's choices, normalised against this machine's encoders.

        Returns the option set the encoder will use plus the corrections that
        had to be made (an unavailable codec, a LUT that is not there, audio the
        container cannot carry), so the result can explain itself.
        """
        from .export_service import normalise_options
        from .settings_service import get_settings_service

        settings = get_settings_service().get_settings()
        raw = request.export_options.model_dump() if request.export_options else None
        return normalise_options(raw, settings, input_path=input_path, allow_sequence=not request.preview)

    def _requested_scale(self, request: ProcessRequest) -> int:
        """The upscale factor the request asks for, for the naming pattern."""
        for step in request.steps:
            if step.feature == "Upscale":
                try:
                    scale = int(step.parameters.get("scale"))
                except (TypeError, ValueError):
                    continue
                if scale > 0:
                    return scale
        from .settings_service import get_settings_service

        try:
            return int(get_settings_service().get_settings().export.default_upscale)
        except Exception:  # pragma: no cover - settings are always readable
            return 1

    def resolve_output_path(self, request: ProcessRequest, input_path: Path, task_id: str) -> Path:
        """Where this request writes, honouring the Export dialog's choices."""
        from .export_service import resolve_output_path as resolve
        from .settings_service import get_settings_service

        if request.output_path:
            return Path(request.output_path)

        if request.preview:
            # Previews are throwaway samples the UI plays; keep them in temp.
            options, _ = self.export_options(request, input_path)
            if input_path.suffix.lower() in _VIDEO_SUFFIXES:
                suffix = f".{options.get('container') or 'mp4'}"
            else:
                suffix = self._get_output_suffix(request.output_format.value, input_path.suffix)
            return temp_processing_dir() / f"{task_id}_preview{suffix}"

        settings = get_settings_service().get_settings()
        options, _ = self.export_options(request, input_path)
        return resolve(
            input_path,
            options,
            naming_pattern=settings.export.export_naming,
            scale=self._requested_scale(request),
            overwrite=settings.export.overwrite_existing,
        )

    async def start_processing(self, task_id: str, request: ProcessRequest) -> ProcessResponse:
        """Start processing a task."""
        with self._tasks_lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task not found: {task_id}")

            if self._active_task is not None:
                raise ValueError("Another task is already processing")

            task = self._tasks[task_id]
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now()
            task.message = "Starting processing"
            self._active_task = task_id


        loop = asyncio.get_event_loop()
        asyncio.create_task(self._run_processing(task_id, request, loop))

        return ProcessResponse(
            task_id=task_id,
            status=TaskStatus.PROCESSING,
            message="Processing started",
        )

    # Input extensions treated as video / image when building a preview sample.
    _VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v", ".ts"}
    _IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    # Longest edge a preview image is downscaled to.
    _PREVIEW_MAX_DIMENSION = 1280

    def _prepare_preview_input(
        self, temp_input: Path, temp_dir: Path, task_id: str, duration: float
    ) -> Optional[Path]:
        """Reduce an input to a small sample so a preview finishes quickly.

        Videos are trimmed to the first ``duration`` seconds; images are
        downscaled to :attr:`_PREVIEW_MAX_DIMENSION`. Returns ``None`` when the
        input is already small enough or cannot be converted, in which case the
        pipeline runs on the original copy.
        """
        suffix = temp_input.suffix.lower()

        if suffix in self._VIDEO_SUFFIXES:
            out = temp_dir / f"{task_id}_preview_input.mp4"
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-t", str(max(0.1, float(duration))),  # input option: read only this much
                "-i", str(temp_input),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-pix_fmt", "yuv420p", "-an",
                str(out),
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0 and out.exists() and out.stat().st_size > 0:
                    logger.info("Preview: trimmed %s to %.1fs", temp_input.name, duration)
                    return out
                logger.warning(
                    "Preview trim failed for %s: %s", temp_input.name, (result.stderr or "")[-300:]
                )
            except Exception as e:
                logger.warning("Preview trim error for %s: %s", temp_input.name, e)
            return None

        if suffix in self._IMAGE_SUFFIXES:
            try:
                from PIL import Image

                with Image.open(temp_input) as img:
                    if max(img.size) <= self._PREVIEW_MAX_DIMENSION:
                        return None
                    scale = self._PREVIEW_MAX_DIMENSION / max(img.size)
                    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
                    resized = img.convert("RGB").resize(size, Image.LANCZOS)
                    out = temp_dir / f"{task_id}_preview_input{suffix}"
                    resized.save(out)
                logger.info("Preview: downscaled %s to %s", temp_input.name, size)
                return out
            except Exception as e:
                logger.warning("Preview downscale error for %s: %s", temp_input.name, e)
            return None

        return None

    async def _run_processing(self, task_id: str, request: ProcessRequest, loop: asyncio.AbstractEventLoop):
        """Run the processing pipeline."""
        task = self._tasks[task_id]
        temp_dir = temp_processing_dir()
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            input_path = Path(request.input_path)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")


            temp_input = temp_dir / f"{task_id}_input{input_path.suffix}"
            shutil.copy2(input_path, temp_input)


            if request.output_path:
                output_path = Path(request.output_path)
            else:
                # Previews are kept in temp, exports go to the folder the dialog
                # asked for (original folder / exports / a chosen folder).
                output_path = self.resolve_output_path(request, input_path, task_id)

            temp_output = temp_dir / f"{task_id}_output{output_path.suffix}"

            # Preview: run the pipeline on a short/small sample of the input.
            if request.preview:
                sample = self._prepare_preview_input(
                    temp_input, temp_dir, task_id, request.preview_duration
                )
                if sample is not None:
                    temp_input = sample

            # The dialog's choices, resolved into the values the encoder uses.
            # ``audio_source`` points at the copied input because the pipeline
            # intermediates carry no audio track.
            export_options, export_warnings = self.export_options(request, input_path)
            export_parameters: Dict[str, Any] = dict(export_options)
            export_parameters["audio_source"] = str(temp_input)

            # Build pipeline chain from steps
            if request.steps:
                # New pipeline chain format
                chain = self._build_chain_from_steps(request.steps, task_id)
            else:
                logger.warning("Legacy parameter format is deprecated; use 'steps' format instead")
                chain = self._build_chain_from_legacy(request, task_id)

            if not chain or not chain.pipelines:
                raise ValueError("No valid pipelines configured")

            # Create processing context
            context = ProcessingContext(
                input_path=temp_input,
                output_path=temp_output,
                parameters={
                    "output_format": request.output_format.value,
                    "quality": request.quality,
                    "export": export_parameters,
                    **self._extract_legacy_params(request),
                },
                metadata={"task_id": task_id},
                progress_callback=lambda p, m: self._update_progress(task_id, p, m),
            )


            event = ProcessingProgressEvent(
                type=EventType.PROCESSING_STARTED,
                task_id=task_id,
                progress=0,
                message="Processing started",
            )
            self._event_bus.publish(event)


            result = await loop.run_in_executor(self._executor, chain.run, context)

            if result.success:
                # A pipeline may write a different path/suffix than requested
                # (e.g. HDR images are written as .tiff), so honour the path it
                # reports instead of assuming the requested temp file exists.
                produced = Path(result.output_path) if result.output_path else temp_output
                if not produced.exists():
                    produced = temp_output

                is_sequence = export_options.get("output_type") == "image_sequence" and produced.is_dir()

                if is_sequence:
                    # Frames are written into a folder next to the planned file
                    # (``<name>.png`` -> folder ``<name>``).
                    output_path = output_path.with_suffix("")
                    if output_path.exists():
                        if output_path.is_dir():
                            shutil.rmtree(output_path)
                        else:
                            output_path.unlink()
                elif produced.suffix and produced.suffix.lower() != output_path.suffix.lower():
                    output_path = output_path.with_suffix(produced.suffix)

                output_path.parent.mkdir(parents=True, exist_ok=True)
                if is_sequence and output_path.exists():
                    shutil.rmtree(output_path)
                shutil.move(str(produced), str(output_path))

                task.status = TaskStatus.COMPLETED
                task.progress = 100.0
                task.message = "Processing completed"
                task.completed_at = datetime.now()
                task.result = self._export_metadata(result.metadata, context, export_options, export_warnings)
                task.output_path = str(output_path)

                event = ProcessingProgressEvent(
                    type=EventType.PROCESSING_COMPLETED,
                    task_id=task_id,
                    progress=100,
                    message="Completed",
                )
            else:
                task.status = TaskStatus.FAILED
                task.message = result.error or "Processing failed"
                task.error = result.error
                task.completed_at = datetime.now()

                event = ProcessingProgressEvent(
                    type=EventType.PROCESSING_FAILED,
                    task_id=task_id,
                    progress=0,
                    message=task.message,
                )

            self._event_bus.publish(event)

        except ProcessingAborted:
            task.status = TaskStatus.ABORTED
            task.message = "Processing aborted"
            task.completed_at = datetime.now()

            event = ProcessingProgressEvent(
                type=EventType.PROCESSING_ABORTED,
                task_id=task_id,
                progress=0,
                message="Aborted",
            )
            self._event_bus.publish(event)

        except Exception as e:
            logger.exception(f"Task {task_id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.message = f"Processing failed: {e}"
            task.error = str(e)
            task.completed_at = datetime.now()

            event = ProcessingProgressEvent(
                type=EventType.PROCESSING_FAILED,
                task_id=task_id,
                progress=0,
                message=str(e),
            )
            self._event_bus.publish(event)


        finally:
            try:
                temp_input_path = temp_dir / f"{task_id}_input{input_path.suffix}"
                temp_output_path = temp_dir / f"{task_id}_output{output_path.suffix}"
                # Image sequences are written into a suffix-less temp folder.
                sequence_dir = temp_dir / f"{task_id}_output"
                for path in (temp_input_path, temp_output_path, sequence_dir):
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    elif path.exists():
                        path.unlink(missing_ok=True)
            except Exception:
                pass

            with self._tasks_lock:
                self._active_task = None

    def _export_metadata(
        self,
        result_metadata: Optional[Dict[str, Any]],
        context: ProcessingContext,
        options: Dict[str, Any],
        warnings: List[str],
    ) -> Dict[str, Any]:
        """What a pipeline reported, plus what the export encoder actually did."""
        from .export_service import summarise

        metadata: Dict[str, Any] = dict(result_metadata or {})
        if options:
            metadata["export"] = {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in options.items()
            }
            metadata["export_summary"] = summarise(options)
            metadata["output_type"] = options.get("output_type")
        collected = list(warnings or []) + list(context.metadata.get("export_warnings") or [])
        if collected:
            metadata["export_warnings"] = list(dict.fromkeys(collected))
        if context.metadata.get("export_encoder"):
            metadata["encoder"] = context.metadata["export_encoder"]
        return metadata

    def _build_chain_from_steps(self, steps: List[PipelineStep], task_id: str) -> Optional[PipelineChain]:
        """Build pipeline chain from step configurations."""
        from .settings_service import experimental_features_enabled

        pipelines = []
        # Reasons a step was dropped, surfaced to the caller so a failed run
        # explains itself instead of ending in "No valid pipelines configured".
        problems: List[str] = []
        include_experimental = experimental_features_enabled()

        for i, step in enumerate(steps):
            if not include_experimental and self._model_registry.is_feature_experimental(step.feature):
                problems.append(
                    f"'{step.feature}' is experimental "
                    "(enable it in Settings -> Advanced)"
                )
                logger.warning(
                    "Skipping experimental feature '%s' (enable it in Settings -> Advanced)",
                    step.feature,
                )
                continue

            model_name = step.model
            if not include_experimental and self._model_registry.is_model_experimental(model_name):
                alternatives = [
                    name for name in self._model_registry.get_available_model_names(step.feature)
                    if not self._model_registry.is_model_experimental(name)
                ]
                if not alternatives:
                    problems.append(
                        f"'{model_name}' is experimental and '{step.feature}' has no "
                        "non-experimental alternative"
                    )
                    logger.warning(
                        "Skipping experimental model '%s' for '%s' (no alternative available)",
                        model_name, step.feature,
                    )
                    continue
                logger.info(
                    "Substituting non-experimental model '%s' for '%s'",
                    alternatives[0], model_name,
                )
                model_name = alternatives[0]

            pipeline_class = self._pipeline_registry.get_pipeline_class(step.feature, model_name)
            if not pipeline_class:
                problems.append(f"no pipeline implements '{step.feature}'")
                logger.error(f"No pipeline for feature: {step.feature}")
                continue

            # Create pipeline instance
            pipeline = instantiate_pipeline(pipeline_class,
                model_name=model_name,
                # An explicit per-step device wins; otherwise honour the backend
                # chosen in Settings (Settings -> GPU -> Compute Backend).
                device=step.parameters.get("device") or configured_compute_device(),
                parameters=step.parameters,
                task_id=f"{task_id}_step{i}",
            )

            # Load model
            model_paths = self._model_registry.get_all_model_paths(model_name)
            if not model_paths:
                problems.append(f"model '{model_name}' is not installed")
                logger.error(f"Model not found: {model_name}")
                continue

            try:
                pipeline.load_model(model_paths if len(model_paths) > 1 else model_paths.get("main"))
            except Exception as e:
                problems.append(f"model '{model_name}' failed to load: {e}")
                logger.error(f"Failed to load model {model_name}: {e}")
                continue

            pipelines.append(pipeline)

        if not pipelines:
            if problems:
                raise PipelineError("; ".join(dict.fromkeys(problems)))
            return None

        return PipelineChain(pipelines)

    def _build_chain_from_legacy(self, request: ProcessRequest, task_id: str) -> Optional[PipelineChain]:
        """Build pipeline chain from legacy parameters.

        DEPRECATED: Use `steps` format instead. This method will be removed in a future version.
        """
        import warnings
        warnings.warn(
            "Legacy parameter format is deprecated. Use `steps` format instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        pipelines = []


        feature_params = [
            ("Upscale", request.upscale),
            ("Face Recovery", request.face_recovery),
            ("Sharpen / Deblur", request.sharpen),
            ("Denoise", request.denoise),
            ("Frame Interpolation", request.frame_interpolation),
            ("Match Color", request.match_color),
            ("Low-Light Enhancement", request.low_light),
            ("SDR to HDR", request.sdr_to_hdr),
            ("Reencoding", request.reencoding),
        ]

        for i, (feature, params) in enumerate(feature_params):
            if params is None:
                continue

            # Resolve model for this feature
            model_name = params.model if hasattr(params, "model") else params.get("model")
            if not model_name:
                # Fall back to the feature's default model
                model_name = self._pipeline_registry.get_default_model(feature)
            if not model_name:
                continue

            pipeline_class = self._pipeline_registry.get_pipeline_class(feature, model_name)
            if not pipeline_class:
                continue


            param_dict = params.model_dump() if hasattr(params, "model_dump") else dict(params)
            param_dict.pop("model", None)
            param_dict.pop("device", None)

            pipeline = instantiate_pipeline(pipeline_class,
                model_name=model_name,
                device=(
                    params.device if hasattr(params, "device") else params.get("device")
                ) or configured_compute_device(),
                parameters=param_dict,
                task_id=f"{task_id}_step{i}",
            )

            model_paths = self._model_registry.get_all_model_paths(model_name)
            if not model_paths:
                logger.error(f"Model not found: {model_name}")
                continue

            try:
                pipeline.load_model(model_paths if len(model_paths) > 1 else model_paths.get("main"))
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                continue

            pipelines.append(pipeline)

        if not pipelines:
            return None

        return PipelineChain(pipelines)

    def _extract_legacy_params(self, request: ProcessRequest) -> Dict[str, Any]:
        """Extract parameters from legacy request format."""
        params = {}
        for attr in ("upscale", "face_recovery", "sharpen", "denoise", "frame_interpolation",
                     "match_color", "low_light", "sdr_to_hdr", "reencoding"):
            value = getattr(request, attr, None)
            if value:
                if hasattr(value, "model_dump"):
                    params[attr] = value.model_dump()
                else:
                    params[attr] = dict(value)
        return params

    def _update_progress(self, task_id: str, progress: float, message: str = ""):
        """Update task progress."""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.progress = min(100.0, max(0.0, progress))
            if message:
                task.message = message

    def get_task_status(self, task_id: str) -> Optional[TaskStatusResponse]:
        """Get task status."""
        with self._tasks_lock:
            if task_id not in self._tasks:
                return None

            task = self._tasks[task_id]
            return TaskStatusResponse(
                task_id=task.task_id,
                status=task.status,
                progress=task.progress,
                message=task.message,
                current_step=None,
                error=task.error,
                result=task.result,
                output_path=task.output_path,
            )

    def get_task(self, task_id: str) -> Optional[ProcessingTask]:
        """Get full task object."""
        with self._tasks_lock:
            return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        with self._tasks_lock:
            if task_id == self._active_task:
                self._shutdown_requested = True
                return True
            return False

    def request_shutdown(self):
        """Request shutdown of all processing."""
        self._shutdown_requested = True

    def is_shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def has_active_processing(self) -> bool:
        return self._active_task is not None and self._tasks.get(self._active_task, {}).status == TaskStatus.PROCESSING

    def get_all_tasks(self) -> List[TaskStatusResponse]:
        """Get all tasks."""
        with self._tasks_lock:
            return [self.get_task_status(tid) for tid in self._tasks.keys()]

    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Remove old completed tasks."""
        now = datetime.now()
        to_remove = []
        with self._tasks_lock:
            for task_id, task in self._tasks.items():
                if task.completed_at:
                    age = (now - task.completed_at).total_seconds() / 3600
                    if age > max_age_hours:
                        to_remove.append(task_id)
            for task_id in to_remove:
                del self._tasks[task_id]

    def shutdown(self):
        """Shutdown the service."""
        self._shutdown_requested = True
        self._executor.shutdown(wait=True)


_processing_service_instance: Optional[ProcessingService] = None


def get_processing_service() -> ProcessingService:
    """Get or create ProcessingService instance."""
    global _processing_service_instance
    if _processing_service_instance is None:
        _processing_service_instance = ProcessingService()
    return _processing_service_instance

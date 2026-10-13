"""
Processing API endpoints for openVideoAI
Handles all enhancement pipelines: upscale, face recovery, sharpen, denoise,
frame interpolation, slow motion, color matching, and video transformations.

Every endpoint follows the same contract as the legacy monolithic api.py:
uploads are staged in the temp processing dir, work runs in a thread-pool
executor, and progress/results are published to ``process_progress`` keyed by
task_id for polling via /api/processing/task/{task_id}.
"""
from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src import system_ressources as system_resources
from src.api.state import (
    TEMP_PROCESSING_DIR,
    is_shutdown_requested,
    process_progress,
    process_tasks,
    read_saved_settings,
)
from src.core.config import DEFAULT_SETTINGS
from src.processing.base import ProcessingAborted
from src.schemas import UpscaleRequest

router = APIRouter()

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional at import time
    torch = None

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is optional at import time
    cv2 = None


# ---------------------------------------------------------------------------
# Lazy processing-module loading
# ---------------------------------------------------------------------------
# The distutils monkey patch must be installed before torch/BasicSR are
# imported, so the heavy processing modules are resolved on first use.

PROCESSING_AVAILABLE = False
_upscale_image: Optional[Callable[..., Dict[str, Any]]] = None
_upscale_video: Optional[Callable[..., Dict[str, Any]]] = None
_get_upscale_model: Optional[Callable[[str], Optional[str]]] = None
_list_available_upscale_models: Optional[Callable[[], List[Dict[str, Any]]]] = None


def _load_processing() -> None:
    """Lazy load processing modules after the distutils monkey patch."""
    global PROCESSING_AVAILABLE
    global _upscale_image, _upscale_video
    global _get_upscale_model, _list_available_upscale_models

    if PROCESSING_AVAILABLE:
        return
    try:
        from src.processing.upscale import (
            get_upscale_model,
            list_available_upscale_models,
            upscale_image,
            upscale_video,
        )

        _upscale_image = upscale_image
        _upscale_video = upscale_video
        _get_upscale_model = get_upscale_model
        _list_available_upscale_models = list_available_upscale_models
        PROCESSING_AVAILABLE = True
    except ImportError as exc:
        print(f"Processing modules not available: {exc}")
        PROCESSING_AVAILABLE = False


def _require_processing() -> None:
    """Raise 503 when the processing stack could not be imported."""
    _load_processing()
    if not PROCESSING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Processing not available")


def _require_running() -> None:
    """Reject new work once the desktop window has been closed."""
    if is_shutdown_requested():
        raise HTTPException(status_code=503, detail="Server is shutting down")


def _effective_settings() -> Dict[str, Any]:
    return {**DEFAULT_SETTINGS, **read_saved_settings()}


def _resolve_device(device: Optional[str], saved_settings: Dict[str, Any]) -> str:
    """Map the requested/saved backend onto a real torch device string.

    Legacy or mis-configured values such as ``gpu:0`` (an OS adapter name, not
    a torch device) are resolved to the best available accelerator instead of
    crashing model loading.
    """
    raw = device or saved_settings.get("computeBackend", "auto")
    resolved = system_resources.resolve_default_device(str(raw))
    if resolved != raw:
        print(f"[api] Device '{raw}' is not a usable torch device; using '{resolved}'")
    return resolved


# ---------------------------------------------------------------------------
# Upload staging + task dispatch
# ---------------------------------------------------------------------------

async def _stage_upload(file: UploadFile, prefix: str = "") -> Path:
    """Persist an uploaded file into the temp processing dir."""
    TEMP_PROCESSING_DIR.mkdir(exist_ok=True)
    safe_filename = Path(file.filename or "upload").name
    input_path = TEMP_PROCESSING_DIR / f"{uuid.uuid4()}_{prefix}{safe_filename}"
    content = await file.read()
    with open(input_path, "wb") as handle:
        handle.write(content)
    return input_path


def _dispatch(run: Callable[[str], Dict[str, Any]], cleanup: List[Optional[Path]]) -> Dict[str, Any]:
    """Register a task, run it in the executor, and publish its lifecycle state.

    ``run`` receives the task_id so it can report incremental progress.
    Status transitions match the legacy implementation exactly:
    processing -> completed | aborted | failed.
    """
    task_id = str(uuid.uuid4())
    process_progress[task_id] = {"status": "processing", "progress": 0}

    def worker() -> None:
        try:
            result = run(task_id)
            process_progress[task_id] = {
                "status": "completed",
                "result": result,
                "progress": 100,
            }
        except ProcessingAborted as exc:
            print(f"[Worker] {exc}", flush=True)
            process_progress[task_id] = {
                "status": "aborted",
                "error": str(exc),
                "progress": 0,
            }
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a task error
            import traceback

            traceback.print_exc()
            process_progress[task_id] = {
                "status": "failed",
                "error": str(exc),
                "progress": 0,
            }
        finally:
            for path in cleanup:
                try:
                    if path is not None:
                        path.unlink()
                except OSError:
                    pass

    loop = asyncio.get_event_loop()
    process_tasks[task_id] = loop.run_in_executor(None, worker)
    return {"task_id": task_id, "status": "started"}


def _apply_torch_runtime_settings(saved_settings: Dict[str, Any], selected_device: str) -> None:
    """Honour CPU thread count and the VRAM budget before loading a model."""
    if torch is None:
        return
    torch.set_num_threads(max(1, int(saved_settings.get("cpuThreads", 8))))
    if selected_device.startswith(("cuda", "hip")) and saved_settings.get("maxVramUsage", 0):
        device_index = (
            int(selected_device.split(":", 1)[1]) if ":" in selected_device else 0
        )
        # maxVramUsage is a percentage (0-100) of total device VRAM
        fraction = min(
            max(float(saved_settings.get("maxVramUsage", 95)) / 100.0, 0.0), 1.0
        )
        torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)


def _patch_distutils_in_worker() -> None:
    """Re-apply the distutils shim inside the worker thread (Python 3.12+)."""
    import sys

    try:
        from setuptools import _distutils

        class _DistutilsModule(ModuleType):
            def __getattr__(self, name):
                if hasattr(_distutils, name):
                    return getattr(_distutils, name)
                try:
                    return __import__(
                        "setuptools._distutils." + name, fromlist=[""]
                    )
                except ImportError:
                    pass
                raise AttributeError("module distutils has no attribute " + name)

        sys.modules["distutils"] = _DistutilsModule("distutils")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Feature discovery
# ---------------------------------------------------------------------------

@router.get("/processing/features")
async def get_processing_features():
    """Get all available processing features with their models."""
    from src.processing import (
        DENOISE_MODELS,
        FACE_MODELS,
        FRAME_INTERP_MODELS,
        REENCODING_PRESETS,
        SHARPEN_MODELS,
        SLOWMO_MODELS,
        UPSCALE_MODELS,
    )

    return {
        "upscale": {"models": list(UPSCALE_MODELS.keys())},
        "face_recovery": {"models": list(FACE_MODELS.keys())},
        "sharpen": {"models": list(SHARPEN_MODELS.keys())},
        "denoise": {"models": list(DENOISE_MODELS.keys())},
        "frame_interpolation": {"models": list(FRAME_INTERP_MODELS.keys())},
        "slow_motion": {"models": list(SLOWMO_MODELS.keys())},
        "colorization": {"models": ["DeOldify"]},
        "match_color": {
            "models": ["Reinhard Transfer", "WCT2", "NLUT", "Deep Color Transfer"]
        },
        "low_light": {"models": ["Zero-DCE"]},
        "distortion_correction": {"models": ["LensCorrection"]},
        "log_restoration": {"models": ["ITMLUT", "FastLLVE"]},
        "sdr_to_hdr": {"models": ["HDRNet"]},
        "object_removal": {"models": ["LaMa"]},
        "focus_fix": {"models": ["DeblurGAN"]},
        "overexposure_fix": {"models": ["AutoEnhance"]},
        "flat_color_profile": {"models": ["ColorGrade"]},
        "reencoding": {
            "engines": ["ffmpeg", "handbrake"],
            "ffmpeg_presets": list(REENCODING_PRESETS.keys()),
        },
    }


@router.get("/processing/upscale/models")
async def get_upscale_models():
    """List available upscale models."""
    _require_processing()
    assert _list_available_upscale_models is not None
    return _list_available_upscale_models()


@router.get("/processing/task/{task_id}")
async def get_processing_status(task_id: str):
    """Get processing task status."""
    if task_id not in process_progress:
        raise HTTPException(status_code=404, detail="Task not found")
    return process_progress[task_id]


@router.get("/processing/result/{task_id}")
async def get_processing_result(task_id: str):
    """Get processing result file."""
    if task_id not in process_progress:
        raise HTTPException(status_code=404, detail="Task not found")

    progress = process_progress[task_id]
    if progress.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed")

    output_path = progress.get("result", {}).get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(output_path, filename=Path(output_path).name)


# ---------------------------------------------------------------------------
# Upscale
# ---------------------------------------------------------------------------

ALLOWED_IMAGE_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff",
}
ALLOWED_VIDEO_TYPES = {
    "video/mp4", "video/mov", "video/quicktime", "video/avi", "video/mkv", "video/webm",
}
VALID_OUTPUT_FORMATS = {"source", "jpg", "png", "webp", "tiff", "mp4"}
IMAGE_SUFFIX_BY_FORMAT = {"jpg": ".jpg", "png": ".png", "webp": ".webp", "tiff": ".tiff"}


@router.post("/processing/upscale/image")
async def process_upscale_image(
    request: UpscaleRequest, background_tasks: BackgroundTasks
):
    """Upscale an image (deprecated - use the file upload endpoint)."""
    raise HTTPException(
        status_code=501,
        detail="Use /api/processing/upscale/file endpoint for file upload",
    )


@router.post("/processing/upscale/file")
async def process_upscale_file(
    file: UploadFile = File(...),
    model_name: str = Form("Real-ESRGAN"),
    scale: int = Form(4),
    device: Optional[str] = Form(None),
    output_format: Optional[str] = Form(None),
    jpg_quality: Optional[int] = Form(None),
):
    """Upscale an uploaded image or video file."""
    _require_processing()
    _require_running()
    assert _upscale_image is not None and _upscale_video is not None
    assert _get_upscale_model is not None and _list_available_upscale_models is not None

    is_image = file.content_type in ALLOWED_IMAGE_TYPES
    is_video = file.content_type in ALLOWED_VIDEO_TYPES
    if not (is_image or is_video):
        raise HTTPException(
            status_code=400, detail=f"Unsupported file type: {file.content_type}"
        )

    # scale=1 is a passthrough — no model required
    passthrough = scale == 1

    if not passthrough:
        model_path = _get_upscale_model(model_name)
        if not model_path:
            downloaded = [
                m for m in _list_available_upscale_models() if m["downloaded"]
            ]
            if downloaded:
                model_name = downloaded[0]["name"]
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No upscale models downloaded. "
                        "Set scale to 1x or download a model first."
                    ),
                )

    saved_settings = _effective_settings()

    # Resolve the compute backend: an explicit request wins, otherwise honour
    # the persisted setting (normalising legacy adapter names to torch devices).
    configured_backend = saved_settings.get("computeBackend", "auto")
    configured_device = str(saved_settings.get("gpuDevice", "auto"))
    if device and device != "auto":
        selected_device = device
    elif configured_backend == "cpu":
        selected_device = "cpu"
    elif configured_backend == "cuda":
        selected_device = (
            configured_device if configured_device.startswith("cuda") else "cuda"
        )
    elif configured_backend == "metal":
        selected_device = (
            configured_device if configured_device.startswith("mps") else "mps"
        )
    elif configured_backend == "hip":
        selected_device = (
            configured_device
            if configured_device.startswith(("hip", "cuda"))
            else "hip"
        )
    elif configured_backend in ("amd", "dml"):
        selected_device = (
            configured_device
            if configured_device.startswith(("privateuseone", "dml", "hip", "cuda"))
            else "dml"
        )
    else:
        selected_device = (
            configured_device if configured_device not in ("", "auto") else "auto"
        )

    selected_device = _resolve_device(selected_device, saved_settings)

    selected_format = output_format or saved_settings.get("outputFormat", "jpg")
    if selected_format not in VALID_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400, detail=f"Unsupported output format: {selected_format}"
        )
    if is_image and selected_format == "mp4":
        raise HTTPException(
            status_code=400, detail="MP4 output is only supported for video files"
        )

    try:
        quality = int(
            jpg_quality
            if jpg_quality is not None
            else saved_settings.get("jpgQuality", 95)
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="JPG quality must be an integer between 1 and 100",
        ) from exc
    if not 1 <= quality <= 100:
        raise HTTPException(
            status_code=400, detail="JPG quality must be between 1 and 100"
        )

    # The legacy default (400) is auto-tuned to a VRAM-aware value on GPU runs.
    tile_size = system_resources.resolve_tile_size(
        int(
            saved_settings.get("defaultTileSize")
            or saved_settings.get("tileSize")
            or 400
        ),
        selected_device,
    )

    input_path = await _stage_upload(file)
    safe_filename = Path(file.filename or "upload").name
    input_suffix = Path(safe_filename).suffix.lower()

    if is_video:
        output_suffix = ".mp4"
    else:
        output_suffix = IMAGE_SUFFIX_BY_FORMAT.get(selected_format, input_suffix)

    output_path = TEMP_PROCESSING_DIR / (
        f"{uuid.uuid4()}_processed_{Path(safe_filename).stem}{output_suffix}"
    )

    fp16 = bool(saved_settings.get("fp16Enabled", True))
    upscale_image_fn = _upscale_image
    upscale_video_fn = _upscale_video

    task_id = str(uuid.uuid4())
    process_progress[task_id] = {"status": "processing", "progress": 0}

    def process_task() -> None:
        _patch_distutils_in_worker()

        def progress_cb(percent):
            # Cooperative abort: the window was closed, stop the task now
            if is_shutdown_requested():
                raise ProcessingAborted(
                    "Processing aborted: application is shutting down"
                )
            if task_id in process_progress:
                process_progress[task_id]["progress"] = percent
                # Force yield to ensure visibility across threads
                import time

                time.sleep(0.001)

        try:
            _apply_torch_runtime_settings(saved_settings, selected_device)
            _load_processing()
            if not PROCESSING_AVAILABLE:
                raise RuntimeError("Processing modules not available")

            if passthrough:
                # scale=1: copy or re-encode without upscaling
                if is_image:
                    progress_cb(10)
                    if selected_format not in ("source", input_suffix.lstrip(".")):
                        if cv2 is None:
                            raise RuntimeError("OpenCV (cv2) is not available")
                        img = cv2.imread(str(input_path))
                        write_params: List[int] = []
                        if selected_format in ("jpg", "jpeg"):
                            write_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                        cv2.imwrite(str(output_path), img, write_params)
                    else:
                        shutil.copy2(input_path, output_path)
                    progress_cb(100)
                else:
                    progress_cb(10)
                    if selected_format in ("mp4", "source"):
                        shutil.copy2(input_path, output_path)
                    else:
                        if cv2 is None:
                            raise RuntimeError("OpenCV (cv2) is not available")
                        cap = cv2.VideoCapture(str(input_path))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out = cv2.VideoWriter(
                            str(output_path), fourcc, fps, (width, height)
                        )
                        frame_count = 0
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            out.write(frame)
                            frame_count += 1
                            if frame_count % 30 == 0:
                                progress_cb(10 + int(frame_count / total_frames * 80))
                        cap.release()
                        out.release()
                    progress_cb(100)

                result: Dict[str, Any] = {
                    "success": True,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "scale_factor": 1,
                    "model": "passthrough",
                    "passthrough": True,
                }
            elif is_image:
                assert upscale_image_fn is not None
                result = upscale_image_fn(
                    str(input_path),
                    str(output_path),
                    model_name=model_name,
                    scale=scale,
                    device=selected_device,
                    tile_size=tile_size,
                    fp16=fp16,
                    jpg_quality=quality,
                    progress_callback=progress_cb,
                )
            else:
                assert upscale_video_fn is not None
                result = upscale_video_fn(
                    str(input_path),
                    str(output_path),
                    model_name=model_name,
                    scale=scale,
                    device=selected_device,
                    tile_size=tile_size,
                    fp16=fp16,
                    jpg_quality=quality,
                    progress_callback=progress_cb,
                )

            process_progress[task_id] = {
                "status": "completed",
                "result": result,
                "progress": 100,
            }
        except ProcessingAborted as exc:
            print(f"[Worker] {exc}", flush=True)
            # Remove the partially-written output file (if any)
            try:
                if output_path.exists():
                    output_path.unlink()
                    print(
                        f"[Worker] Removed partial output: {output_path.name}",
                        flush=True,
                    )
            except OSError:
                pass
            process_progress[task_id] = {
                "status": "aborted",
                "error": str(exc),
                "progress": 0,
            }
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a task error
            import traceback

            traceback.print_exc()
            process_progress[task_id] = {
                "status": "failed",
                "error": str(exc),
                "progress": 0,
            }
        finally:
            try:
                input_path.unlink()
            except OSError:
                pass

    loop = asyncio.get_event_loop()
    process_tasks[task_id] = loop.run_in_executor(None, process_task)
    return {"task_id": task_id, "status": "started"}


# ---------------------------------------------------------------------------
# Face recovery
# ---------------------------------------------------------------------------

@router.post("/processing/face-recovery/file")
@router.post("/processing/face/file")
async def process_face_recovery_file(
    file: UploadFile = File(...),
    model_name: str = Form("CodeFormer"),
    fidelity_weight: float = Form(0.5),
    device: Optional[str] = Form(None),
):
    """Apply face recovery to an uploaded image or video."""
    from src.processing import FaceRecoveryPipeline, get_face_model

    _require_processing()
    _require_running()

    saved_settings = _effective_settings()
    selected_device = _resolve_device(device, saved_settings)

    input_path = await _stage_upload(file)
    safe_filename = Path(file.filename or "upload").name
    output_path = TEMP_PROCESSING_DIR / (
        f"{uuid.uuid4()}_face_restored_"
        f"{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    )

    def run(_task_id: str) -> Dict[str, Any]:
        pipeline = FaceRecoveryPipeline(
            model_name=model_name,
            device=selected_device,
            fidelity_weight=fidelity_weight,
        )
        model_path = get_face_model(model_name)
        if model_path is None:
            raise RuntimeError(
                f"Face recovery model '{model_name}' not found. "
                "Please download it from the Model Manager."
            )
        pipeline.load_model(model_path)
        return pipeline.process(str(input_path), str(output_path))

    return _dispatch(run, [input_path])


# ---------------------------------------------------------------------------
# Sharpen / deblur
# ---------------------------------------------------------------------------

@router.post("/processing/sharpen/file")
async def process_sharpen_file(
    file: UploadFile = File(...),
    model_name: str = Form("NAFNet"),
    strength: float = Form(1.0),
    overlap: Optional[int] = Form(None),
    unsharp_amount: Optional[float] = Form(None),
    device: Optional[str] = Form(None),
):
    """Apply sharpening/deblurring to an uploaded image or video."""
    from src.processing import SharpenPipeline, get_sharpen_model, list_sharpen_models

    _require_processing()
    _require_running()

    saved_settings = _effective_settings()
    if overlap is None:
        overlap = int(saved_settings.get("sharpenOverlap", 128))
    if unsharp_amount is None:
        unsharp_amount = float(saved_settings.get("sharpenUnsharpAmount", 0.5))

    # Fail fast instead of silently falling back to classical sharpening:
    # a missing weights file must never produce a "sharpened" but still blurry
    # image without the user knowing.
    model_path = get_sharpen_model(model_name)
    if not model_path:
        names = [m["name"] for m in list_sharpen_models() if m["downloaded"]]
        detail = (
            f"Sharpening model '{model_name}' is not downloaded. "
            + (
                f"Installed: {', '.join(names)}."
                if names
                else (
                    "No sharpen models are installed - "
                    "download one from the model manager."
                )
            )
        )
        raise HTTPException(status_code=400, detail=detail)

    selected_device = _resolve_device(device, saved_settings)

    input_path = await _stage_upload(file)
    safe_filename = Path(file.filename or "upload").name
    output_path = TEMP_PROCESSING_DIR / (
        f"{uuid.uuid4()}_sharpened_"
        f"{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    )

    def run(_task_id: str) -> Dict[str, Any]:
        pipeline = SharpenPipeline(
            model_name=model_name,
            device=selected_device,
            strength=strength,
            overlap=overlap,
            unsharp_amount=unsharp_amount,
        )
        pipeline.load_model(model_path)
        result = pipeline.process(str(input_path), str(output_path))
        result.setdefault("overlap", overlap)
        result.setdefault("window", "hamming")
        result.setdefault("unsharp_amount", unsharp_amount)
        result.setdefault("model_path", model_path)
        return result

    return _dispatch(run, [input_path])


# ---------------------------------------------------------------------------
# Denoise
# ---------------------------------------------------------------------------

@router.post("/processing/denoise/file")
async def process_denoise_file(
    file: UploadFile = File(...),
    model_name: str = Form("SCUNet"),
    strength: float = Form(0.5),
    device: Optional[str] = Form(None),
):
    """Apply denoising to an uploaded image or video."""
    from src.processing import DenoisePipeline, get_denoise_model

    _require_processing()
    _require_running()

    saved_settings = _effective_settings()
    selected_device = _resolve_device(device, saved_settings)

    input_path = await _stage_upload(file)
    safe_filename = Path(file.filename or "upload").name
    output_path = TEMP_PROCESSING_DIR / (
        f"{uuid.uuid4()}_denoised_"
        f"{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    )

    def run(_task_id: str) -> Dict[str, Any]:
        pipeline = DenoisePipeline(
            model_name=model_name, device=selected_device, strength=strength
        )
        model_path = get_denoise_model(model_name)
        if model_path is None:
            raise RuntimeError(
                f"Denoise model '{model_name}' not found. "
                "Please download it from the Model Manager."
            )
        pipeline.load_model(model_path)
        return pipeline.process(str(input_path), str(output_path))

    return _dispatch(run, [input_path])


# ---------------------------------------------------------------------------
# Frame interpolation
# ---------------------------------------------------------------------------

def _require_video(file: UploadFile, feature: str) -> None:
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400, detail=f"{feature} only supports video files"
        )


@router.post("/processing/frame-interpolation/file")
async def process_frame_interpolation_file(
    file: UploadFile = File(...),
    model_name: str = Form("RIFE"),
    multiplier: int = Form(2),
    device: Optional[str] = Form(None),
):
    """Apply frame interpolation to an uploaded video."""
    from src.processing import FrameInterpolationPipeline, get_frame_interp_model

    _require_processing()
    _require_running()
    _require_video(file, "Frame interpolation")

    saved_settings = _effective_settings()
    selected_device = _resolve_device(device, saved_settings)

    input_path = await _stage_upload(file)
    safe_filename = Path(file.filename or "upload").name
    output_path = TEMP_PROCESSING_DIR / (
        f"{uuid.uuid4()}_interp_{multiplier}x_{Path(safe_filename).stem}.mp4"
    )

    def run(_task_id: str) -> Dict[str, Any]:
        pipeline = FrameInterpolationPipeline(
            model_name=model_name, device=selected_device, multiplier=multiplier
        )
        model_path = get_frame_interp_model(model_name)
        if model_path is None:
            raise RuntimeError(
                f"Frame interpolation model '{model_name}' not found. "
                "Please download it from the Model Manager."
            )
        pipeline.load_model(model_path)
        return pipeline.process(str(input_path), str(output_path))

    return _dispatch(run, [input_path])


# ---------------------------------------------------------------------------
# Slow motion
# ---------------------------------------------------------------------------

@router.post("/processing/slow-motion/file")
async def process_slow_motion_file(
    file: UploadFile = File(...),
    model_name: str = Form("RIFE-SloMo"),
    slowdown_factor: float = Form(4.0),
    device: Optional[str] = Form(None),
):
    """Create a slow-motion video from an uploaded video."""
    from src.processing import SlowMotionPipeline

    _require_processing()
    _require_running()
    _require_video(file, "Slow motion")

    saved_settings = _effective_settings()
    selected_device = _resolve_device(device, saved_settings)

    input_path = await _stage_upload(file)
    safe_filename = Path(file.filename or "upload").name
    output_path = TEMP_PROCESSING_DIR / (
        f"{uuid.uuid4()}_slowmo_{slowdown_factor}x_{Path(safe_filename).stem}.mp4"
    )

    def run(_task_id: str) -> Dict[str, Any]:
        pipeline = SlowMotionPipeline(
            model_name=model_name,
            device=selected_device,
            slowdown_factor=slowdown_factor,
        )
        pipeline.load_model()
        return pipeline.process(str(input_path), str(output_path))

    return _dispatch(run, [input_path])


# ---------------------------------------------------------------------------
# Match color
# ---------------------------------------------------------------------------

@router.post("/processing/match-color/file")
async def process_match_color_file(
    file: UploadFile = File(...),
    reference: Optional[UploadFile] = File(None),
    model_name: str = Form("Reinhard Transfer"),
    strength: float = Form(1.0),
    preserve_luminance: bool = Form(True),
    device: Optional[str] = Form(None),
):
    """Match colors from a reference image onto the target."""
    from src.processing import MatchColorPipeline, get_match_color_model

    _require_processing()
    _require_running()

    saved_settings = _effective_settings()
    selected_device = _resolve_device(device, saved_settings)

    input_path = await _stage_upload(file)
    safe_filename = Path(file.filename or "upload").name
    output_path = TEMP_PROCESSING_DIR / (
        f"{uuid.uuid4()}_color_matched_"
        f"{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    )

    ref_path: Optional[Path] = None
    if reference is not None:
        TEMP_PROCESSING_DIR.mkdir(exist_ok=True)
        ref_name = Path(reference.filename or "reference").name
        ref_path = TEMP_PROCESSING_DIR / f"{uuid.uuid4()}_ref_{ref_name}"
        with open(ref_path, "wb") as handle:
            handle.write(await reference.read())

    def run(_task_id: str) -> Dict[str, Any]:
        pipeline = MatchColorPipeline(
            model_name=model_name,
            device=selected_device,
            strength=strength,
            preserve_luminance=preserve_luminance,
        )
        # For non-neural models like Reinhard, model_paths can be None;
        # load_model handles that case.
        pipeline.load_model(get_match_color_model(model_name))
        return pipeline.process(
            str(input_path),
            str(output_path),
            reference_path=str(ref_path) if ref_path else None,
        )

    return _dispatch(run, [input_path, ref_path])


# ---------------------------------------------------------------------------
# Low-light enhancement
# ---------------------------------------------------------------------------

@router.post("/processing/low-light/file")
async def process_low_light_file(
    file: UploadFile = File(...),
    model_name: str = Form("Zero-DCE"),
    strength: float = Form(0.7),
    device: Optional[str] = Form(None),
):
    """Enhance low-light images or videos."""
    from src.processing import LowLightEnhancementPipeline

    _require_processing()
    _require_running()

    input_path = await _stage_upload(file)
    safe_filename = Path(file.filename or "upload").name
    output_path = TEMP_PROCESSING_DIR / (
        f"{uuid.uuid4()}_lowlight_"
        f"{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    )

    def run(_task_id: str) -> Dict[str, Any]:
        pipeline = LowLightEnhancementPipeline(
            model_name=model_name, strength=strength
        )
        pipeline.load_model()
        return pipeline.process(str(input_path), str(output_path))

    return _dispatch(run, [input_path])


# ---------------------------------------------------------------------------
# Log restoration
# ---------------------------------------------------------------------------

@router.post("/processing/log-restoration/file")
async def process_log_restoration_file(
    file: UploadFile = File(...),
    model_name: str = Form("ITMLUT"),
    color_space_target: str = Form("Rec.709"),
    device: Optional[str] = Form(None),
):
    """Restore LOG footage to a standard color space."""
    from src.processing import LogRestorationPipeline

    _require_processing()
    _require_running()

    input_path = await _stage_upload(file)
    safe_filename = Path(file.filename or "upload").name
    output_path = TEMP_PROCESSING_DIR / (
        f"{uuid.uuid4()}_log_restored_"
        f"{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    )

    def run(_task_id: str) -> Dict[str, Any]:
        pipeline = LogRestorationPipeline(
            model_name=model_name, color_space_target=color_space_target
        )
        pipeline.load_model()
        return pipeline.process(str(input_path), str(output_path))

    return _dispatch(run, [input_path])


# ---------------------------------------------------------------------------
# SDR -> HDR
# ---------------------------------------------------------------------------

@router.post("/processing/sdr-to-hdr/file")
async def process_sdr_to_hdr_file(
    file: UploadFile = File(...),
    model_name: str = Form("HDRNet"),
    target_format: str = Form("HDR10"),
    device: Optional[str] = Form(None),
):
    """Convert SDR footage to HDR."""
    from src.processing import SDRtoHDRPipeline

    _require_processing()
    _require_running()

    input_path = await _stage_upload(file)
    safe_filename = Path(file.filename or "upload").name
    output_path = TEMP_PROCESSING_DIR / (
        f"{uuid.uuid4()}_hdr_{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    )

    def run(_task_id: str) -> Dict[str, Any]:
        pipeline = SDRtoHDRPipeline(model_name=model_name, target_format=target_format)
        pipeline.load_model()
        return pipeline.process(str(input_path), str(output_path))

    return _dispatch(run, [input_path])


# ---------------------------------------------------------------------------
# Overexposure fix
# ---------------------------------------------------------------------------

@router.post("/processing/overexposure-fix/file")
async def process_overexposure_fix_file(
    file: UploadFile = File(...),
    model_name: str = Form("AutoEnhance"),
    device: Optional[str] = Form(None),
):
    """Fix overexposure and contrast issues."""
    from src.processing import OverexposureFixPipeline

    _require_processing()
    _require_running()

    input_path = await _stage_upload(file)
    safe_filename = Path(file.filename or "upload").name
    output_path = TEMP_PROCESSING_DIR / (
        f"{uuid.uuid4()}_fixed_"
        f"{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    )

    def run(_task_id: str) -> Dict[str, Any]:
        pipeline = OverexposureFixPipeline(model_name=model_name)
        pipeline.load_model()
        return pipeline.process(str(input_path), str(output_path))

    return _dispatch(run, [input_path])


# ---------------------------------------------------------------------------
# Flat color profile
# ---------------------------------------------------------------------------

@router.post("/processing/flat-profile/file")
async def process_flat_profile_file(
    file: UploadFile = File(...),
    model_name: str = Form("ColorGrade"),
    device: Optional[str] = Form(None),
):
    """Apply a flat color profile for downstream color correction."""
    from src.processing import FlatColorProfilePipeline

    _require_processing()
    _require_running()

    input_path = await _stage_upload(file)
    safe_filename = Path(file.filename or "upload").name
    output_path = TEMP_PROCESSING_DIR / (
        f"{uuid.uuid4()}_flat_{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    )

    def run(_task_id: str) -> Dict[str, Any]:
        pipeline = FlatColorProfilePipeline(model_name=model_name)
        pipeline.load_model()
        return pipeline.process(str(input_path), str(output_path))

    return _dispatch(run, [input_path])


# ---------------------------------------------------------------------------
# Reencoding
# ---------------------------------------------------------------------------

@router.post("/processing/reencode/file")
async def process_reencode_file(
    file: UploadFile = File(...),
    engine: str = Form("ffmpeg"),
    preset: str = Form("h264_fast"),
    device: Optional[str] = Form(None),
):
    """Reencode a video using FFmpeg or Handbrake."""
    from src.processing import ReencodingPipeline

    _require_processing()
    _require_running()
    _require_video(file, "Reencoding")

    input_path = await _stage_upload(file)
    safe_filename = Path(file.filename or "upload").name
    output_path = TEMP_PROCESSING_DIR / (
        f"{uuid.uuid4()}_reencoded_{Path(safe_filename).stem}.mp4"
    )

    def run(_task_id: str) -> Dict[str, Any]:
        pipeline = ReencodingPipeline(engine=engine, preset=preset)
        pipeline.load_model()
        return pipeline.process(str(input_path), str(output_path))

    return _dispatch(run, [input_path])


# ---------------------------------------------------------------------------
# Pipeline Chaining (multiple features in sequence)
# ---------------------------------------------------------------------------

@router.post("/processing/pipeline/file")
async def process_pipeline_file(
    file: UploadFile = File(...),
    pipeline_config: Dict[str, Any] = Form(...),
    device: Optional[str] = Form(None),
):
    """Execute multiple enhancement pipelines in sequence (Denoise → Upscale → Face Recovery, etc.)"""
    from src.processing import PipelineChain
    _require_processing()
    _require_running()

    input_path = await _stage_upload(file)
    safe_filename = Path(file.filename or "upload").name
    output_path = TEMP_PROCESSING_DIR / (
        f"{uuid.uuid4()}_pipelined_{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    )

    # Build pipeline from configuration
    features = pipeline_config.get("features", [])
    selected_device = device or _resolve_device(device, _effective_settings())
    
    # Create pipeline classes based on features
    pipeline_classes = []
    
    for feature in features:
        if feature == "upscale":
            from src.processing import UpscalePipeline
            pipeline_classes.append((UpscalePipeline, {
                "model_name": pipeline_config.get("upscale_model", "Real-ESRGAN"),
                "scale": pipeline_config.get("upscale_scale", 4),
            }))
        elif feature == "face-recovery":
            from src.processing import FaceRecoveryPipeline
            pipeline_classes.append((FaceRecoveryPipeline, {
                "model_name": pipeline_config.get("face_recovery_model", "CodeFormer"),
                "fidelity_weight": pipeline_config.get("face_fidelity", 0.5),
            }))
        elif feature == "sharpen":
            from src.processing import SharpenPipeline
            pipeline_classes.append((SharpenPipeline, {
                "model_name": pipeline_config.get("sharpen_model", "NAFNet"),
                "strength": pipeline_config.get("sharpen_strength", 0.5),
                "overlap": pipeline_config.get("sharpen_overlap", 128),
                "unsharp_amount": pipeline_config.get("sharpen_unsharp_amount", 0.5),
            }))
        elif feature == "denoise":
            from src.processing import DenoisePipeline
            pipeline_classes.append((DenoisePipeline, {
                "model_name": pipeline_config.get("denoise_model", "SCUNet"),
                "strength": pipeline_config.get("denoise_strength", 0.5),
            }))
        elif feature == "frame-interpolation":
            from src.processing import FrameInterpolationPipeline
            pipeline_classes.append((FrameInterpolationPipeline, {
                "model_name": pipeline_config.get("frame_interp_model", "RIFE"),
                "multiplier": pipeline_config.get("frame_interp_multiplier", 2),
            }))
        elif feature == "slow-motion":
            from src.processing import SlowMotionPipeline
            pipeline_classes.append((SlowMotionPipeline, {
                "model_name": pipeline_config.get("slow_motion_model", "RIFE-SloMo"),
                "slowdown_factor": pipeline_config.get("slowdown_factor", 4.0),
            }))
        elif feature == "match-color":
            from src.processing import MatchColorPipeline
            pipeline_classes.append((MatchColorPipeline, {
                "model_name": pipeline_config.get("match_color_model", "Reinhard Transfer"),
                "strength": pipeline_config.get("match_color_strength", 1.0),
                "preserve_luminance": pipeline_config.get("preserve_luminance", True),
            }))
        elif feature == "low-light":
            from src.processing import LowLightEnhancementPipeline
            pipeline_classes.append((LowLightEnhancementPipeline, {
                "model_name": pipeline_config.get("low_light_model", "Zero-DCE"),
                "strength": pipeline_config.get("low_light_strength", 0.7),
            }))
        else:
            # Skip unknown features
            continue
    
    if not pipeline_classes:
        raise HTTPException(status_code=400, detail="No valid features specified in pipeline configuration")

    def run(_task_id: str) -> Dict[str, Any]:
        # Create pipeline instances
        pipelines = []
        for pipeline_class, kwargs in pipeline_classes:
            pipeline = pipeline_class(device=selected_device, **kwargs)
            pipelines.append(pipeline)
        
        # Load models for all pipelines
        for i, pipeline in enumerate(pipelines):
            feature = features[i]
            if feature == "upscale":
                model_path = _get_upscale_model(pipeline.model_name)
            elif feature == "face-recovery":
                model_path = get_face_model(pipeline.model_name)
            elif feature == "sharpen":
                model_path = get_sharpen_model(pipeline.model_name)
            elif feature == "denoise":
                model_path = get_denoise_model(pipeline.model_name)
            elif feature == "frame-interpolation":
                model_path = get_frame_interp_model(pipeline.model_name)
            elif feature == "slow-motion":
                model_path = None
            elif feature == "match-color":
                model_path = get_match_color_model(pipeline.model_name)
            elif feature == "low-light":
                model_path = None
            else:
                model_path = None
            
            if model_path:
                pipeline.load_model(model_path)
            else:
                pipeline.load_model()
        
        # Execute pipeline chain
        chain = PipelineChain(pipelines)
        return chain.process(str(input_path), str(output_path))

    return _dispatch(run, [input_path])

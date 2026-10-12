"""
FastAPI backend for openVideoAI
Serves the UI and provides REST API for model management
"""
from .distutils_compat import install_distutils_compat

install_distutils_compat()

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Callable, List, Literal, Optional, Dict, Any
import asyncio
import json
import uuid
import os
import shutil
from pathlib import Path
from types import ModuleType
import platform

from . import system_ressources as system_resources
import psutil

try:
    import torch
except ImportError:
    torch = None

try:
    import cv2
except ImportError:
    cv2 = None

from src.models.initialize import (
    get_model_manager, initialize_models, get_model_status,
    get_downloaded_models, get_available_models, download_model,
    ensure_model, is_model_downloaded, get_model_by_name,
    get_all_models, get_global_features, ModelInfo, DownloadProgress
)
from src.schemas import (
    SettingsModel,
    ModelResponse,
    DownloadRequest,
    DownloadStatus,
    UpscaleRequest,
    ProcessResponse,
)
from src.core.config import (
    DEFAULT_SETTINGS as CORE_DEFAULT_SETTINGS,
    MAX_SETTINGS as CORE_MAX_SETTINGS,
    EXPORTS_DIR
)
from src.processing.base import ProcessingAborted

from .core import paths as _paths

PROJECT_ROOT = _paths.project_root()
SETTINGS_FILE = _paths.settings_file()
TEMP_PROCESSING_DIR = _paths.temp_processing_dir()


UpscaleFunction = Callable[..., Dict[str, Any]]
ModelPathLookup = Callable[[str], Optional[str]]
ModelList = Callable[[], List[Dict[str, Any]]]

# Processing imports - lazy loaded to allow distutils monkey patch
PROCESSING_AVAILABLE = False
_upscale_image: Optional[UpscaleFunction] = None
_upscale_video: Optional[UpscaleFunction] = None
_get_upscale_model: Optional[ModelPathLookup] = None
_list_available_upscale_models: Optional[ModelList] = None
_UpscalePipeline = None

def _load_processing():
    """Lazy load processing modules after distutils monkey patch"""
    global PROCESSING_AVAILABLE, _upscale_image, _upscale_video, _get_upscale_model, _list_available_upscale_models, _UpscalePipeline
    if PROCESSING_AVAILABLE:
        return
    try:
        from src.processing.upscale import (
            upscale_image, upscale_video, get_upscale_model,
            list_available_upscale_models, UpscalePipeline
        )
        _upscale_image = upscale_image
        _upscale_video = upscale_video
        _get_upscale_model = get_upscale_model
        _list_available_upscale_models = list_available_upscale_models
        _UpscalePipeline = UpscalePipeline
        PROCESSING_AVAILABLE = True
    except ImportError as e:
        print(f"Processing modules not available: {e}")
        PROCESSING_AVAILABLE = False


# Default settings for the application
DEFAULT_SETTINGS = dict(CORE_DEFAULT_SETTINGS)
MAX_SETTINGS = dict(CORE_MAX_SETTINGS)

app = FastAPI(title="openVideoAI", version="0.1.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Progress tracking for active downloads
download_progress: Dict[str, DownloadProgress] = {}
download_tasks: Dict[str, asyncio.Task] = {}


def _build_compatibility(model: Dict[str, Any]) -> Dict[str, Any]:
    """Compute platform compatibility for a model dict from the catalog.

    Models that declare requiresCuda / requiresMetal / canRunOnCpu in
    models.json are matched against the detected hardware backends
    (CUDA, AMD DirectML/ROCm, Metal, or CPU).
    """
    compat = system_resources.check_model_compatibility(
        requires_cuda=bool(model.get('requires_cuda', model.get('requiresCuda', False))),
        requires_metal=bool(model.get('requires_metal', model.get('requiresMetal', False))),
        can_run_on_cpu=bool(model.get('can_run_on_cpu', model.get('canRunOnCpu', True))),
    )
    return {
        "compatible": compat["compatible"],
        "compatibility_reason": compat["reason"] or None,
    }


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    def progress_cb(msg: str, progress: float):
        print(f"[Init] {msg} ({progress*100:.0f}%)")
    
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, initialize_models, progress_cb)


@app.get("/api/models", response_model=List[ModelResponse])
async def list_models():
    """Get all models with their download status"""
    status = get_model_status()
    return [
        ModelResponse(
            name=m['name'],
            description=m['description'],
            size=m['size'],
            weight=m['weight'],
            downloaded=m['downloaded'],
            features=m['features'],
            is_ai=m.get('is_ai', True),
            is_builtin=m.get('is_builtin', False),
            requires_cuda=m.get('requires_cuda', False),
            requires_metal=m.get('requires_metal', False),
            can_run_on_cpu=m.get('can_run_on_cpu', True),
            has_download_url=m.get('has_download_url', False),
            compatible=_build_compatibility(m)["compatible"],
            compatibility_reason=_build_compatibility(m)["compatibility_reason"],
        )
        for m in status['models']
    ]


@app.get("/api/features")
async def get_features():
    """Get all globally defined features from models.json"""
    features = get_global_features()
    return [
        {"name": f['name'], "description": f['description']}
        for f in features
    ]


@app.get("/api/modules")
async def get_modules():
    """Get configured processing modules from models.json"""
    catalog_path = get_model_manager().downloader.models_json
    with open(catalog_path, 'r', encoding='utf-8') as catalog_file:
        data = json.load(catalog_file)
    return data.get('modules', [])


@app.get("/api/features/implemented")
async def get_implemented_features():
    """Get features that have UI implementations in the frontend"""
    # These match the FEATURES keys in app.js
    implemented = {
        "Upscale": {"key": "base", "required": True, "hasToggle": False, "hasStrength": False},
        "Face Recovery": {"key": "face", "required": False, "hasToggle": True, "hasStrength": False},
        "Sharpen / Deblur": {"key": "sharpen", "required": False, "hasToggle": True, "hasStrength": True},
        "Denoise": {"key": "denoise", "required": False, "hasToggle": True, "hasStrength": True},
        "Frame Interpolation": {"key": "frame", "required": False, "hasToggle": True, "hasStrength": False},
        "Match Color": {"key": "color", "required": False, "hasToggle": True, "hasStrength": True, "requiresReferenceImage": True},
        "Log Restoration": {"key": "log", "required": False, "hasToggle": True, "hasStrength": False},
    }
    return implemented


@app.get("/api/models/{model_name}", response_model=ModelResponse)
async def get_model(model_name: str):
    """Get details for a specific model"""
    model = get_model_by_name(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    downloaded = is_model_downloaded(model_name)
    compatibility = _build_compatibility({
        'requires_cuda': model.requires_cuda,
        'requires_metal': model.requires_metal,
    })
    return ModelResponse(
        name=model.name,
        description=model.description_short,
        size=model.size,
        weight=model.weight,
        downloaded=downloaded,
        features=[f['name'] for f in model.features],
        is_ai=model.is_ai,
        is_builtin=model.is_builtin,
        requires_cuda=model.requires_cuda,
        requires_metal=model.requires_metal,
        can_run_on_cpu=model.can_run_on_cpu,
        has_download_url=bool(model.download_url),
        compatible=compatibility["compatible"],
        compatibility_reason=compatibility["compatibility_reason"],
    )


@app.post("/api/models/download")
async def start_download(request: DownloadRequest, background_tasks: BackgroundTasks):
    """Start downloading a model"""
    model_name = request.model_name
    
    if is_model_downloaded(model_name):
        raise HTTPException(status_code=400, detail="Model already downloaded")
    
    if model_name in download_tasks and not download_tasks[model_name].done():
        raise HTTPException(status_code=400, detail="Download already in progress")
    
    model = get_model_by_name(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Hardware compatibility gate: require the backend the model declares in models.json.
    compatibility = _build_compatibility({
        'requires_cuda': model.requires_cuda,
        'requires_metal': model.requires_metal,
    })
    if not compatibility["compatible"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot download '{model_name}': {compatibility['compatibility_reason']}",
        )
    
    # Create progress tracking
    progress = DownloadProgress(
        model_name=model_name,
        total_bytes=0,
        downloaded_bytes=0,
        speed_bps=0,
        eta_seconds=0,
        status='starting'
    )
    download_progress[model_name] = progress
    
    # Start download in background
    async def download_task():
        def progress_callback(p: DownloadProgress):
            download_progress[model_name] = p
        
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None, download_model, model_name, progress_callback
        )
        
        # Update final status
        if success:
            download_progress[model_name].status = 'completed'
        else:
            download_progress[model_name].status = 'failed'
    
    task = asyncio.create_task(download_task())
    download_tasks[model_name] = task
    
    return {"message": "Download started", "model_name": model_name}


@app.get("/api/models/{model_name}/progress", response_model=DownloadStatus)
async def get_download_progress(model_name: str):
    """Get download progress for a model"""
    if model_name not in download_progress:
        raise HTTPException(status_code=404, detail="No download in progress")
    
    p = download_progress[model_name]
    progress_pct = 0.0
    if p.total_bytes > 0:
        progress_pct = (p.downloaded_bytes / p.total_bytes) * 100
    
    return DownloadStatus(
        model_name=p.model_name,
        total_bytes=p.total_bytes,
        downloaded_bytes=p.downloaded_bytes,
        speed_bps=p.speed_bps,
        eta_seconds=p.eta_seconds,
        status=p.status,
        error=p.error,
        progress_percent=progress_pct
    )


@app.post("/api/models/{model_name}/cancel")
async def cancel_download(model_name: str):
    """Cancel an ongoing download"""
    manager = get_model_manager()
    manager.downloader.cancel_download(model_name)
    
    if model_name in download_tasks:
        download_tasks[model_name].cancel()
        del download_tasks[model_name]
    
    if model_name in download_progress:
        download_progress[model_name].status = 'cancelled'
    
    return {"message": "Download cancelled"}


@app.delete("/api/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a downloaded model"""
    manager = get_model_manager()
    if manager.downloader.delete_model(model_name):
        return {"message": "Model deleted"}
    raise HTTPException(status_code=404, detail="Model not found")


@app.get("/api/status")
async def get_status():
    """Get overall system status"""
    status = get_model_status()
    return {
        "total_models": status['total'],
        "downloaded": status['downloaded'],
        "available": status['available'],
        "models_dir": str(Path("models").absolute())
    }


# Processing endpoints


# Progress tracking for active processing
process_progress: Dict[str, Dict[str, Any]] = {}
process_tasks: Dict[str, asyncio.Future[None]] = {}

# Cooperative shutdown flag: set when the desktop window closes so any in-flight
# processing task aborts instead of silently continuing in the background
# (which previously left a zombie server holding the port).
_shutdown_requested = False


def request_processing_shutdown() -> None:
    """Signal running processing tasks to abort (called on window close)."""
    global _shutdown_requested
    _shutdown_requested = True


def has_active_processing() -> bool:
    """True when at least one processing task is still running."""
    return any(p.get('status') == 'processing' for p in process_progress.values())


@app.get("/api/processing/upscale/models")
async def get_upscale_models():
    """List available upscale models"""
    _load_processing()
    list_models = _list_available_upscale_models
    if not PROCESSING_AVAILABLE or list_models is None:
        raise HTTPException(status_code=503, detail="Processing not available")
    
    return list_models()


@app.post("/api/processing/upscale/image")
async def process_upscale_image(request: UpscaleRequest, background_tasks: BackgroundTasks):
    """Upscale an image"""
    _load_processing()
    get_model_path = _get_upscale_model
    list_models = _list_available_upscale_models
    if not PROCESSING_AVAILABLE or get_model_path is None or list_models is None:
        raise HTTPException(status_code=503, detail="Processing not available")
    
    # Find a downloaded upscale model
    model_path = get_model_path(request.model_name)
    if not model_path:
        # Try to find any downloaded upscale model
        available = list_models()
        downloaded = [m for m in available if m['downloaded']]
        if downloaded:
            model_path = downloaded[0]['path']
            request.model_name = downloaded[0]['name']
        else:
            raise HTTPException(status_code=400, detail="No upscale models downloaded. Download one first.")
    
    # Generate output path if not provided
    if request.output_path:
        output_path = request.output_path
    else:
        input_path = Path(request.model_name)  # This will be overridden
        # We need the input file path - for now require it in request
        raise HTTPException(status_code=400, detail="output_path required for now. Use /api/processing/upscale/file endpoint instead.")
    
    # Actually, let's create a better endpoint that accepts file upload
    raise HTTPException(status_code=501, detail="Use /api/processing/upscale/file endpoint for file upload")


@app.post("/api/processing/upscale/file")
async def process_upscale_file(
    file: UploadFile = File(...),
    model_name: str = Form("Real-ESRGAN"),
    scale: int = Form(4),
    device: Optional[str] = Form(None),
    output_format: Optional[str] = Form(None),
    jpg_quality: Optional[int] = Form(None)
):
    """Upscale an uploaded image or video file"""
    _load_processing()
    get_model_path = _get_upscale_model
    list_models = _list_available_upscale_models
    upscale_image_fn = _upscale_image
    upscale_video_fn = _upscale_video
    if (
        not PROCESSING_AVAILABLE
        or get_model_path is None
        or list_models is None
        or upscale_image_fn is None
        or upscale_video_fn is None
    ):
        raise HTTPException(status_code=503, detail="Processing not available")
    
    # Reject new work while the app is shutting down (window closed)
    if _shutdown_requested:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    
    # Check file type
    allowed_image = {'image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'}
    allowed_video = {'video/mp4', 'video/mov', 'video/quicktime', 'video/avi', 'video/mkv', 'video/webm'}
    
    is_image = file.content_type in allowed_image
    is_video = file.content_type in allowed_video
    
    if not (is_image or is_video):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    # scale=1 is a passthrough — no model required
    passthrough = (scale == 1)

    # Get model path (only needed when actually upscaling)
    model_path = None
    if not passthrough:
        model_path = get_model_path(model_name)
        if not model_path:
            available = list_models()
            downloaded = [m for m in available if m['downloaded']]
            if downloaded:
                model_path = downloaded[0]['path']
                model_name = downloaded[0]['name']
            else:
                raise HTTPException(status_code=400, detail="No upscale models downloaded. Set scale to 1x or download a model first.")
    
    import uuid
    
    saved_settings = {**DEFAULT_SETTINGS, **_read_saved_settings()}
    configured_backend = saved_settings.get("computeBackend", "auto")
    configured_device = str(saved_settings.get("gpuDevice", "auto"))
    if device and device != "auto":
        selected_device = device
    elif configured_backend == "cpu":
        selected_device = "cpu"
    elif configured_backend == "cuda":
        selected_device = configured_device if configured_device.startswith("cuda") else "cuda"
    elif configured_backend == "metal":
        selected_device = configured_device if configured_device.startswith("mps") else "mps"
    elif configured_backend == "hip":
        selected_device = configured_device if configured_device.startswith(("hip", "cuda")) else "hip"
    elif configured_backend in ("amd", "dml"):
        selected_device = configured_device if configured_device.startswith(("privateuseone", "dml", "hip", "cuda")) else "dml"
    else:
        selected_device = configured_device if configured_device not in ("", "auto") else "auto"

    # Guardrail: the string handed to torch.device() must be a device that
    # exists on this machine. Legacy/mis-configured values such as "gpu:0"
    # (an OS adapter name, not a torch device) resolve to the best available
    # accelerator instead of crashing model loading.
    selected_device_raw = selected_device
    selected_device = system_resources.resolve_default_device(selected_device_raw)
    if selected_device != selected_device_raw:
        print(f"[api] Device '{selected_device_raw}' is not a usable torch device; using '{selected_device}'")

    selected_format = output_format or saved_settings.get("outputFormat", "jpg")
    valid_formats = {"source", "jpg", "png", "webp", "tiff", "mp4"}
    if selected_format not in valid_formats:
        raise HTTPException(status_code=400, detail=f"Unsupported output format: {selected_format}")
    if is_image and selected_format == "mp4":
        raise HTTPException(status_code=400, detail="MP4 output is only supported for video files")
    try:
        quality = int(jpg_quality if jpg_quality is not None else saved_settings.get("jpgQuality", 95))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="JPG quality must be an integer between 1 and 100") from exc
    if not 1 <= quality <= 100:
        raise HTTPException(status_code=400, detail="JPG quality must be between 1 and 100")

    # Tile size for the upscale job. The legacy default (400) is auto-tuned
    # to a VRAM-aware value on GPU runs (see resolve_tile_size); any other
    # explicit value is used as-is.
    tile_size = system_resources.resolve_tile_size(
        int(saved_settings.get("defaultTileSize") or saved_settings.get("tileSize") or 400),
        selected_device,
    )

    temp_dir = TEMP_PROCESSING_DIR
    temp_dir.mkdir(exist_ok=True)
    
    safe_filename = Path(file.filename or "upload").name
    input_filename = f"{uuid.uuid4()}_{safe_filename}"
    input_path = temp_dir / input_filename
    
    try:
        content = await file.read()
        with open(input_path, 'wb') as f:
            f.write(content)
        
        # Generate output path
        input_suffix = Path(safe_filename).suffix.lower()
        if is_video:
            output_suffix = ".mp4"
        else:
            output_suffix = {
                "jpg": ".jpg",
                "png": ".png",
                "webp": ".webp",
                "tiff": ".tiff"
            }.get(selected_format, input_suffix)
        output_filename = f"{uuid.uuid4()}_processed_{Path(safe_filename).stem}{output_suffix}"
        output_path = temp_dir / output_filename
        
        # Process in background
        def process_task(
            upscale_image_fn: UpscaleFunction = upscale_image_fn,
            upscale_video_fn: UpscaleFunction = upscale_video_fn,
        ):
            # Debug: check distutils in worker thread
            import sys
            print(f"Worker thread: distutils in sys.modules = {'distutils' in sys.modules}")
            if 'distutils' in sys.modules:
                print(f"Worker thread: distutils = {sys.modules['distutils']}")
            
            # Monkey patch distutils for Python 3.12+ in worker thread
            import sys
            try:
                from setuptools import _distutils
                class _DistutilsModule(ModuleType):
                    def __getattr__(self, name):
                        if hasattr(_distutils, name):
                            return getattr(_distutils, name)
                        try:
                            return __import__('setuptools._distutils.' + name, fromlist=[''])
                        except ImportError:
                            pass
                        raise AttributeError('module distutils has no attribute ' + name)
                sys.modules['distutils'] = _DistutilsModule("distutils")
                print("Worker thread: distutils monkey patch applied")
            except ImportError:
                pass
            
            try:
                if torch is not None:
                    torch.set_num_threads(max(1, int(saved_settings.get("cpuThreads", 8))))
                    if selected_device.startswith(("cuda", "hip")) and saved_settings.get("maxVramUsage", 0):
                        device_index = int(selected_device.split(":", 1)[1]) if ":" in selected_device else 0
                        # maxVramUsage is a percentage (0-100) of total device VRAM
                        fraction = min(
                            max(float(saved_settings.get("maxVramUsage", 95)) / 100.0, 0.0),
                            1.0
                        )
                        torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)

                # Progress callback to update task progress
                def progress_cb(p):
                    # Cooperative abort: the window was closed, stop the task now
                    if _shutdown_requested:
                        raise ProcessingAborted("Processing aborted: application is shutting down")
                    if task_id in process_progress:
                        process_progress[task_id]['progress'] = p
                        # Force yield to ensure visibility across threads
                        import time
                        time.sleep(0.001)

                # Lazy load processing modules
                _load_processing()
                if not PROCESSING_AVAILABLE:
                    raise RuntimeError("Processing modules not available")

                if passthrough:
                    # scale=1: just copy/convert without upscaling
                    import shutil
                    if is_image:
                        progress_cb(10)
                        # For images, we may want to re-encode with different format/quality
                        if selected_format != 'source' and selected_format != input_suffix.lstrip('.'):
                            # Need to re-encode
                            img = cv2.imread(str(input_path))
                            write_params = []
                            if selected_format in ('jpg', 'jpeg'):
                                write_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                            cv2.imwrite(str(output_path), img, write_params)
                        else:
                            shutil.copy2(input_path, output_path)
                        progress_cb(100)
                    else:
                        # Video passthrough - copy or re-encode to mp4
                        progress_cb(10)
                        if selected_format == 'mp4' or selected_format == 'source':
                            shutil.copy2(input_path, output_path)
                        else:
                            # Re-encode with cv2
                            cap = cv2.VideoCapture(str(input_path))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
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

                    result = {
                        'success': True,
                        'input_path': str(input_path),
                        'output_path': str(output_path),
                        'scale_factor': 1,
                        'model': 'passthrough',
                        'passthrough': True
                    }
                else:
                    # Normal upscale
                    if is_image:
                        result = upscale_image_fn(
                            str(input_path), str(output_path),
                            model_name=model_name, scale=scale, device=selected_device,
                            tile_size=tile_size,
                            fp16=bool(saved_settings.get("fp16Enabled", True)),
                            jpg_quality=quality,
                            progress_callback=progress_cb
                        )
                    else:
                        result = upscale_video_fn(
                            str(input_path), str(output_path),
                            model_name=model_name, scale=scale, device=selected_device,
                            tile_size=tile_size,
                            fp16=bool(saved_settings.get("fp16Enabled", True)),
                            jpg_quality=quality,
                            progress_callback=progress_cb
                        )

                process_progress[task_id] = {
                    'status': 'completed',
                    'result': result,
                    'progress': 100
                }
            except ProcessingAborted as e:
                print(f"[Worker] {e}", flush=True)
                # Remove the partially-written output file (if any)
                try:
                    if output_path.exists():
                        output_path.unlink()
                        print(f"[Worker] Removed partial output: {output_path.name}", flush=True)
                except Exception:
                    pass
                process_progress[task_id] = {
                    'status': 'aborted',
                    'error': str(e),
                    'progress': 0
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                process_progress[task_id] = {
                    'status': 'failed',
                    'error': str(e),
                    'progress': 0
                }
            finally:
                # Cleanup temp input
                try:
                    input_path.unlink()
                except:
                    pass
        
        task_id = str(uuid.uuid4())
        process_progress[task_id] = {'status': 'processing', 'progress': 0}
        
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, process_task)
        process_tasks[task_id] = task
        
        return {"task_id": task_id, "status": "started"}
        
    except Exception as e:
        # Cleanup on error
        try:
            input_path.unlink()
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/processing/task/{task_id}")
async def get_processing_status(task_id: str):
    """Get processing task status"""
    if task_id not in process_progress:
        raise HTTPException(status_code=404, detail="Task not found")
    return process_progress[task_id]


@app.get("/api/processing/result/{task_id}")
async def get_processing_result(task_id: str):
    """Get processing result file"""
    if task_id not in process_progress:
        raise HTTPException(status_code=404, detail="Task not found")
    
    progress = process_progress[task_id]
    if progress.get('status') != 'completed':
        raise HTTPException(status_code=400, detail="Processing not completed")
    
    result = progress.get('result', {})
    output_path = result.get('output_path')
    
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(output_path, filename=Path(output_path).name)


# ============================================================================
# COMPREHENSIVE PROCESSING ENDPOINTS
# ============================================================================

@app.get("/api/processing/features")
async def get_processing_features():
    """Get all available processing features with their models"""
    from src.processing import (
        UPSCALE_MODELS, FACE_MODELS, SHARPEN_MODELS, DENOISE_MODELS,
        FRAME_INTERP_MODELS, SLOWMO_MODELS, REENCODING_PRESETS
    )
    
    return {
        "upscale": {"models": list(UPSCALE_MODELS.keys())},
        "face_recovery": {"models": list(FACE_MODELS.keys())},
        "sharpen": {"models": list(SHARPEN_MODELS.keys())},
        "denoise": {"models": list(DENOISE_MODELS.keys())},
        "frame_interpolation": {"models": list(FRAME_INTERP_MODELS.keys())},
        "slow_motion": {"models": list(SLOWMO_MODELS.keys())},
        "colorization": {"models": ["DeOldify"]},
        "match_color": {"models": ["Reinhard Transfer", "WCT2", "NLUT", "Deep Color Transfer"]},
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
        }
    }


@app.post("/api/processing/face-recovery/file")
@app.post("/api/processing/face/file")
async def process_face_recovery_file(
    file: UploadFile = File(...),
    model_name: str = Form("CodeFormer"),
    fidelity_weight: float = Form(0.5),
    device: Optional[str] = Form(None),
):
    """Apply face recovery to uploaded image or video"""
    from src.processing import FaceRecoveryPipeline, get_face_model
    
    _load_processing()
    if not PROCESSING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Processing not available")
    
    if _shutdown_requested:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    
    # Save uploaded file
    temp_dir = TEMP_PROCESSING_DIR
    temp_dir.mkdir(exist_ok=True)
    
    safe_filename = Path(file.filename or "upload").name
    input_filename = f"{uuid.uuid4()}_{safe_filename}"
    input_path = temp_dir / input_filename
    
    content = await file.read()
    with open(input_path, 'wb') as f:
        f.write(content)
    
    # Determine output path
    output_filename = f"{uuid.uuid4()}_face_restored_{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    output_path = temp_dir / output_filename
    
    # Get selected device
    saved_settings = {**DEFAULT_SETTINGS, **_read_saved_settings()}
    selected_device = system_resources.resolve_default_device(
        device or saved_settings.get("computeBackend", "auto")
    )
    
    def process_task():
        try:
            pipeline = FaceRecoveryPipeline(
                model_name=model_name,
                device=selected_device,
                fidelity_weight=fidelity_weight
            )
            pipeline.load_model(get_face_model(model_name))
            result = pipeline.process(str(input_path), str(output_path))
            process_progress[task_id] = {'status': 'completed', 'result': result, 'progress': 100}
        except ProcessingAborted as e:
            process_progress[task_id] = {'status': 'aborted', 'error': str(e), 'progress': 0}
        except Exception as e:
            process_progress[task_id] = {'status': 'failed', 'error': str(e), 'progress': 0}
        finally:
            try:
                input_path.unlink()
            except:
                pass
    
    task_id = str(uuid.uuid4())
    process_progress[task_id] = {'status': 'processing', 'progress': 0}
    
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, process_task)
    process_tasks[task_id] = task
    
    return {"task_id": task_id, "status": "started"}


@app.post("/api/processing/sharpen/file")
async def process_sharpen_file(
    file: UploadFile = File(...),
    model_name: str = Form("NAFNet"),
    strength: float = Form(1.0),
    overlap: Optional[int] = Form(None),
    unsharp_amount: Optional[float] = Form(None),
    device: Optional[str] = Form(None),
):
    """Apply sharpening/deblurring to uploaded image or video"""
    from src.processing import SharpenPipeline, get_sharpen_model, list_sharpen_models
    
    _load_processing()
    if not PROCESSING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Processing not available")
    
    if _shutdown_requested:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    
    saved_settings = {**DEFAULT_SETTINGS, **_read_saved_settings()}
    if overlap is None:
        overlap = int(saved_settings.get("sharpenOverlap", 128))
    if unsharp_amount is None:
        unsharp_amount = float(saved_settings.get("sharpenUnsharpAmount", 0.5))

    # Fail fast instead of silently falling back to classical sharpening:
    # a missing weights file must never produce a "sharpened" but still blurry
    # image without the user knowing.
    model_path = get_sharpen_model(model_name)
    if not model_path:
        available = list_sharpen_models()
        names = [m["name"] for m in available if m["downloaded"]]
        detail = (
            f"Sharpening model '{model_name}' is not downloaded. "
            + (f"Installed: {', '.join(names)}." if names
               else "No sharpen models are installed - download one from the model manager.")
        )
        raise HTTPException(status_code=400, detail=detail)
    
    temp_dir = TEMP_PROCESSING_DIR
    temp_dir.mkdir(exist_ok=True)
    
    safe_filename = Path(file.filename or "upload").name
    input_filename = f"{uuid.uuid4()}_{safe_filename}"
    input_path = temp_dir / input_filename
    
    content = await file.read()
    with open(input_path, 'wb') as f:
        f.write(content)
    
    output_filename = f"{uuid.uuid4()}_sharpened_{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    output_path = temp_dir / output_filename
    
    selected_device = system_resources.resolve_default_device(
        device or saved_settings.get("computeBackend", "auto")
    )
    
    def process_task():
        try:
            pipeline = SharpenPipeline(model_name=model_name, device=selected_device,
                                       strength=strength, overlap=overlap,
                                       unsharp_amount=unsharp_amount)
            pipeline.load_model(model_path)
            result = pipeline.process(str(input_path), str(output_path))
            result.setdefault('overlap', overlap)
            result.setdefault('window', 'hamming')
            result.setdefault('unsharp_amount', unsharp_amount)
            result.setdefault('model_path', model_path)
            process_progress[task_id] = {'status': 'completed', 'result': result, 'progress': 100}
        except ProcessingAborted as e:
            process_progress[task_id] = {'status': 'aborted', 'error': str(e), 'progress': 0}
        except Exception as e:
            process_progress[task_id] = {'status': 'failed', 'error': str(e), 'progress': 0}
        finally:
            try:
                input_path.unlink()
            except:
                pass
    
    task_id = str(uuid.uuid4())
    process_progress[task_id] = {'status': 'processing', 'progress': 0}
    
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, process_task)
    process_tasks[task_id] = task
    
    return {"task_id": task_id, "status": "started"}


@app.post("/api/processing/denoise/file")
async def process_denoise_file(
    file: UploadFile = File(...),
    model_name: str = Form("SCUNet"),
    strength: float = Form(0.5),
    device: Optional[str] = Form(None),
):
    """Apply denoising to uploaded image or video"""
    from src.processing import DenoisePipeline, get_denoise_model
    
    _load_processing()
    if not PROCESSING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Processing not available")
    
    if _shutdown_requested:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    
    temp_dir = TEMP_PROCESSING_DIR
    temp_dir.mkdir(exist_ok=True)
    
    safe_filename = Path(file.filename or "upload").name
    input_filename = f"{uuid.uuid4()}_{safe_filename}"
    input_path = temp_dir / input_filename
    
    content = await file.read()
    with open(input_path, 'wb') as f:
        f.write(content)
    
    output_filename = f"{uuid.uuid4()}_denoised_{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    output_path = temp_dir / output_filename
    
    saved_settings = {**DEFAULT_SETTINGS, **_read_saved_settings()}
    selected_device = system_resources.resolve_default_device(
        device or saved_settings.get("computeBackend", "auto")
    )
    
    def process_task():
        try:
            pipeline = DenoisePipeline(model_name=model_name, device=selected_device, strength=strength)
            pipeline.load_model(get_denoise_model(model_name))
            result = pipeline.process(str(input_path), str(output_path))
            process_progress[task_id] = {'status': 'completed', 'result': result, 'progress': 100}
        except ProcessingAborted as e:
            process_progress[task_id] = {'status': 'aborted', 'error': str(e), 'progress': 0}
        except Exception as e:
            process_progress[task_id] = {'status': 'failed', 'error': str(e), 'progress': 0}
        finally:
            try:
                input_path.unlink()
            except:
                pass
    
    task_id = str(uuid.uuid4())
    process_progress[task_id] = {'status': 'processing', 'progress': 0}
    
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, process_task)
    process_tasks[task_id] = task
    
    return {"task_id": task_id, "status": "started"}


@app.post("/api/processing/frame-interpolation/file")
async def process_frame_interpolation_file(
    file: UploadFile = File(...),
    model_name: str = Form("RIFE"),
    multiplier: int = Form(2),
    device: Optional[str] = Form(None),
):
    """Apply frame interpolation to uploaded video"""
    from src.processing import FrameInterpolationPipeline, get_frame_interp_model
    
    _load_processing()
    if not PROCESSING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Processing not available")
    
    if _shutdown_requested:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    
    # Check file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Frame interpolation only supports video files")
    
    temp_dir = TEMP_PROCESSING_DIR
    temp_dir.mkdir(exist_ok=True)
    
    safe_filename = Path(file.filename or "upload").name
    input_filename = f"{uuid.uuid4()}_{safe_filename}"
    input_path = temp_dir / input_filename
    
    content = await file.read()
    with open(input_path, 'wb') as f:
        f.write(content)
    
    output_filename = f"{uuid.uuid4()}_interp_{multiplier}x_{Path(safe_filename).stem}.mp4"
    output_path = temp_dir / output_filename
    
    saved_settings = {**DEFAULT_SETTINGS, **_read_saved_settings()}
    selected_device = system_resources.resolve_default_device(
        device or saved_settings.get("computeBackend", "auto")
    )
    
    def process_task():
        try:
            pipeline = FrameInterpolationPipeline(
                model_name=model_name,
                device=selected_device,
                multiplier=multiplier
            )
            pipeline.load_model(get_frame_interp_model(model_name))
            result = pipeline.process(str(input_path), str(output_path))
            process_progress[task_id] = {'status': 'completed', 'result': result, 'progress': 100}
        except ProcessingAborted as e:
            process_progress[task_id] = {'status': 'aborted', 'error': str(e), 'progress': 0}
        except Exception as e:
            process_progress[task_id] = {'status': 'failed', 'error': str(e), 'progress': 0}
        finally:
            try:
                input_path.unlink()
            except:
                pass
    
    task_id = str(uuid.uuid4())
    process_progress[task_id] = {'status': 'processing', 'progress': 0}
    
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, process_task)
    process_tasks[task_id] = task
    
    return {"task_id": task_id, "status": "started"}


@app.post("/api/processing/slow-motion/file")
async def process_slow_motion_file(
    file: UploadFile = File(...),
    model_name: str = Form("RIFE-SloMo"),
    slowdown_factor: float = Form(4.0),
    device: Optional[str] = Form(None),
):
    """Create slow-motion video from uploaded video"""
    from src.processing import SlowMotionPipeline
    
    _load_processing()
    if not PROCESSING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Processing not available")
    
    if _shutdown_requested:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Slow motion only supports video files")
    
    temp_dir = TEMP_PROCESSING_DIR
    temp_dir.mkdir(exist_ok=True)
    
    safe_filename = Path(file.filename or "upload").name
    input_filename = f"{uuid.uuid4()}_{safe_filename}"
    input_path = temp_dir / input_filename
    
    content = await file.read()
    with open(input_path, 'wb') as f:
        f.write(content)
    
    output_filename = f"{uuid.uuid4()}_slowmo_{slowdown_factor}x_{Path(safe_filename).stem}.mp4"
    output_path = temp_dir / output_filename
    
    saved_settings = {**DEFAULT_SETTINGS, **_read_saved_settings()}
    selected_device = system_resources.resolve_default_device(
        device or saved_settings.get("computeBackend", "auto")
    )
    
    def process_task():
        try:
            pipeline = SlowMotionPipeline(
                model_name=model_name,
                device=selected_device,
                slowdown_factor=slowdown_factor
            )
            pipeline.load_model()
            result = pipeline.process(str(input_path), str(output_path))
            process_progress[task_id] = {'status': 'completed', 'result': result, 'progress': 100}
        except ProcessingAborted as e:
            process_progress[task_id] = {'status': 'aborted', 'error': str(e), 'progress': 0}
        except Exception as e:
            process_progress[task_id] = {'status': 'failed', 'error': str(e), 'progress': 0}
        finally:
            try:
                input_path.unlink()
            except:
                pass
    
    task_id = str(uuid.uuid4())
    process_progress[task_id] = {'status': 'processing', 'progress': 0}
    
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, process_task)
    process_tasks[task_id] = task
    
    return {"task_id": task_id, "status": "started"}


@app.post("/api/processing/match-color/file")
async def process_match_color_file(
    file: UploadFile = File(...),
    reference: Optional[UploadFile] = File(None),
    model_name: str = Form("Reinhard Transfer"),
    strength: float = Form(1.0),
    preserve_luminance: bool = Form(True),
    device: Optional[str] = Form(None),
):
    """Match colors from reference image to target"""
    from src.processing import MatchColorPipeline, get_match_color_model
    
    _load_processing()
    if not PROCESSING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Processing not available")
    
    if _shutdown_requested:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    
    temp_dir = TEMP_PROCESSING_DIR
    temp_dir.mkdir(exist_ok=True)
    
    # Save target file
    safe_filename = Path(file.filename or "upload").name
    input_filename = f"{uuid.uuid4()}_{safe_filename}"
    input_path = temp_dir / input_filename
    
    content = await file.read()
    with open(input_path, 'wb') as f:
        f.write(content)
    
    # Save reference file if provided
    ref_path = None
    if reference:
        ref_filename = f"{uuid.uuid4()}_ref_{Path(reference.filename or 'reference').name}"
        ref_path = temp_dir / ref_filename
        ref_content = await reference.read()
        with open(ref_path, 'wb') as f:
            f.write(ref_content)
    
    output_filename = f"{uuid.uuid4()}_color_matched_{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    output_path = temp_dir / output_filename

    saved_settings = {**DEFAULT_SETTINGS, **_read_saved_settings()}
    selected_device = system_resources.resolve_default_device(
        device or saved_settings.get("computeBackend", "auto")
    )

    def process_task():
        try:
            pipeline = MatchColorPipeline(
                model_name=model_name,
                device=selected_device,
                strength=strength,
                preserve_luminance=preserve_luminance,
            )
            pipeline.load_model(get_match_color_model(model_name))
            result = pipeline.process(
                str(input_path),
                str(output_path),
                reference_path=str(ref_path) if ref_path else None
            )
            process_progress[task_id] = {'status': 'completed', 'result': result, 'progress': 100}
        except ProcessingAborted as e:
            process_progress[task_id] = {'status': 'aborted', 'error': str(e), 'progress': 0}
        except Exception as e:
            process_progress[task_id] = {'status': 'failed', 'error': str(e), 'progress': 0}
        finally:
            try:
                input_path.unlink()
                if ref_path:
                    ref_path.unlink()
            except:
                pass
    
    task_id = str(uuid.uuid4())
    process_progress[task_id] = {'status': 'processing', 'progress': 0}
    
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, process_task)
    process_tasks[task_id] = task
    
    return {"task_id": task_id, "status": "started"}


@app.post("/api/processing/low-light/file")
async def process_low_light_file(
    file: UploadFile = File(...),
    model_name: str = Form("Zero-DCE"),
    strength: float = Form(0.7),
    device: Optional[str] = Form(None),
):
    """Enhance low-light images or videos"""
    from src.processing import LowLightEnhancementPipeline
    
    _load_processing()
    if not PROCESSING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Processing not available")
    
    if _shutdown_requested:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    
    temp_dir = TEMP_PROCESSING_DIR
    temp_dir.mkdir(exist_ok=True)
    
    safe_filename = Path(file.filename or "upload").name
    input_filename = f"{uuid.uuid4()}_{safe_filename}"
    input_path = temp_dir / input_filename
    
    content = await file.read()
    with open(input_path, 'wb') as f:
        f.write(content)
    
    output_filename = f"{uuid.uuid4()}_lowlight_{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    output_path = temp_dir / output_filename
    
    def process_task():
        try:
            pipeline = LowLightEnhancementPipeline(model_name=model_name, strength=strength)
            pipeline.load_model()
            result = pipeline.process(str(input_path), str(output_path))
            process_progress[task_id] = {'status': 'completed', 'result': result, 'progress': 100}
        except ProcessingAborted as e:
            process_progress[task_id] = {'status': 'aborted', 'error': str(e), 'progress': 0}
        except Exception as e:
            process_progress[task_id] = {'status': 'failed', 'error': str(e), 'progress': 0}
        finally:
            try:
                input_path.unlink()
            except:
                pass
    
    task_id = str(uuid.uuid4())
    process_progress[task_id] = {'status': 'processing', 'progress': 0}
    
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, process_task)
    process_tasks[task_id] = task
    
    return {"task_id": task_id, "status": "started"}


@app.post("/api/processing/log-restoration/file")
async def process_log_restoration_file(
    file: UploadFile = File(...),
    model_name: str = Form("ITMLUT"),
    color_space_target: str = Form("Rec.709"),
    device: Optional[str] = Form(None),
):
    """Restore LOG footage to standard color space"""
    from src.processing import LogRestorationPipeline
    
    _load_processing()
    if not PROCESSING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Processing not available")
    
    if _shutdown_requested:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    
    temp_dir = TEMP_PROCESSING_DIR
    temp_dir.mkdir(exist_ok=True)
    
    safe_filename = Path(file.filename or "upload").name
    input_filename = f"{uuid.uuid4()}_{safe_filename}"
    input_path = temp_dir / input_filename
    
    content = await file.read()
    with open(input_path, 'wb') as f:
        f.write(content)
    
    output_filename = f"{uuid.uuid4()}_log_restored_{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    output_path = temp_dir / output_filename
    
    def process_task():
        try:
            pipeline = LogRestorationPipeline(
                model_name=model_name,
                color_space_target=color_space_target
            )
            pipeline.load_model()
            result = pipeline.process(str(input_path), str(output_path))
            process_progress[task_id] = {'status': 'completed', 'result': result, 'progress': 100}
        except ProcessingAborted as e:
            process_progress[task_id] = {'status': 'aborted', 'error': str(e), 'progress': 0}
        except Exception as e:
            process_progress[task_id] = {'status': 'failed', 'error': str(e), 'progress': 0}
        finally:
            try:
                input_path.unlink()
            except:
                pass
    
    task_id = str(uuid.uuid4())
    process_progress[task_id] = {'status': 'processing', 'progress': 0}
    
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, process_task)
    process_tasks[task_id] = task
    
    return {"task_id": task_id, "status": "started"}


@app.post("/api/processing/sdr-to-hdr/file")
async def process_sdr_to_hdr_file(
    file: UploadFile = File(...),
    model_name: str = Form("HDRNet"),
    target_format: str = Form("HDR10"),
    device: Optional[str] = Form(None),
):
    """Convert SDR to HDR"""
    from src.processing import SDRtoHDRPipeline
    
    _load_processing()
    if not PROCESSING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Processing not available")
    
    if _shutdown_requested:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    
    temp_dir = TEMP_PROCESSING_DIR
    temp_dir.mkdir(exist_ok=True)
    
    safe_filename = Path(file.filename or "upload").name
    input_filename = f"{uuid.uuid4()}_{safe_filename}"
    input_path = temp_dir / input_filename
    
    content = await file.read()
    with open(input_path, 'wb') as f:
        f.write(content)
    
    output_filename = f"{uuid.uuid4()}_hdr_{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    output_path = temp_dir / output_filename
    
    def process_task():
        try:
            pipeline = SDRtoHDRPipeline(model_name=model_name, target_format=target_format)
            pipeline.load_model()
            result = pipeline.process(str(input_path), str(output_path))
            process_progress[task_id] = {'status': 'completed', 'result': result, 'progress': 100}
        except ProcessingAborted as e:
            process_progress[task_id] = {'status': 'aborted', 'error': str(e), 'progress': 0}
        except Exception as e:
            process_progress[task_id] = {'status': 'failed', 'error': str(e), 'progress': 0}
        finally:
            try:
                input_path.unlink()
            except:
                pass
    
    task_id = str(uuid.uuid4())
    process_progress[task_id] = {'status': 'processing', 'progress': 0}
    
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, process_task)
    process_tasks[task_id] = task
    
    return {"task_id": task_id, "status": "started"}


@app.post("/api/processing/overexposure-fix/file")
async def process_overexposure_fix_file(
    file: UploadFile = File(...),
    model_name: str = Form("AutoEnhance"),
    device: Optional[str] = Form(None),
):
    """Fix overexposure and contrast issues"""
    from src.processing import OverexposureFixPipeline
    
    _load_processing()
    if not PROCESSING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Processing not available")
    
    if _shutdown_requested:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    
    temp_dir = TEMP_PROCESSING_DIR
    temp_dir.mkdir(exist_ok=True)
    
    safe_filename = Path(file.filename or "upload").name
    input_filename = f"{uuid.uuid4()}_{safe_filename}"
    input_path = temp_dir / input_filename
    
    content = await file.read()
    with open(input_path, 'wb') as f:
        f.write(content)
    
    output_filename = f"{uuid.uuid4()}_fixed_{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    output_path = temp_dir / output_filename
    
    def process_task():
        try:
            pipeline = OverexposureFixPipeline(model_name=model_name)
            pipeline.load_model()
            result = pipeline.process(str(input_path), str(output_path))
            process_progress[task_id] = {'status': 'completed', 'result': result, 'progress': 100}
        except ProcessingAborted as e:
            process_progress[task_id] = {'status': 'aborted', 'error': str(e), 'progress': 0}
        except Exception as e:
            process_progress[task_id] = {'status': 'failed', 'error': str(e), 'progress': 0}
        finally:
            try:
                input_path.unlink()
            except:
                pass
    
    task_id = str(uuid.uuid4())
    process_progress[task_id] = {'status': 'processing', 'progress': 0}
    
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, process_task)
    process_tasks[task_id] = task
    
    return {"task_id": task_id, "status": "started"}


@app.post("/api/processing/flat-profile/file")
async def process_flat_profile_file(
    file: UploadFile = File(...),
    model_name: str = Form("ColorGrade"),
    device: Optional[str] = Form(None),
):
    """Apply flat color profile for color correction"""
    from src.processing import FlatColorProfilePipeline
    
    _load_processing()
    if not PROCESSING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Processing not available")
    
    if _shutdown_requested:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    
    temp_dir = TEMP_PROCESSING_DIR
    temp_dir.mkdir(exist_ok=True)
    
    safe_filename = Path(file.filename or "upload").name
    input_filename = f"{uuid.uuid4()}_{safe_filename}"
    input_path = temp_dir / input_filename
    
    content = await file.read()
    with open(input_path, 'wb') as f:
        f.write(content)
    
    output_filename = f"{uuid.uuid4()}_flat_{Path(safe_filename).stem}{Path(safe_filename).suffix}"
    output_path = temp_dir / output_filename
    
    def process_task():
        try:
            pipeline = FlatColorProfilePipeline(model_name=model_name)
            pipeline.load_model()
            result = pipeline.process(str(input_path), str(output_path))
            process_progress[task_id] = {'status': 'completed', 'result': result, 'progress': 100}
        except ProcessingAborted as e:
            process_progress[task_id] = {'status': 'aborted', 'error': str(e), 'progress': 0}
        except Exception as e:
            process_progress[task_id] = {'status': 'failed', 'error': str(e), 'progress': 0}
        finally:
            try:
                input_path.unlink()
            except:
                pass
    
    task_id = str(uuid.uuid4())
    process_progress[task_id] = {'status': 'processing', 'progress': 0}
    
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, process_task)
    process_tasks[task_id] = task
    
    return {"task_id": task_id, "status": "started"}


@app.post("/api/processing/reencode/file")
async def process_reencode_file(
    file: UploadFile = File(...),
    engine: str = Form("ffmpeg"),
    preset: str = Form("h264_fast"),
    device: Optional[str] = Form(None),
):
    """Reencode video using FFMPEG or Handbrake"""
    from src.processing import ReencodingPipeline
    
    _load_processing()
    if not PROCESSING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Processing not available")
    
    if _shutdown_requested:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Reencoding only supports video files")
    
    temp_dir = TEMP_PROCESSING_DIR
    temp_dir.mkdir(exist_ok=True)
    
    safe_filename = Path(file.filename or "upload").name
    input_filename = f"{uuid.uuid4()}_{safe_filename}"
    input_path = temp_dir / input_filename
    
    content = await file.read()
    with open(input_path, 'wb') as f:
        f.write(content)
    
    output_filename = f"{uuid.uuid4()}_reencoded_{Path(safe_filename).stem}.mp4"
    output_path = temp_dir / output_filename
    
    def process_task():
        try:
            pipeline = ReencodingPipeline(engine=engine, preset=preset)
            pipeline.load_model()
            result = pipeline.process(str(input_path), str(output_path))
            process_progress[task_id] = {'status': 'completed', 'result': result, 'progress': 100}
        except ProcessingAborted as e:
            process_progress[task_id] = {'status': 'aborted', 'error': str(e), 'progress': 0}
        except Exception as e:
            process_progress[task_id] = {'status': 'failed', 'error': str(e), 'progress': 0}
        finally:
            try:
                input_path.unlink()
            except:
                pass
    
    task_id = str(uuid.uuid4())
    process_progress[task_id] = {'status': 'processing', 'progress': 0}
    
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, process_task)
    process_tasks[task_id] = task
    
    return {"task_id": task_id, "status": "started"}


# Serve static UI files
ui_path = _paths.ui_dir()
if ui_path.exists():
    # Mount UI folder for direct access to CSS, JS, etc. at /ui/
    app.mount("/ui", StaticFiles(directory=str(ui_path)), name="ui")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the main UI page"""
    index_file = ui_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>UI not found</h1><p>Please run from project root</p>")


@app.get("/model-manager", response_class=HTMLResponse)
async def serve_model_manager():
    """Serve the model manager UI"""
    manager_file = ui_path / "model-manager.html"
    if manager_file.exists():
        return FileResponse(manager_file)
    return HTMLResponse("<h1>Model Manager UI not found</h1>")


@app.get("/settings", response_class=HTMLResponse)
async def serve_settings():
    """Serve the settings UI"""
    settings_file = ui_path / "settings.html"
    if settings_file.exists():
        return FileResponse(settings_file)
    return HTMLResponse("<h1>Settings UI not found</h1>")


# Settings API endpoints


def _read_saved_settings() -> Dict[str, Any]:
    if not SETTINGS_FILE.exists():
        return {}
    try:
        settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read settings.json: {exc}") from exc
    if not isinstance(settings, dict):
        raise HTTPException(status_code=500, detail="settings.json must contain a JSON object")
    return settings


def _write_saved_settings(settings: Dict[str, Any]) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not save settings.json: {exc}") from exc


@app.get("/api/settings")
async def get_settings():
    """Get persisted settings; clients layer defaults over this partial object."""
    return _read_saved_settings()


@app.post("/api/settings")
async def update_settings(request: SettingsModel):
    """Update settings"""
    current = _read_saved_settings()
    update_data = request.model_dump(exclude_unset=True)
    current.update(update_data)
    _write_saved_settings(current)
    
    return {"message": "Settings updated", "settings": current}


@app.post("/api/settings/model-dir")
async def update_model_dir(request: dict):
    """Update model directory"""
    path = request.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="Path required")
    
    # Update settings
    current = _read_saved_settings()
    current["modelDir"] = path
    _write_saved_settings(current)
    
    # Update model manager
    manager = get_model_manager()
    manager.downloader.models_dir = Path(path)
    manager.downloader.models_dir.mkdir(parents=True, exist_ok=True)
    
    return {"message": "Model directory updated", "path": path}


@app.post("/api/settings/export-dir")
async def update_export_dir(request: dict):
    """Update standard export directory"""
    path = request.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="Path required")
    
    current = _read_saved_settings()
    current["exportDir"] = path
    _write_saved_settings(current)
    
    Path(path).mkdir(parents=True, exist_ok=True)
    return {"message": "Export directory updated", "path": path}


@app.delete("/api/models/cache")
async def clear_model_cache():
    """Clear model cache (temp files)"""
    if TEMP_PROCESSING_DIR.exists():
        for entry in TEMP_PROCESSING_DIR.iterdir():
            try:
                if entry.is_symlink() or entry.is_file():
                    entry.unlink()
                elif entry.is_dir():
                    shutil.rmtree(entry)
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Could not clear cache entry {entry.name}: {exc}") from exc
    return {"message": "Cache cleared"}


@app.post("/api/models/verify")
async def verify_all_models():
    """Verify all downloaded models"""
    manager = get_model_manager()
    downloaded = manager.downloader.get_downloaded_models()
    
    verified = 0
    failed = 0
    for model in downloaded:
        if manager.downloader.verify_model(model):
            verified += 1
        else:
            failed += 1
    
    return {"verified": verified, "failed": failed}


# System info endpoints
@app.get("/api/system/info")
async def get_system_info():
    """Get system hardware information"""
    return system_resources.get_system_hardware_info()


@app.get("/api/system/gpus")
async def get_gpus():
    """Get list of available GPUs"""
    return {"gpus": system_resources.get_hardware_gpus()}


@app.get("/api/system/capabilities")
async def get_system_capabilities():
    """Get platform capabilities for model compatibility"""
    return system_resources.get_platform_capabilities()



# Catch-all for SPA-style routing (must be last)
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """Serve index.html for any unmatched routes (SPA fallback)"""
    # Only serve index.html for non-API, non-static paths
    if full_path.startswith("api/") or full_path.startswith("ui/"):
        raise HTTPException(status_code=404, detail="Not found")
    
    index_file = ui_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>UI not found</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
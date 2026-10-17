"""
Processing API routes.
"""
import asyncio
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path

from ...api.schemas import (
    ProcessRequest,
    ProcessResponse,
    TaskStatusResponse,
    TaskStatus,
    ProcessingFeaturesResponse,
    FeatureInfo,
    PipelineStep,
    UpscaleParameters,
    FaceRecoveryParameters,
    SharpenParameters,
    DenoiseParameters,
    FrameInterpolationParameters,
    MatchColorParameters,
    LowLightParameters,
    SDRToHDRParameters,
    ReencodingParameters,
)
from ...api.dependencies import (
    get_processing_service,
    get_pipeline_registry,
    get_model_registry,
    validate_feature_available,
    validate_pipeline_step,
)
from ...services.processing_service import ProcessingService
from ...processing.pipeline.registry import PipelineRegistry
from ...models.registry import ModelRegistry, default_model

router = APIRouter(prefix="/processing", tags=["processing"])


@router.get("/features", response_model=ProcessingFeaturesResponse)
async def get_processing_features(
    pipeline_registry: PipelineRegistry = Depends(get_pipeline_registry),
    model_registry: ModelRegistry = Depends(get_model_registry),
):
    """Get all available processing features with their models."""
    from ...services.settings_service import experimental_features_enabled

    include_experimental = experimental_features_enabled()
    features = {}

    for feature_name in pipeline_registry.get_all_features():
        if not include_experimental and model_registry.is_feature_experimental(feature_name):
            continue
        info = pipeline_registry.get_pipeline_info(feature_name)
        if info:
            # Get downloaded models for this feature
            downloaded = model_registry.get_downloaded_for_feature(feature_name)
            downloaded_names = [m.config.name for m in downloaded]

            available = info.available_models
            if not include_experimental:
                available = model_registry.drop_experimental(available)
            if not available:
                continue
            default_model_name = info.default_model
            if default_model_name not in available:
                default_model_name = available[0]

            features[feature_name] = FeatureInfo(
                name=feature_name,
                description=info.config.description,
                implemented=True,
                available_models=available,
                default_model=default_model_name,
                parameters_schema=info.config.parameter_schema,
            )

    return ProcessingFeaturesResponse(features=features)


@router.post("", response_model=ProcessResponse)
async def start_processing(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    processing_service: ProcessingService = Depends(get_processing_service),
):
    """Start a new processing task."""
    # Validate input file
    input_path = Path(request.input_path)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Input file not found: {request.input_path}")

    # Validate pipeline steps
    if request.steps:
        for step in request.steps:
            await validate_pipeline_step(step.model_dump())

    # Create and start task
    task = processing_service.create_task(request)
    response = await processing_service.start_processing(task.task_id, request)
    return response


@router.post("/upscale/file", response_model=ProcessResponse)
async def upscale_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form(default_model("Upscale")),
    scale: int = Form(4),
    device: Optional[str] = Form("auto"),
    tile_size: int = Form(400),
    fp16: bool = Form(True),
    output_format: str = Form("source"),
    processing_service: ProcessingService = Depends(get_processing_service),
    model_registry: ModelRegistry = Depends(get_model_registry),
):
    """Upscale an uploaded file."""
    # Save uploaded file
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    input_path = temp_dir / f"upload_{file.filename}"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Build request
    request = ProcessRequest(
        input_path=str(input_path),
        steps=[
            PipelineStep(
                feature="Upscale",
                model=model,
                parameters={
                    "scale": scale,
                    "device": device,
                    "tile_size": tile_size,
                    "fp16": fp16,
                }
            )
        ],
        output_format=output_format,
    )

    task = processing_service.create_task(request)
    response = await processing_service.start_processing(task.task_id, request)
    
    # Schedule cleanup of temp file after processing
    background_tasks.add_task(lambda p: p.unlink(missing_ok=True), input_path)
    return response


@router.post("/face-recovery/file", response_model=ProcessResponse)
async def face_recovery_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form(default_model("Face Recovery")),
    fidelity_weight: float = Form(0.5),
    device: Optional[str] = Form("auto"),
    processing_service: ProcessingService = Depends(get_processing_service),
):
    """Apply face recovery to uploaded file."""
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    input_path = temp_dir / f"upload_{file.filename}"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    request = ProcessRequest(
        input_path=str(input_path),
        steps=[
            PipelineStep(
                feature="Face Recovery",
                model=model,
                parameters={
                    "fidelity_weight": fidelity_weight,
                    "device": device,
                }
            )
        ],
    )

    task = processing_service.create_task(request)
    response = await processing_service.start_processing(task.task_id, request)
    background_tasks.add_task(lambda p: p.unlink(missing_ok=True), input_path)
    return response


@router.post("/sharpen/file", response_model=ProcessResponse)
async def sharpen_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form(default_model("Sharpen / Deblur")),
    strength: float = Form(1.0),
    overlap: int = Form(128),
    unsharp_amount: float = Form(0.5),
    device: Optional[str] = Form("auto"),
    processing_service: ProcessingService = Depends(get_processing_service),
):
    """Apply sharpening/deblurring to uploaded file."""
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    input_path = temp_dir / f"upload_{file.filename}"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    request = ProcessRequest(
        input_path=str(input_path),
        steps=[
            PipelineStep(
                feature="Sharpen / Deblur",
                model=model,
                parameters={
                    "strength": strength,
                    "overlap": overlap,
                    "unsharp_amount": unsharp_amount,
                    "device": device,
                }
            )
        ],
    )

    task = processing_service.create_task(request)
    response = await processing_service.start_processing(task.task_id, request)
    background_tasks.add_task(lambda p: p.unlink(missing_ok=True), input_path)
    return response


@router.post("/denoise/file", response_model=ProcessResponse)
async def denoise_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form(default_model("Denoise")),
    strength: float = Form(0.5),
    device: Optional[str] = Form("auto"),
    processing_service: ProcessingService = Depends(get_processing_service),
):
    """Apply denoising to uploaded file."""
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    input_path = temp_dir / f"upload_{file.filename}"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    request = ProcessRequest(
        input_path=str(input_path),
        steps=[
            PipelineStep(
                feature="Denoise",
                model=model,
                parameters={
                    "strength": strength,
                    "device": device,
                }
            )
        ],
    )

    task = processing_service.create_task(request)
    response = await processing_service.start_processing(task.task_id, request)
    background_tasks.add_task(lambda p: p.unlink(missing_ok=True), input_path)
    return response


@router.post("/frame-interpolation/file", response_model=ProcessResponse)
async def frame_interpolation_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form(default_model("Frame Interpolation")),
    multiplier: int = Form(2),
    slowmo_factor: float = Form(1.0),
    device: Optional[str] = Form("auto"),
    processing_service: ProcessingService = Depends(get_processing_service),
):
    """Apply frame interpolation to uploaded video."""
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    input_path = temp_dir / f"upload_{file.filename}"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    request = ProcessRequest(
        input_path=str(input_path),
        steps=[
            PipelineStep(
                feature="Frame Interpolation",
                model=model,
                parameters={
                    "multiplier": multiplier,
                    "slowmo_factor": slowmo_factor,
                    "device": device,
                }
            )
        ],
    )

    task = processing_service.create_task(request)
    response = await processing_service.start_processing(task.task_id, request)
    return response


@router.post("/match-color/file", response_model=ProcessResponse)
async def match_color_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    reference: UploadFile = File(...),
    model: str = Form(default_model("Match Color")),
    strength: float = Form(1.0),
    preserve_luminance: bool = Form(True),
    device: Optional[str] = Form("auto"),
    processing_service: ProcessingService = Depends(get_processing_service),
):
    """Match colors from reference image to target."""
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    input_path = temp_dir / f"upload_{file.filename}"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    ref_path = temp_dir / f"ref_{reference.filename}"
    with open(ref_path, "wb") as f:
        content = await reference.read()
        f.write(content)

    request = ProcessRequest(
        input_path=str(input_path),
        steps=[
            PipelineStep(
                feature="Match Color",
                model=model,
                parameters={
                    "strength": strength,
                    "preserve_luminance": preserve_luminance,
                    "reference_path": str(ref_path),
                    "device": device,
                }
            )
        ],
    )

    task = processing_service.create_task(request)
    response = await processing_service.start_processing(task.task_id, request)
    return response


@router.post("/preview/file", response_model=ProcessResponse)
async def preview_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    steps: str = Form(...),  # JSON string of PipelineStep[]
    output_format: str = Form("source"),
    quality: str = Form("medium"),
    duration: float = Form(2.0),
    processing_service: ProcessingService = Depends(get_processing_service),
):
    """Run the pipeline on a short sample of the uploaded file.

    Videos are trimmed to ``duration`` seconds and images downscaled, so the
    result can be inspected in seconds. The output stays in the temp folder and
    is fetched through ``GET /processing/task/{task_id}/result``.
    """
    import json
    from ...api.schemas import PipelineStep

    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    input_path = temp_dir / f"upload_{file.filename}"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        steps_data = json.loads(steps)
        pipeline_steps = [PipelineStep(**step) for step in steps_data]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid steps JSON: {e}")

    request = ProcessRequest(
        input_path=str(input_path),
        steps=pipeline_steps,
        output_format=output_format,
        quality=quality,
        preview=True,
        preview_duration=duration,
    )

    task = processing_service.create_task(request)
    response = await processing_service.start_processing(task.task_id, request)
    return response


@router.post("/file", response_model=ProcessResponse)
async def process_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    steps: str = Form(...),  # JSON string of PipelineStep[]
    output_format: str = Form("source"),
    quality: str = Form("medium"),
    processing_service: ProcessingService = Depends(get_processing_service),
):
    """Process an uploaded file with a multi-step pipeline."""
    import json
    from ...api.schemas import PipelineStep

    # Save uploaded file
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    input_path = temp_dir / f"upload_{file.filename}"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Parse pipeline steps
    try:
        steps_data = json.loads(steps)
        pipeline_steps = [PipelineStep(**step) for step in steps_data]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid steps JSON: {e}")

    request = ProcessRequest(
        input_path=str(input_path),
        steps=pipeline_steps,
        output_format=output_format,
        quality=quality,
    )

    task = processing_service.create_task(request)
    response = await processing_service.start_processing(task.task_id, request)
    return response


@router.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    processing_service: ProcessingService = Depends(get_processing_service),
):
    """Get processing task status."""
    status = processing_service.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


@router.get("/task/{task_id}/result")
async def get_task_result(
    task_id: str,
    processing_service: ProcessingService = Depends(get_processing_service),
):
    """Get processing result file."""
    task = processing_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Processing not completed")

    if not task.output_path or not Path(task.output_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(task.output_path, filename=Path(task.output_path).name)


@router.post("/task/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    processing_service: ProcessingService = Depends(get_processing_service),
):
    """Cancel a processing task."""
    task = processing_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status not in (TaskStatus.PENDING, TaskStatus.PROCESSING):
        raise HTTPException(status_code=400, detail="Task cannot be cancelled")

    success = processing_service.cancel_task(task_id)
    if success:
        return {"message": "Task cancellation requested"}
    else:
        raise HTTPException(status_code=400, detail="Could not cancel task")


@router.get("/tasks", response_model=List[TaskStatusResponse])
async def list_tasks(
    processing_service: ProcessingService = Depends(get_processing_service),
):
    """List all processing tasks."""
    return processing_service.get_all_tasks()


# SSE endpoint for real-time task progress
@router.get("/task/{task_id}/stream")
async def stream_task_progress(
    task_id: str,
    processing_service: ProcessingService = Depends(get_processing_service),
):
    """Stream task progress via Server-Sent Events."""
    from ...events.bus import get_event_bus, EventType, ProcessingProgressEvent

    event_bus = get_event_bus()

    async def event_generator():
        queue = asyncio.Queue()

        def handler(event: ProcessingProgressEvent):
            if event.task_id == task_id:
                asyncio.create_task(queue.put(event))

        unsubscribe = event_bus.subscribe(
            EventType.PROCESSING_PROGRESS,
            handler,
            async_handler=True
        )
        event_bus.subscribe(
            EventType.PROCESSING_COMPLETED,
            handler,
            async_handler=True
        )
        event_bus.subscribe(
            EventType.PROCESSING_FAILED,
            handler,
            async_handler=True
        )
        event_bus.subscribe(
            EventType.PROCESSING_ABORTED,
            handler,
            async_handler=True
        )

        try:
            # Send initial status
            task = processing_service.get_task(task_id)
            if task:
                yield f"data: {TaskStatusResponse(task_id=task.task_id, status=task.status, progress=task.progress, message=task.message).model_dump_json()}\n\n"

            while True:
                event = await queue.get()
                yield f"data: {event.model_dump_json()}\n\n"

                if event.type in (EventType.PROCESSING_COMPLETED, EventType.PROCESSING_FAILED, EventType.PROCESSING_ABORTED):
                    break
        finally:
            unsubscribe()

    return StreamingResponse(event_generator(), media_type="text/event-stream")
"""
Model management API routes.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse

from ...api.schemas import (
    ModelResponse,
    ModelRecommendationsResponse,
    DownloadRequest,
    DownloadStatus,
    SystemStatus,
    ErrorResponse,
)
from ...api.dependencies import (
    get_model_service,
    get_model_registry,
    get_pipeline_registry,
    validate_model_exists,
    validate_model_downloaded,
    get_pagination,
    PaginationParams,
)
from ...services.model_service import ModelService
from ...models.registry import ModelRegistry, get_registry
from ...utils.system_resources import check_model_compatibility

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=List[ModelResponse])
async def list_models(
    pagination: PaginationParams = Depends(get_pagination),
    feature: Optional[str] = Query(None, description="Filter by feature"),
    downloaded_only: bool = Query(False, description="Show only downloaded models"),
    compatible_only: bool = Query(False, description="Show only compatible models"),
    is_ai: Optional[bool] = Query(None, description="Filter by AI vs non-AI models"),
    model_service: ModelService = Depends(get_model_service),
    registry: ModelRegistry = Depends(get_model_registry),
):
    """Get all models with their download status."""
    models_data = model_service.get_all_models_with_status()

    # Apply filters
    if feature:
        models_data = [m for m in models_data if feature in m.get("features", [])]

    if downloaded_only:
        models_data = [m for m in models_data if m.get("downloaded")]

    if compatible_only:
        models_data = [m for m in models_data if m.get("compatible", True)]

    if is_ai is not None:
        models_data = [m for m in models_data if m.get("is_ai", True) == is_ai]

    # Apply pagination
    start = pagination.offset
    end = start + pagination.limit
    models_data = models_data[start:end]

    # Build response with compatibility info
    response = []
    for m in models_data:
        compat = check_model_compatibility(
            requires_cuda=m.get("requires_cuda", False),
            requires_metal=m.get("requires_metal", False),
            can_run_on_cpu=m.get("can_run_on_cpu", True),
        )
        response.append(ModelResponse(
            name=m["name"],
            description=m["description_short"],
            size=m["size"],
            weight=m["weight"],
            downloaded=m["downloaded"],
            implemented=m.get("implemented", True),
            features=m["features"],
            is_ai=m["is_ai"],
            is_builtin=m["is_builtin"],
            requires_cuda=m["requires_cuda"],
            requires_metal=m["requires_metal"],
            can_run_on_cpu=m["can_run_on_cpu"],
            has_download_url=bool(m.get("download_url")),
            compatible=compat["compatible"],
            compatibility_reason=compat["reason"],
            local_path=m.get("local_path"),
            full_description=m.get("description"),
            filename=m.get("filename"),
            module=m.get("module"),
            source_url=m.get("url"),
            download_url=m.get("download_url"),
            verified=bool(m.get("verified", False)),
            error=m.get("error"),
            additional_paths=m.get("additional_paths") or {},
        ))

    return response


@router.get("/features")
async def list_features():
    """Feature metadata declared in models.json (name, display name, description)."""
    return get_registry().get_feature_metadata()


@router.get("/groups")
async def list_groups():
    """Simple-Mode sidebar groups declared in models.json."""
    return get_registry().get_groups()


@router.get("/experimental")
async def list_experimental():
    """Everything models.json marks as experimental, plus the current opt-in state.

    Powers the info popover next to Settings -> Advanced -> "Experimental
    Features". The list is catalog-driven: nothing is marked today, so it is
    empty until an entry (feature or model) gains ``"experimental": true``.
    """
    from ...services.settings_service import experimental_features_enabled

    return {
        "enabled": experimental_features_enabled(),
        "items": get_registry().get_experimental_items(),
    }


@router.get("/features/implemented")
async def list_implemented_features(
    registry: ModelRegistry = Depends(get_pipeline_registry),
):
    """Get features that have processing pipelines implemented.

    UI hints (toggle, strength, reference image) are derived from each
    pipeline's parameter schema and the catalog's feature metadata rather than
    being hardcoded here. Experimental features (and models) are omitted unless
    the user enabled them in Settings -> Advanced.
    """
    from ...processing.pipeline.registry import get_pipeline_registry
    from ...services.settings_service import experimental_features_enabled

    pipeline_registry = get_pipeline_registry()
    model_registry = get_registry()
    include_experimental = experimental_features_enabled()

    implemented = {}
    for feature in pipeline_registry.get_all_features():
        if not include_experimental and model_registry.is_feature_experimental(feature):
            continue
        info = pipeline_registry.get_pipeline_info(feature)
        if not info:
            continue

        available = list(info.available_models)
        if not include_experimental:
            available = model_registry.drop_experimental(available)
        # A feature with nothing runnable left is not offered at all.
        if not available:
            continue

        default_model_name = info.default_model
        if default_model_name not in available:
            default_model_name = available[0]

        meta = model_registry.get_feature(feature) or {}
        schema_props = (info.config.parameter_schema or {}).get("properties", {})
        always_on = bool(meta.get("alwaysOn"))
        implemented[feature] = {
            "key": feature.lower().replace(" ", "_").replace("/", "_"),
            "displayName": model_registry.get_feature_display_name(feature),
            "description": meta.get("description", ""),
            "required": always_on,
            "hasToggle": not always_on,
            "hasStrength": "strength" in schema_props,
            "requiresReferenceImage": "reference_image_path" in schema_props,
            "experimental": model_registry.is_feature_experimental(feature),
            "availableModels": available,
            "defaultModel": default_model_name,
        }
    return implemented


@router.get("/recommendations", response_model=ModelRecommendationsResponse)
async def get_model_recommendations(
    media: Optional[str] = Query(
        None,
        description="Media being processed: 'image' or 'video'. Omit when nothing is loaded.",
    ),
    feature: Optional[str] = Query(None, description="Limit to a single feature"),
):
    """Which model Smart Select would use for each feature, and why.

    Backs the "Smart Select" setting: the Enhancement sidebar looks the feature
    up here instead of duplicating the catalog and hardware rules. The rankings
    are computed from the catalog plus the detected hardware, so the endpoint is
    safe to call whether or not the setting is on.
    """
    from ...services.model_recommender import recommend_models
    from ...services.settings_service import (
        experimental_features_enabled,
        smart_select_enabled,
    )

    features = [feature] if feature else None
    recommendations = recommend_models(
        features=features,
        media=media,
        include_experimental=experimental_features_enabled(),
    )

    return ModelRecommendationsResponse(
        media=media if media in ("image", "video") else "both",
        smart_select=smart_select_enabled(),
        recommendations=recommendations,
    )


@router.get("/status", response_model=SystemStatus)
async def get_system_status(
    model_service: ModelService = Depends(get_model_service),
):
    """Get overall system status."""
    status = model_service.get_system_status()

    # Add hardware info
    from ...utils.system_resources import get_system_info
    sys_info = get_system_info()

    return SystemStatus(
        total_models=status["total_models"],
        downloaded=status["downloaded"],
        available=status["available"],
        models_dir=status["models_dir"],
        gpu_info=sys_info.get("gpu"),
        cpu_threads=sys_info.get("cpu_threads", 0),
        ram_gb=sys_info.get("ram_gb", 0),
        vram_gb=sys_info.get("vram_gb"),
    )


@router.get("/{model_name}", response_model=ModelResponse)
async def get_model(
    model_name: str = Depends(validate_model_exists),
    registry: ModelRegistry = Depends(get_model_registry),
):
    """Get details for a specific model."""
    model = registry.get_model(model_name)
    status = registry.get_status(model_name)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    downloaded = status.downloaded if status else False
    compat = check_model_compatibility(
        requires_cuda=model.requires_cuda,
        requires_metal=model.requires_metal,
        can_run_on_cpu=model.can_run_on_cpu,
    )

    local_path = None
    if downloaded and not model.is_builtin:
        path = registry.get_model_path(model_name)
        if path:
            local_path = str(path)

    return ModelResponse(
        name=model.name,
        description=model.description_short,
        size=model.size,
        weight=model.weight,
        downloaded=downloaded,
    implemented=model.is_implemented,
        features=[f["name"] for f in model.features],
        is_ai=model.is_ai,
        is_builtin=model.is_builtin,
        requires_cuda=model.requires_cuda,
        requires_metal=model.requires_metal,
        can_run_on_cpu=model.can_run_on_cpu,
        has_download_url=bool(model.download_url),
        compatible=compat["compatible"],
        compatibility_reason=compat["reason"],
        local_path=local_path,
        full_description=model.description,
        filename=model.filename,
        module=model.module,
        source_url=model.url,
        download_url=model.download_url,
        verified=bool(getattr(status, "verified", False)) if status else False,
        error=getattr(status, "error", None) if status else None,
        additional_paths=(
            {k: str(v) for k, v in status.additional_paths.items()}
            if status and status.additional_paths
            else {}
        ),
    )


@router.post("/download", response_model=DownloadStatus)
async def start_download(
    request: DownloadRequest,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(get_model_service),
):
    """Start downloading a model.

    The model name is taken from the request body (``DownloadRequest``),
    matching the UI client, rather than a query parameter.
    """
    model_name = request.model_name

    if model_service.is_downloaded(model_name) and not request.force:
        raise HTTPException(status_code=400, detail="Model already downloaded")

    # Check compatibility
    model = model_service.get_model(model_name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

    if model.is_builtin:
        raise HTTPException(status_code=400, detail="Built-in models don't need download")

    compat = check_model_compatibility(
        requires_cuda=model.requires_cuda,
        requires_metal=model.requires_metal,
        can_run_on_cpu=model.can_run_on_cpu,
    )
    if not compat["compatible"]:
        raise HTTPException(status_code=400, detail=f"Cannot download: {compat['reason']}")

    # Start download in background
    success = await model_service.download_model(model_name, request.force)

    status = DownloadStatus(
        model_name=model_name,
        total_bytes=0,
        downloaded_bytes=0,
        speed_bps=0,
        eta_seconds=0,
        status="completed" if success else "failed",
        error=None if success else "Download failed",
        progress_percent=100 if success else 0,
    )
    return status


@router.get("/{model_name}/progress", response_model=DownloadStatus)
async def get_download_progress(
    model_name: str = Depends(validate_model_exists),
    model_service: ModelService = Depends(get_model_service),
):
    """Get the last reported progress for a model download.

    Progress is tracked while a download runs (see ``GET /{model_name}/progress/stream``
    for live updates) and reported as ``not_started`` for models with no
    recorded activity.
    """
    progress = model_service.get_download_progress(model_name)
    if progress is None:
        return DownloadStatus(
            model_name=model_name,
            total_bytes=0,
            downloaded_bytes=0,
            speed_bps=0,
            eta_seconds=0,
            status="not_started",
            progress_percent=0,
        )

    total = progress.total_bytes or 0
    return DownloadStatus(
        model_name=model_name,
        total_bytes=progress.total_bytes,
        downloaded_bytes=progress.downloaded_bytes,
        speed_bps=progress.speed_bps,
        eta_seconds=progress.eta_seconds,
        status=progress.status,
        error=progress.error,
        progress_percent=(progress.downloaded_bytes / total * 100.0) if total else 0.0,
    )


@router.post("/{model_name}/cancel")
async def cancel_download(
    model_name: str = Depends(validate_model_exists),
    model_service: ModelService = Depends(get_model_service),
):
    """Cancel an ongoing download."""
    success = model_service.cancel_download(model_name)
    if not success:
        raise HTTPException(status_code=404, detail="No active download for this model")
    return {"message": "Download cancelled"}


@router.delete("/{model_name}")
async def delete_model(
    model_name: str = Depends(validate_model_downloaded),
    model_service: ModelService = Depends(get_model_service),
):
    """Delete a downloaded model."""
    success = model_service.delete_model(model_name)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"message": "Model deleted"}


# SSE endpoint for real-time download progress
@router.get("/{model_name}/progress/stream")
async def stream_download_progress(
    model_name: str = Depends(validate_model_exists),
):
    """Stream download progress via Server-Sent Events."""
    from ...events.bus import get_event_bus, EventType, ModelDownloadProgressEvent

    event_bus = get_event_bus()

    async def event_generator():
        queue = asyncio.Queue()

        def handler(event: ModelDownloadProgressEvent):
            if event.model_name == model_name:
                asyncio.create_task(queue.put(event))

        unsubscribe = event_bus.subscribe(
            EventType.MODEL_DOWNLOAD_PROGRESS,
            handler,
            async_handler=True
        )
        event_bus.subscribe(
            EventType.MODEL_DOWNLOAD_COMPLETED,
            handler,
            async_handler=True
        )
        event_bus.subscribe(
            EventType.MODEL_DOWNLOAD_FAILED,
            handler,
            async_handler=True
        )

        try:
            # Send initial status
            yield f"data: {ModelDownloadProgressEvent(type=EventType.MODEL_DOWNLOAD_PROGRESS, model_name=model_name, progress=0).model_dump_json()}\n\n"

            while True:
                event = await queue.get()
                yield f"data: {event.model_dump_json()}\n\n"

                if event.type in (EventType.MODEL_DOWNLOAD_COMPLETED, EventType.MODEL_DOWNLOAD_FAILED):
                    break
        finally:
            unsubscribe()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


import asyncio
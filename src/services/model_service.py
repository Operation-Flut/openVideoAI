"""
Model Service - Business logic for model management.
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from ..models.registry import ModelRegistry, get_registry, ModelConfig, ModelStatus, media_label
from ..models.downloader import ModelDownloader, DownloadProgress
from ..core.paths import models_dir
from ..events.bus import EventBus, EventType, ModelDownloadProgressEvent, get_event_bus

logger = logging.getLogger(__name__)


class ModelService:
    """Service for model management operations."""

    def __init__(self):
        self._registry = get_registry()
        self._downloader = ModelDownloader(str(models_dir()))
        self._event_bus = get_event_bus()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._active_downloads: Dict[str, asyncio.Task] = {}
        # Last progress reported per model, so the polling endpoint can answer
        # with real numbers instead of a placeholder.
        self._progress: Dict[str, DownloadProgress] = {}
        # The downloader reports progress through callbacks rather than a
        # per-call hook, so publish progress events from one shared listener.
        self._downloader.add_progress_callback(self._on_download_progress)

    def _on_download_progress(self, progress: DownloadProgress) -> None:
        """Forward downloader progress to the event bus."""
        self._progress[progress.model_name] = progress
        if progress.status in ("completed", "failed"):
            # Terminal states are published by download_model() itself.
            return
        total = progress.total_bytes or 0
        percent = (progress.downloaded_bytes / total * 100.0) if total else 0.0
        self._event_bus.publish(ModelDownloadProgressEvent(
            type=EventType.MODEL_DOWNLOAD_PROGRESS,
            model_name=progress.model_name,
            progress=percent,
            downloaded_bytes=progress.downloaded_bytes,
            total_bytes=progress.total_bytes,
            speed_bps=progress.speed_bps,
        ))

    def get_all_models(self) -> List[ModelConfig]:
        """Get all models from catalog."""
        return self._registry.get_all_models()

    def get_all_models_with_status(self) -> List[Dict[str, Any]]:
        """Get all models with download status."""
        self._registry.refresh_status()
        statuses = self._registry.get_all_statuses()

        result = []
        for status in statuses:
            config = status.config
            result.append({
                "name": config.name,
                "description": config.description,
                "description_short": config.description_short,
                "best_for": config.best_for,
                "size": config.size,
                "weight": config.weight,
                "url": config.url,
                "download_url": config.download_url,
                "filename": config.filename,
                "features": [f["name"] for f in config.features],
                "is_ai": config.is_ai,
                "media": config.media,
                "media_label": media_label(config.media),
                "requires_cuda": config.requires_cuda,
                "requires_metal": config.requires_metal,
                "can_run_on_cpu": config.can_run_on_cpu,
                "is_builtin": config.is_builtin,
                "module": config.module,
                "downloaded": status.downloaded,
                "implemented": config.is_implemented,
                # Built-ins (classical / FFmpeg-backed models) run from code and
                # have no file on disk, so they report no local path (their
                # internal sentinel path would look like a real download).
                "local_path": (
                    str(status.local_path)
                    if status.local_path and not config.is_builtin
                    else None
                ),
                "additional_paths": {k: str(v) for k, v in status.additional_paths.items()},
                "verified": status.verified,
                "error": status.error,
                "compatible": True,  # Will be computed by API
                "compatibility_reason": None,
            })
        return result

    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        return self._registry.get_model(name)

    def get_model_status(self, name: str) -> Optional[ModelStatus]:
        """Get model status by name."""
        return self._registry.get_status(name)

    def get_download_progress(self, name: str) -> Optional[DownloadProgress]:
        """Last progress reported for a model, if a download is/has been tracked."""
        return self._progress.get(name)

    def is_downloaded(self, name: str) -> bool:
        """Check if model is downloaded."""
        return self._registry.is_downloaded(name)

    def get_downloaded_models(self) -> List[ModelStatus]:
        """Get all downloaded models."""
        self._registry.refresh_status()
        return self._registry.get_downloaded_models()

    def get_models_for_feature(self, feature: str) -> List[ModelConfig]:
        """Get models that provide a specific feature."""
        return self._registry.get_models_for_feature(feature)

    def get_downloaded_for_feature(self, feature: str) -> List[ModelStatus]:
        """Get downloaded models for a feature."""
        self._registry.refresh_status()
        return self._registry.get_downloaded_for_feature(feature)

    def get_best_model_for_feature(
        self,
        feature: str,
        prefer_ai: bool = True,
        device_type: str = "auto"
    ) -> Optional[ModelStatus]:
        """Get best available model for a feature."""
        self._registry.refresh_status()
        return self._registry.get_best_model_for_feature(feature, prefer_ai, device_type)

    def get_model_path(self, name: str) -> Optional[Path]:
        """Get local path for model."""
        return self._registry.get_model_path(name)

    def get_all_model_paths(self, name: str) -> Dict[str, Path]:
        """Get all local paths for model (main + additional)."""
        return self._registry.get_all_model_paths(name)

    async def download_model(
        self,
        model_name: str,
        force: bool = False,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """Download a model with progress tracking."""
        if model_name in self._active_downloads:
            raise ValueError(f"Download already in progress for {model_name}")

        model = self._registry.get_model(model_name)
        if not model:
            raise ValueError(f"Model not found: {model_name}")

        if model.is_builtin:
            raise ValueError(f"Model {model_name} is built-in, no download needed")

        if self._registry.is_downloaded(model_name) and not force:
            raise ValueError(f"Model {model_name} already downloaded")

        # Check compatibility
        from ..utils.system_resources import check_model_compatibility
        compat = check_model_compatibility(
            requires_cuda=model.requires_cuda,
            requires_metal=model.requires_metal,
            can_run_on_cpu=model.can_run_on_cpu,
        )
        if not compat["compatible"]:
            raise ValueError(f"Model incompatible: {compat['reason']}")

        # Publish initial event
        event = ModelDownloadProgressEvent(
            type=EventType.MODEL_DOWNLOAD_STARTED,
            model_name=model_name,
            progress=0,
            downloaded_bytes=0,
            total_bytes=0,
            speed_bps=0,
        )
        self._event_bus.publish(event)

        async def download_task():
            # The downloader operates on catalog entries (ModelInfo), not on
            # bare names, and reports progress via registered callbacks.
            def sync_progress(p: DownloadProgress):
                if progress_callback and p.model_name == model_name:
                    progress_callback(p)

            try:
                info = self._downloader.get_model_by_name(model_name)
                if info is None:
                    raise ValueError(f"Model not found in catalog: {model_name}")

                self._downloader.add_progress_callback(sync_progress)
                try:
                    loop = asyncio.get_event_loop()
                    success = await loop.run_in_executor(
                        self._executor,
                        self._downloader.download_model,
                        info,
                    )
                finally:
                    self._downloader.remove_progress_callback(sync_progress)

                # Publish completion event
                event = ModelDownloadProgressEvent(
                    type=EventType.MODEL_DOWNLOAD_COMPLETED if success else EventType.MODEL_DOWNLOAD_FAILED,
                    model_name=model_name,
                    progress=100 if success else 0,
                )
                self._event_bus.publish(event)

                if success:
                    self._registry.refresh_status()

                return success

            except Exception as e:
                logger.exception(f"Download failed for {model_name}: {e}")
                event = ModelDownloadProgressEvent(
                    type=EventType.MODEL_DOWNLOAD_FAILED,
                    model_name=model_name,
                    progress=0,
                )
                self._event_bus.publish(event)
                return False
            finally:
                self._active_downloads.pop(model_name, None)

        task = asyncio.create_task(download_task())
        self._active_downloads[model_name] = task

        return await task

    def cancel_download(self, model_name: str) -> bool:
        """Cancel an ongoing download."""
        if model_name in self._active_downloads:
            self._active_downloads[model_name].cancel()
            self._active_downloads.pop(model_name, None)
            self._downloader.cancel_download(model_name)
            return True
        return False

    def delete_model(self, model_name: str) -> bool:
        """Delete a downloaded model."""
        success = self._downloader.delete_model(model_name)
        if success:
            self._registry.refresh_status()
        return success

    def get_features(self) -> List[str]:
        """Get all available features."""
        return self._registry.get_features()

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        self._registry.refresh_status()
        statuses = self._registry.get_all_statuses()

        return {
            "total_models": len(statuses),
            "downloaded": sum(1 for s in statuses if s.downloaded),
            "available": sum(1 for s in statuses if not s.downloaded and s.config.download_url),
            "models_dir": str(models_dir()),
        }

    def shutdown(self):
        """Shutdown the service."""
        for task in self._active_downloads.values():
            task.cancel()
        self._executor.shutdown(wait=True)


_model_service_instance: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Get or create ModelService instance."""
    global _model_service_instance
    if _model_service_instance is None:
        _model_service_instance = ModelService()
    return _model_service_instance
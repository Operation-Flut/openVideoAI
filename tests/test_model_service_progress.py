from src.events.bus import EventBus, EventType
from src.models.downloader import DownloadProgress
from src.services.model_service import ModelService


def make_service() -> ModelService:
    service = ModelService.__new__(ModelService)
    service._progress = {}
    service._event_bus = EventBus()
    return service


def test_terminal_download_progress_replaces_verifying_state():
    service = make_service()
    verifying = DownloadProgress(
        model_name="Test Model",
        total_bytes=1024,
        downloaded_bytes=1024,
        speed_bps=0,
        eta_seconds=0,
        status="verifying",
    )
    completed = DownloadProgress(
        model_name="Test Model",
        total_bytes=1024,
        downloaded_bytes=1024,
        speed_bps=0,
        eta_seconds=0,
        status="completed",
    )

    service._on_download_progress(verifying)
    service._on_download_progress(completed)

    assert service.get_download_progress("Test Model") is completed
    assert service.get_download_progress("Test Model").status == "completed"
    assert len(service._event_bus.get_history(EventType.MODEL_DOWNLOAD_PROGRESS)) == 1


def test_failed_download_progress_replaces_active_state():
    service = make_service()
    downloading = DownloadProgress(
        model_name="Test Model",
        total_bytes=1024,
        downloaded_bytes=1024,
        speed_bps=0,
        eta_seconds=0,
        status="downloading",
    )
    failed = DownloadProgress(
        model_name="Test Model",
        total_bytes=0,
        downloaded_bytes=0,
        speed_bps=0,
        eta_seconds=0,
        status="failed",
        error="verification failed",
    )

    service._on_download_progress(downloading)
    service._on_download_progress(failed)

    assert service.get_download_progress("Test Model") is failed
    assert service.get_download_progress("Test Model").status == "failed"
    assert len(service._event_bus.get_history(EventType.MODEL_DOWNLOAD_PROGRESS)) == 1

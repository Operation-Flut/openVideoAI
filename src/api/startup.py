"""
Startup progress tracking for openVideoAI.
"""
from typing import Dict

# Global startup state
_startup_progress: Dict[str, object] = {
    "stage": "starting",
    "progress": 0,
    "message": "Starting openVideoAI..."
}
_startup_complete = False


def get_startup_progress() -> Dict[str, object]:
    """Get current startup progress."""
    global _startup_progress, _startup_complete
    if _startup_complete:
        return {"stage": "ready", "progress": 100, "message": "Ready"}
    return _startup_progress.copy()


def set_startup_progress(stage: str, progress: int, message: str) -> None:
    """Set current startup progress."""
    global _startup_progress
    _startup_progress = {"stage": stage, "progress": progress, "message": message}


def mark_startup_complete() -> None:
    """Mark startup as complete."""
    global _startup_complete, _startup_progress
    _startup_complete = True
    _startup_progress = {"stage": "ready", "progress": 100, "message": "Ready"}
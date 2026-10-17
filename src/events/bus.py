"""
Event bus for progress tracking and inter-component communication.
Supports both synchronous and asynchronous event handling.
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Standard event types for the application."""
    # Processing events
    PROCESSING_STARTED = "processing.started"
    PROCESSING_PROGRESS = "processing.progress"
    PROCESSING_COMPLETED = "processing.completed"
    PROCESSING_FAILED = "processing.failed"
    PROCESSING_ABORTED = "processing.aborted"

    # Model events
    MODEL_DOWNLOAD_STARTED = "model.download.started"
    MODEL_DOWNLOAD_PROGRESS = "model.download.progress"
    MODEL_DOWNLOAD_COMPLETED = "model.download.completed"
    MODEL_DOWNLOAD_FAILED = "model.download.failed"
    MODEL_DELETED = "model.deleted"

    # System events
    SETTINGS_CHANGED = "settings.changed"
    GPU_DETECTED = "gpu.detected"
    SHUTDOWN_REQUESTED = "shutdown.requested"


@dataclass
class Event:
    """Base event class."""
    type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    correlation_id: Optional[str] = None  # For tracking related events


@dataclass
class ProcessingProgressEvent(Event):
    """Event for processing progress updates."""
    type: EventType = EventType.PROCESSING_PROGRESS
    task_id: str = ""
    progress: float = 0.0  # 0-100
    message: str = ""
    current_frame: Optional[int] = None
    total_frames: Optional[int] = None


@dataclass
class ModelDownloadProgressEvent(Event):
    """Event for model download progress updates."""
    type: EventType = EventType.MODEL_DOWNLOAD_PROGRESS
    model_name: str = ""
    progress: float = 0.0  # 0-100
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_bps: float = 0.0


class EventBus:
    """
    Central event bus for publishing and subscribing to events.
    Supports both sync and async handlers.
    """

    def __init__(self):
        self._sync_handlers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self._async_handlers: Dict[EventType, List[Callable[[Event], asyncio.coroutine]]] = {}
        self._wildcard_sync: List[Callable[[Event], None]] = []
        self._wildcard_async: List[Callable[[Event], asyncio.coroutine]] = []
        self._event_history: List[Event] = []
        self._max_history = 1000

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], None],
        *,
        async_handler: bool = False
    ) -> Callable[[], None]:
        """
        Subscribe to an event type.

        Returns:
            Unsubscribe function
        """
        if async_handler:
            if event_type not in self._async_handlers:
                self._async_handlers[event_type] = []
            self._async_handlers[event_type].append(handler)
        else:
            if event_type not in self._sync_handlers:
                self._sync_handlers[event_type] = []
            self._sync_handlers[event_type].append(handler)

        def unsubscribe():
            if async_handler:
                self._async_handlers[event_type].remove(handler)
            else:
                self._sync_handlers[event_type].remove(handler)

        return unsubscribe

    def subscribe_all(
        self,
        handler: Callable[[Event], None],
        *,
        async_handler: bool = False
    ) -> Callable[[], None]:
        """Subscribe to all events."""
        if async_handler:
            self._wildcard_async.append(handler)
        else:
            self._wildcard_sync.append(handler)

        def unsubscribe():
            if async_handler:
                self._wildcard_async.remove(handler)
            else:
                self._wildcard_sync.remove(handler)

        return unsubscribe

    def publish(self, event: Event) -> None:
        """Publish an event synchronously."""
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)

        # Call sync handlers
        for handler in self._sync_handlers.get(event.type, []):
            try:
                handler(event)
            except Exception as e:
                logger.exception(f"Error in sync handler for {event.type}: {e}")

        # Call wildcard sync handlers
        for handler in self._wildcard_sync:
            try:
                handler(event)
            except Exception as e:
                logger.exception(f"Error in wildcard sync handler: {e}")

    async def publish_async(self, event: Event) -> None:
        """Publish an event asynchronously."""
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)

        # Call sync handlers first (non-blocking)
        for handler in self._sync_handlers.get(event.type, []):
            try:
                handler(event)
            except Exception as e:
                logger.exception(f"Error in sync handler for {event.type}: {e}")

        # Call async handlers
        tasks = []
        for handler in self._async_handlers.get(event.type, []):
            tasks.append(asyncio.create_task(self._safe_async_call(handler, event)))

        for handler in self._wildcard_async:
            tasks.append(asyncio.create_task(self._safe_async_call(handler, event)))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_async_call(self, handler: Callable, event: Event) -> None:
        try:
            await handler(event)
        except Exception as e:
            logger.exception(f"Error in async handler for {event.type}: {e}")

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Event]:
        """Get event history, optionally filtered by type."""
        events = self._event_history
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def set_event_bus(bus: EventBus) -> None:
    """Set the global event bus instance (for testing)."""
    global _event_bus
    _event_bus = bus
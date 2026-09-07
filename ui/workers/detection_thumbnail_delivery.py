"""Thumbnail publication scoped to the originating scene detection."""

from collections.abc import Callable
from typing import Any

from ui.workers.thumbnail_delivery import ThumbnailDelivery


class DetectionThumbnailDelivery(ThumbnailDelivery):
    def __init__(
        self, window: Any, worker: Any, guard: Any, completed: Callable[[], None]
    ) -> None:
        self.guard = guard
        super().__init__(
            window,
            worker,
            ready=window._on_thumbnail_ready,
            completed=completed,
            progress=window._on_thumbnail_progress,
        )

    def _current(self) -> bool:
        return (
            super()._current()
            and self.window._active_detection_guard is self.guard
            and self.window.project.session.session_id == self.guard.session_id
        )

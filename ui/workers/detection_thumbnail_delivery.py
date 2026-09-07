"""Deliver thumbnails only for the detection that requested them."""

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Slot


class DetectionThumbnailDelivery(QObject):
    def __init__(
        self, window: Any, worker: Any, guard: Any, completed: Callable[[], None]
    ) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.guard = guard
        self.completed = completed
        worker.progress.connect(self.progress)
        worker.thumbnail_ready.connect(self.thumbnail_ready)
        worker.finished.connect(self.finished)

    def _current(self) -> bool:
        return (
            self.window.thumbnail_worker is self.worker
            and self.window._active_detection_guard is self.guard
            and self.window.project.session.session_id == self.guard.session_id
        )

    @Slot(int, int)
    def progress(self, current: int, total: int) -> None:
        if self._current():
            self.window._on_thumbnail_progress(current, total)

    @Slot(str, str)
    def thumbnail_ready(self, clip_id: str, path: str) -> None:
        if self._current():
            self.window._on_thumbnail_ready(clip_id, path)

    @Slot()
    def finished(self) -> None:
        try:
            if self._current():
                self.completed()
        finally:
            if self.window.thumbnail_worker is self.worker:
                self.window.thumbnail_worker = None
            self.worker.deleteLater()
            self.deleteLater()

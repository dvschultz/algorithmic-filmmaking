"""Retain thumbnail workers and publish only to their originating editor state."""

import logging
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Slot

from core.operations.thumbnails import ThumbnailOutcome

logger = logging.getLogger(__name__)


class ThumbnailDelivery(QObject):
    def __init__(
        self,
        window: Any,
        worker: Any,
        *,
        ready: Callable[[str, str], None],
        completed: Callable[[], None],
        progress: Callable[[int, int], None] | None = None,
        valid: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self._cleaned = False
        self.ready = ready
        self.completed = completed
        self.progress_callback = progress
        self.valid = valid
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.application = worker.application
        self.delivered: set[str] = set()
        self._outcomes: dict[str, ThumbnailOutcome] | None = None
        previous = getattr(window, "_thumbnail_delivery", None)
        if previous is not None:
            previous.worker.cancel()
        window._thumbnail_delivery = self
        if not hasattr(window, "_active_thumbnail_workers"):
            window._active_thumbnail_workers = set()
        window._active_thumbnail_workers.add(worker)
        worker.setParent(self)
        worker.outcome_ready.connect(self.outcome)
        worker.progress.connect(self.progress)
        worker.error.connect(self.error)
        worker.finished.connect(self.finished)

    def _current(self) -> bool:
        return bool(
            self.window.thumbnail_worker is self.worker
            and self.application is not None
            and self.window.project is self.application.project
            and self.window.project.session.session_id == self.application.session_id
            and self.window.project.path == self.application.path
            and not self.worker.is_cancelled()
            and (self.reply is None or self.reply.is_current(self.window))
            and (self.valid is None or self.valid())
        )

    @Slot(object)
    def outcome(self, outcome: ThumbnailOutcome) -> None:
        if not self._current() or outcome.clip_id in self.delivered:
            return
        if self._outcomes is None:
            self._outcomes = {item.clip_id: item for item in self.worker.result}
        if self._outcomes.get(outcome.clip_id) != outcome:
            return
        self.delivered.add(outcome.clip_id)
        try:
            if (
                outcome.status == "succeeded"
                and outcome.path is not None
                and self.application.apply(outcome)
                and self._current()
            ):
                self.ready(outcome.clip_id, outcome.path)
            elif outcome.status == "failed":
                logger.warning(
                    "Thumbnail %s: %s", outcome.clip_id, outcome.message or outcome.code
                )
        except Exception as exc:
            logger.warning("Could not apply thumbnail: %s", exc)

    @Slot(int, int)
    def progress(self, current: int, total: int) -> None:
        if self._current() and self.progress_callback is not None:
            self.progress_callback(current, total)

    @Slot(str)
    def error(self, message: str) -> None:
        if self._current():
            logger.warning("Thumbnail generation: %s", message)

    @Slot()
    def finished(self) -> None:
        if self._cleaned or self.sender() is not self.worker or self.worker.isRunning():
            return
        try:
            if self._current():
                self.completed()
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Release a natively finished worker, or an unstarted failed dispatch."""
        if self._cleaned:
            return
        if self.worker.isRunning():
            raise RuntimeError("Cannot release a running thumbnail worker")
        self._cleaned = True
        self.window._active_thumbnail_workers.discard(self.worker)
        if self.window.thumbnail_worker is self.worker:
            self.window.thumbnail_worker = None
        if getattr(self.window, "_thumbnail_delivery", None) is self:
            self.window._thumbnail_delivery = None
        self.worker.deleteLater()
        self.deleteLater()

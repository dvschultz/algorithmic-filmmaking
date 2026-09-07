"""Deliver extracted frames on the originating project owner's Qt thread."""

from typing import Any

from PySide6.QtCore import QObject, Slot

from core.operations.frame_extraction import (
    FrameExtractionApplication,
    FrameExtractionOutcome,
)


class FrameExtractionDelivery(QObject):
    def __init__(self, window: Any, worker: Any) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.application = FrameExtractionApplication(window.project, worker.task)
        worker.progress.connect(self.progress)
        worker.outcome_ready.connect(self.complete)
        worker.finished.connect(self.finished)

    def current(self) -> bool:
        return (
            getattr(self.window, "_frame_extraction_worker", None) is self.worker
            and not self.worker.is_cancelled()
            and self.application.is_current(self.window.project)
        )

    @Slot(int, int)
    def progress(self, current: int, total: int) -> None:
        if self.current():
            self.window.status_bar.showMessage(f"Extracting frames: {current}/{total}")

    @Slot(object)
    def complete(self, outcome: FrameExtractionOutcome) -> None:
        if not self.current() or self.application.consumed:
            return
        if outcome.status == "failed":
            self.window.status_bar.showMessage(f"Extraction error: {outcome.message}")
            return
        if outcome.status != "succeeded":
            return
        try:
            applied = self.application.apply(self.window.project, outcome)
        except Exception as exc:
            self.window.status_bar.showMessage(
                f"Could not apply extracted frames: {exc}"
            )
            return
        if applied:
            self.window._on_frames_extracted(outcome.frames, self.worker.task.source_id)
        else:
            self.window.status_bar.showMessage(
                "Source changed during extraction. Extract frames again."
            )

    @Slot()
    def finished(self) -> None:
        if getattr(self.window, "_frame_extraction_worker", None) is self.worker:
            self.window._frame_extraction_worker = None
        self.worker.deleteLater()
        self.deleteLater()

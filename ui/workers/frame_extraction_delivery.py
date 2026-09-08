"""Deliver extracted frames on the originating project owner's Qt thread."""

from typing import Any

from PySide6.QtCore import Slot

from ui.workers.qt_lifetime import RetiringQObject

from core.operations.frame_extraction import (
    FrameExtractionApplication,
    FrameExtractionOutcome,
    FrameExtractionTask,
)


class FrameExtractionDelivery(RetiringQObject):
    def __init__(self, window: Any, worker: Any) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.application = FrameExtractionApplication(window.project, worker.task)
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.applied = False
        self.failure: str | None = None
        worker.progress.connect(self.progress)
        worker.outcome_ready.connect(self.complete)
        worker.finished.connect(self.finished)

    def current(self) -> bool:
        return (
            getattr(self.window, "_frame_extraction_worker", None) is self.worker
            and not self.worker.is_cancelled()
            and self.application.is_current(self.window.project)
            and (self.reply is None or self.reply.is_current(self.window))
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
            self.failure = outcome.message
            self.window.status_bar.showMessage(f"Extraction error: {outcome.message}")
            return
        if outcome.status != "succeeded":
            return
        try:
            cache = self.worker.cache
            receipt = (
                cache.results[self.worker.task.source_id] if cache is not None else None
            )
            recovered_task = None
            if receipt is not None:
                from core.jobs.gui_frame_extraction import FrameExtractionRecord

                recorded = cache.recorded
                if recorded is None:
                    raise ValueError("Extraction receipt is missing")
                recovered_task = FrameExtractionTask.from_dict(recorded.task)
                if not receipt.matches(
                    FrameExtractionRecord.build(recovered_task, outcome)
                ):
                    raise ValueError(
                        "Queued extraction differs from its recorded result"
                    )
            applied = self.application.apply(
                self.window.project, outcome, recovered_task=recovered_task
            )
            if applied and receipt is not None:
                self.window.project.record_job_result(receipt.result_id, receipt.digest)
        except Exception as exc:
            self.failure = str(exc)
            self.window.status_bar.showMessage(
                f"Could not apply extracted frames: {exc}"
            )
            return
        if applied:
            self.applied = True
            self.window._on_frames_extracted(outcome.frames, self.worker.task.source_id)
        else:
            self.failure = "Source changed during extraction. Extract frames again."
            self.window.status_bar.showMessage(
                "Source changed during extraction. Extract frames again."
            )

    @Slot()
    def finished(self) -> None:
        if self.reply is not None and self.application.is_current(self.window.project):
            result: dict = {"success": self.applied}
            if self.applied:
                result["result"] = {
                    "source_id": self.worker.task.source_id,
                    "frame_ids": [frame.id for frame in self.worker.result.frames],
                    "frame_count": len(self.worker.result.frames),
                }
            else:
                result["error"] = (
                    self.failure or "Frame extraction cancelled or discarded"
                )
            self.reply.send(self.window, result)
        if getattr(self.window, "_frame_extraction_worker", None) is self.worker:
            self.window._frame_extraction_worker = None
        self.worker.deleteLater()
        self.retire()

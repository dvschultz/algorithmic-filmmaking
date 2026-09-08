"""Queue imported images to the original project and agent requester."""

from typing import Any
from PySide6.QtCore import Slot

from ui.workers.qt_lifetime import RetiringQObject

from core.operations.image_import import (
    ImageImportApplication,
    ImageImportOutcome,
    ImageImportTask,
)
from core.jobs.image_import import ImageImportRecord


class ImageImportDelivery(RetiringQObject):
    def __init__(self, window: Any, worker: Any) -> None:
        super().__init__(window)
        self.window, self.worker = window, worker
        self.application = ImageImportApplication(window.project, worker.task)
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        worker.gui_tool_reply = self.reply
        self.applied = False
        self.failure: str | None = None
        worker.progress.connect(self.progress)
        worker.outcome_ready.connect(self.complete)
        worker.finished.connect(self.finished)

    def current(self) -> bool:
        return (
            getattr(self.window, "_image_import_worker", None) is self.worker
            and not self.worker.is_cancelled()
            and self.application.is_current(self.window.project)
            and (self.reply is None or self.reply.is_current(self.window))
        )

    @Slot(int, int)
    def progress(self, current: int, total: int) -> None:
        if self.current():
            self.window.status_bar.showMessage(f"Importing images: {current}/{total}")

    @Slot(object)
    def complete(self, outcome: ImageImportOutcome) -> None:
        if not self.current() or self.application.consumed:
            return
        if outcome.status != "succeeded":
            self.failure = "; ".join(outcome.errors) or "Image import cancelled"
        else:
            try:
                cache = self.worker.cache
                receipt = None
                recovered_task = None
                if cache is not None:
                    if cache.recorded is None:
                        raise ValueError("Image import receipt is missing")
                    recovered_task = ImageImportTask.from_dict(cache.recorded.task)
                    receipt = cache.results[cache.batch_id]
                    if not receipt.matches(
                        ImageImportRecord.build(recovered_task, outcome)
                    ):
                        raise ValueError(
                            "Queued image import differs from recorded result"
                        )
                applied = self.application.apply(
                    self.window.project, outcome, recovered_task=recovered_task
                )
                if applied and receipt is not None:
                    self.window.project.record_job_result(
                        receipt.result_id, receipt.digest
                    )
                self.applied = applied
                if not self.applied:
                    self.failure = "Image import target changed"
            except Exception as exc:
                self.failure = str(exc)
        if self.applied:
            self.window.frames_tab.update_frame_browser()
            self.window._update_chat_project_state()
            message = f"Imported {len(outcome.frames)} images"
            if outcome.errors:
                message += f"; {len(outcome.errors)} failed: {outcome.errors[0]}"
            self.window.status_bar.showMessage(message)
        elif self.failure:
            self.window.status_bar.showMessage(self.failure)

    @Slot()
    def finished(self) -> None:
        if (
            self.reply is not None
            and getattr(self.window, "_image_import_worker", None) is self.worker
        ):
            outcome = self.worker.result
            success = self.applied and self.application.is_current(self.window.project)
            result: dict = {"success": success}
            if success:
                result["result"] = {
                    "success": True,
                    "imported_count": len(outcome.frames),
                    "frame_ids": [frame.id for frame in outcome.frames],
                }
                if outcome.errors:
                    result["result"]["errors"] = list(outcome.errors)
            else:
                result["error"] = self.failure or "Image import cancelled or discarded"
            self.reply.send(self.window, result)
        if getattr(self.window, "_image_import_worker", None) is self.worker:
            self.window._image_import_worker = None
        self.worker.deleteLater()
        self.retire()

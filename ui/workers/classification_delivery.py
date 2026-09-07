"""Owner-thread classification publication bound to its launch context."""

from typing import Any

from PySide6.QtCore import QObject, Slot

from core.operations.classification import (
    ClassificationApplication,
    ClassificationOutcome,
)
from ui.workers.analysis_pipeline_delivery import pipeline_can_continue


class ClassificationDelivery(QObject):
    def __init__(
        self,
        window: Any,
        worker: Any,
        *,
        worker_attribute: str = "classification_worker",
        pipeline: bool = False,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.worker_attribute = worker_attribute
        self.application = ClassificationApplication(window.project, worker.tasks)
        self.pipeline = pipeline
        self.run = getattr(window, "_analysis_run", None) if pipeline else None
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.delivered: set[str] = set()
        worker.finished.connect(self.deleteLater)
        worker.labels_ready.connect(self.result)

    @Slot(str, list)
    def result(self, target_id: str, labels: list) -> None:
        window = self.window
        if (
            getattr(window, self.worker_attribute, None) is not self.worker
            or window.project is not self.application.project
            or window.project.session.session_id != self.application.session_id
            or self.worker.is_cancelled()
            or (self.reply is not None and not self.reply.is_current(window))
            or (
                self.pipeline
                and (
                    getattr(window, "_analysis_run", None) is not self.run
                    or not pipeline_can_continue(window)
                )
            )
            or target_id in self.delivered
        ):
            return
        self.delivered.add(target_id)
        try:
            outcome = ClassificationOutcome(
                target_id,
                "succeeded",
                tuple((label, float(confidence)) for label, confidence in labels),
            )
            accepted = self.application.apply(window.project, outcome)
        except Exception as exc:
            window._on_classification_error(f"Could not apply classification: {exc}")
            return
        if not accepted:
            window._on_classification_error(
                "Classification discarded because the target changed. Run analysis again."
            )

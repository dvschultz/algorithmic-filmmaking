"""Owner-thread object_detection publication bound to its launch context."""

from typing import Any

from PySide6.QtCore import QObject, Slot

from core.operations.object_detection import (
    ObjectDetectionApplication,
    ObjectDetectionOutcome,
    DetectedObject,
)
from ui.workers.analysis_pipeline_delivery import pipeline_can_continue


class ObjectDetectionDelivery(QObject):
    def __init__(
        self,
        window: Any,
        worker: Any,
        *,
        worker_attribute: str = "detection_worker_yolo",
        pipeline: bool = False,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.worker_attribute = worker_attribute
        self.application = ObjectDetectionApplication(
            window.project, worker.tasks, worker.options
        )
        self.pipeline = pipeline
        self.run = getattr(window, "_analysis_run", None) if pipeline else None
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.delivered: set[str] = set()
        worker.finished.connect(self.deleteLater)
        worker.objects_ready.connect(self.result)

    @Slot(str, list, int)
    def result(self, target_id: str, detections: list, person_count: int) -> None:
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
            outcome = ObjectDetectionOutcome(
                target_id,
                "succeeded",
                tuple(DetectedObject.from_dict(value) for value in detections),
                person_count,
            )
            accepted = self.application.apply(window.project, outcome)
        except Exception as exc:
            window._on_object_detection_error(
                f"Could not apply object detection: {exc}"
            )
            return
        if not accepted:
            window._on_object_detection_error(
                "Object detection discarded because the target changed. Run analysis again."
            )

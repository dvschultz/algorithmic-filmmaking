"""Owner-thread faces publication bound to its launch context."""

from typing import Any

from PySide6.QtCore import Slot

from ui.workers.qt_lifetime import RetiringQObject

from core.operations.faces import (
    FaceApplication,
    FaceOutcome,
    Face,
)
from ui.workers.analysis_pipeline_delivery import pipeline_can_continue


class FaceDelivery(RetiringQObject):
    def __init__(
        self,
        window: Any,
        worker: Any,
        *,
        worker_attribute: str = "face_detection_worker",
        pipeline: bool = False,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.worker_attribute = worker_attribute
        self.application = FaceApplication(window.project, worker.tasks)
        self.pipeline = pipeline
        self.run = getattr(window, "_analysis_run", None) if pipeline else None
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.delivered: set[str] = set()
        worker.finished.connect(self.retire)
        worker.faces_ready.connect(self.result)

    @Slot(str, list)
    def result(self, target_id: str, detections: list) -> None:
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
            outcome = FaceOutcome(
                target_id,
                "succeeded",
                tuple(Face.from_dict(value) for value in detections),
            )
            receipt = None
            cache = getattr(self.worker, "cache", None)
            if cache is not None:
                if (
                    window.project.path is None
                    or window.project.path.resolve() != cache.path
                ):
                    raise ValueError(
                        "Project save location changed during face detection"
                    )
                receipt = cache.results[target_id]
                if not receipt.matches(outcome):
                    raise ValueError(
                        "Queued face detection differs from its recorded result"
                    )
            accepted = self.application.apply(window.project, outcome)
            if accepted and receipt is not None:
                window.project.record_job_result(receipt.result_id, receipt.digest)
        except Exception as exc:
            window._on_face_detection_error(f"Could not apply face detection: {exc}")
            return
        if not accepted:
            window._on_face_detection_error(
                "Face detection discarded because the target changed. Run analysis again."
            )

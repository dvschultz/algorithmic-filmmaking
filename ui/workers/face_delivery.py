"""Owner-thread faces publication bound to its launch context."""

from typing import Any
from dataclasses import asdict

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
        self.application = FaceApplication(
            window.project, worker.tasks, getattr(worker, "options", None)
        )
        self.pipeline = pipeline
        self.run = getattr(window, "_analysis_run", None) if pipeline else None
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.delivered: set[str] = set()
        worker.finished.connect(self.retire)
        if hasattr(worker, "outcome_ready"):
            worker.outcome_ready.connect(self.outcome)
        else:
            worker.faces_ready.connect(self.result)

    @Slot(str, list)
    def result(self, target_id: str, detections: list) -> None:
        if not self._current(target_id):
            return
        try:
            self.outcome(
                FaceOutcome(
                    target_id,
                    "succeeded",
                    tuple(Face.from_dict(value) for value in detections),
                )
            )
        except (ValueError, TypeError, KeyError) as exc:
            self.delivered.add(target_id)
            self.window._on_face_detection_error(
                f"Could not apply face detection: {exc}"
            )

    def _current(self, target_id: str) -> bool:
        window = self.window
        return not (
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
        )

    @Slot(object)
    def outcome(self, outcome: FaceOutcome) -> None:
        if not isinstance(outcome, FaceOutcome) or not outcome.can_apply:
            return
        target_id = outcome.clip_id
        if not self._current(target_id):
            return
        window = self.window
        self.delivered.add(target_id)
        try:
            outcome = FaceOutcome.from_dict(asdict(outcome))
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
                receipt = cache.results.get(target_id)
                matches = (
                    receipt.matches(outcome)
                    if receipt is not None
                    else getattr(cache, "transient_outcomes", {}).get(target_id)
                    == asdict(outcome)
                )
                if not matches:
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

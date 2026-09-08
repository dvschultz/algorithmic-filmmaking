"""Owner-thread gaze publication bound to its launch context."""

from typing import Any
from dataclasses import asdict

from PySide6.QtCore import Slot

from ui.workers.qt_lifetime import RetiringQObject

from core.operations.gaze import (
    GazeApplication,
    GazeOutcome,
)
from ui.workers.analysis_pipeline_delivery import pipeline_can_continue


class GazeDelivery(RetiringQObject):
    def __init__(
        self,
        window: Any,
        worker: Any,
        *,
        worker_attribute: str = "_gaze_worker",
        pipeline: bool = False,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.worker_attribute = worker_attribute
        self.application = GazeApplication(
            window.project, worker.tasks, getattr(worker, "options", None)
        )
        self.pipeline = pipeline
        self.run = getattr(window, "_analysis_run", None) if pipeline else None
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.delivered: set[str] = set()
        worker.finished.connect(self.retire)
        worker.gaze_ready.connect(self.result)
        if hasattr(worker, "observation_ready"):
            worker.observation_ready.connect(self.observation)

    @Slot(str, float, float, str)
    def result(self, target_id: str, yaw: float, pitch: float, category: str) -> None:
        self.observation(GazeOutcome(target_id, "succeeded", yaw, pitch, category))

    @Slot(object)
    def observation(self, outcome: GazeOutcome) -> None:
        window = self.window
        if not isinstance(outcome, GazeOutcome):
            window._on_gaze_error("Invalid queued gaze observation")
            return
        target_id = outcome.clip_id
        if (
            not outcome.can_apply
            or getattr(window, self.worker_attribute, None) is not self.worker
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
            outcome = GazeOutcome.from_dict(asdict(outcome))
            receipt = None
            cache = getattr(self.worker, "cache", None)
            if cache is not None:
                if (
                    window.project.path is None
                    or window.project.path.resolve() != cache.path
                ):
                    raise ValueError(
                        "Project save location changed during gaze analysis"
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
                        "Queued gaze observation differs from its recorded result"
                    )
            accepted = self.application.apply(window.project, outcome)
            if accepted and receipt is not None:
                window.project.record_job_result(receipt.result_id, receipt.digest)
        except Exception as exc:
            window._on_gaze_error(f"Could not apply gaze detection: {exc}")
            return
        if not accepted:
            window._on_gaze_error(
                "Gaze detection discarded because the target changed. Run analysis again."
            )
        elif outcome.status == "succeeded" and hasattr(window, "_on_gaze_ready"):
            window._on_gaze_ready(
                target_id, outcome.yaw, outcome.pitch, outcome.category
            )

"""Owner-thread gazes publication bound to its launch context."""

from typing import Any

from PySide6.QtCore import QObject, Slot

from core.operations.gaze import (
    GazeApplication,
    GazeOutcome,
)
from ui.workers.analysis_pipeline_delivery import pipeline_can_continue


class GazeDelivery(QObject):
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
        self.application = GazeApplication(window.project, worker.tasks)
        self.pipeline = pipeline
        self.run = getattr(window, "_analysis_run", None) if pipeline else None
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.delivered: set[str] = set()
        worker.finished.connect(self.deleteLater)
        worker.gaze_ready.connect(self.result)

    @Slot(str, float, float, str)
    def result(self, target_id: str, yaw: float, pitch: float, category: str) -> None:
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
            outcome = GazeOutcome.from_result(
                target_id,
                {
                    "gaze_yaw": yaw,
                    "gaze_pitch": pitch,
                    "gaze_category": category,
                },
            )
            accepted = self.application.apply(window.project, outcome)
        except Exception as exc:
            window._on_gaze_error(f"Could not apply gaze detection: {exc}")
            return
        if not accepted:
            window._on_gaze_error(
                "Gaze detection discarded because the target changed. Run analysis again."
            )
        elif hasattr(window, "_on_gaze_ready"):
            window._on_gaze_ready(target_id, yaw, pitch, category)

"""Owner-thread cinematography publication scoped to the originating work."""

import json
from typing import Any

from PySide6.QtCore import QObject, Slot

from core.operations.cinematography import (
    CinematographyApplication,
    CinematographyOutcome,
)
from models.cinematography import CinematographyAnalysis
from ui.workers.analysis_pipeline_delivery import pipeline_can_continue


class CinematographyDelivery(QObject):
    def __init__(
        self,
        window: Any,
        worker: Any,
        *,
        worker_attribute: str = "cinematography_worker",
        pipeline: bool = False,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.worker_attribute = worker_attribute
        self.application = CinematographyApplication(window.project, worker.tasks)
        self.pipeline = pipeline
        self.run = getattr(window, "_analysis_run", None) if pipeline else None
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.delivered: set[str] = set()
        worker.finished.connect(self.deleteLater)
        worker.clip_completed.connect(self.result)

    @Slot(str, object)
    def result(self, target_id: str, analysis: CinematographyAnalysis) -> None:
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
            if not isinstance(analysis, CinematographyAnalysis):
                raise ValueError("Invalid cinematography result")
            outcome = CinematographyOutcome(
                target_id,
                "succeeded",
                json.dumps(analysis.to_dict(), sort_keys=True, allow_nan=False),
            )
            accepted = self.application.apply(window.project, outcome)
        except Exception as exc:
            window._on_cinematography_error(f"Could not apply cinematography: {exc}")
            return
        if accepted:
            if self.application.tasks[target_id].target_type == "clip":
                window._on_cinematography_clip_ready(target_id, outcome.analysis)
        else:
            window._on_cinematography_error(
                "Cinematography discarded because the target changed. Run analysis again."
            )

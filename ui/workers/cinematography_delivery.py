"""Owner-thread cinematography publication scoped to the originating work."""

import json
from dataclasses import asdict
from typing import Any

from PySide6.QtCore import Slot

from ui.workers.qt_lifetime import RetiringQObject

from core.operations.cinematography import (
    CinematographyApplication,
    CinematographyOutcome,
)
from models.cinematography import CinematographyAnalysis
from ui.workers.analysis_pipeline_delivery import pipeline_can_continue


class CinematographyDelivery(RetiringQObject):
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
        self.application = CinematographyApplication(
            window.project, worker.tasks, getattr(worker, "options", None)
        )
        self.pipeline = pipeline
        self.run = getattr(window, "_analysis_run", None) if pipeline else None
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.delivered: set[str] = set()
        worker.finished.connect(self.retire)
        if hasattr(worker, "outcome_ready"):
            worker.outcome_ready.connect(self.receive)
        else:
            worker.clip_completed.connect(self.result)

    @Slot(str, object)
    def result(self, target_id: str, analysis: CinematographyAnalysis) -> None:
        if not self._current(target_id):
            return
        try:
            if not isinstance(analysis, CinematographyAnalysis):
                raise ValueError("Invalid cinematography result")
            outcome = CinematographyOutcome(
                target_id,
                "succeeded",
                json.dumps(analysis.to_dict(), sort_keys=True, allow_nan=False),
            )
        except (ValueError, TypeError) as exc:
            self.delivered.add(target_id)
            self.window._on_cinematography_error(
                f"Could not apply cinematography: {exc}"
            )
            return
        self.receive(outcome)

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
    def receive(self, outcome: CinematographyOutcome) -> None:
        if not isinstance(outcome, CinematographyOutcome) or not outcome.can_apply:
            return
        target_id = outcome.clip_id
        if not self._current(target_id):
            return
        window = self.window
        self.delivered.add(target_id)
        try:
            receipt = None
            cache = getattr(self.worker, "cache", None)
            if cache is not None:
                if (
                    window.project.path is None
                    or window.project.path.resolve() != cache.path
                ):
                    raise ValueError(
                        "Project save location changed during cinematography"
                    )
                receipt = cache.results.get(target_id)
                if (receipt is not None and not receipt.matches(outcome)) or (
                    receipt is None
                    and getattr(cache, "transient_outcomes", {}).get(target_id)
                    != asdict(outcome)
                ):
                    raise ValueError(
                        "Queued cinematography differs from its recorded result"
                    )
            accepted = self.application.apply(window.project, outcome)
            if accepted and receipt is not None:
                window.project.record_job_result(receipt.result_id, receipt.digest)
        except Exception as exc:
            window._on_cinematography_error(f"Could not apply cinematography: {exc}")
            return
        if accepted and outcome.has_result:
            if self.application.tasks[target_id].target_type == "clip":
                window._on_cinematography_clip_ready(target_id, outcome.analysis)
        elif not accepted:
            window._on_cinematography_error(
                "Cinematography discarded because the target changed. Run analysis again."
            )

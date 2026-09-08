"""Scope queued custom-query results to their original worker and project."""

from typing import Any
from dataclasses import asdict

from PySide6.QtCore import Slot

from ui.workers.qt_lifetime import RetiringQObject

from core.operations.custom_query import CustomQueryApplication, CustomQueryOutcome
from ui.workers.analysis_pipeline_delivery import pipeline_can_continue


class CustomQueryDelivery(RetiringQObject):
    def __init__(self, window: Any, worker: Any) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.application = CustomQueryApplication(
            window.project, worker.tasks, getattr(worker, "options", None)
        )
        self.run = getattr(window, "_analysis_run", None)
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.delivered: set[str] = set()
        worker.finished.connect(self.retire)
        if hasattr(worker, "outcome_ready"):
            worker.outcome_ready.connect(self.receive)
        else:
            worker.query_result_ready.connect(self.result)

    @Slot(str, str, bool, float, str)
    def result(
        self, clip_id: str, query: str, match: bool, confidence: float, model: str
    ) -> None:
        self.receive(
            CustomQueryOutcome(clip_id, query, "succeeded", match, confidence, model)
        )

    @Slot(object)
    def receive(self, outcome: CustomQueryOutcome) -> None:
        if not isinstance(outcome, CustomQueryOutcome) or not outcome.can_apply:
            return
        clip_id = outcome.clip_id
        window = self.window
        if (
            window.custom_query_worker is not self.worker
            or window.project is not self.application.project
            or window.project.session.session_id != self.application.session_id
            or self.worker.is_cancelled()
            or getattr(window, "_analysis_run", None) is not self.run
            or (self.reply is not None and not self.reply.is_current(window))
            or not pipeline_can_continue(window)
            or clip_id in self.delivered
        ):
            return
        self.delivered.add(clip_id)
        try:
            receipt = None
            cache = getattr(self.worker, "cache", None)
            if cache is not None:
                if (
                    window.project.path is None
                    or window.project.path.resolve() != cache.path
                ):
                    raise ValueError(
                        "Project save location changed during custom query"
                    )
                receipt = cache.results.get(clip_id)
                if (receipt is not None and not receipt.matches(outcome)) or (
                    receipt is None
                    and getattr(cache, "transient_outcomes", {}).get(clip_id)
                    != asdict(outcome)
                ):
                    raise ValueError(
                        "Queued custom query differs from its recorded result"
                    )
            accepted = self.application.apply(window.project, outcome)
            if accepted and receipt is not None:
                window.project.record_job_result(receipt.result_id, receipt.digest)
        except Exception as exc:
            window._on_custom_query_error(f"Could not apply custom query: {exc}")
            return
        if accepted and outcome.has_result:
            window._on_custom_query_ready(
                clip_id, outcome.query, outcome.match, outcome.confidence, outcome.model
            )
        elif not accepted:
            window._on_custom_query_error(
                "Custom query discarded because the target changed. Run the query again."
            )

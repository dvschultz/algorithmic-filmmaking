"""Scope queued custom-query results to their original worker and project."""

from typing import Any

from PySide6.QtCore import QObject, Slot

from core.operations.custom_query import CustomQueryApplication, CustomQueryOutcome
from ui.workers.analysis_pipeline_delivery import pipeline_can_continue


class CustomQueryDelivery(QObject):
    def __init__(self, window: Any, worker: Any) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.application = CustomQueryApplication(window.project, worker.tasks)
        self.run = getattr(window, "_analysis_run", None)
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.delivered: set[str] = set()
        worker.finished.connect(self.deleteLater)
        worker.query_result_ready.connect(self.result)

    @Slot(str, str, bool, float, str)
    def result(
        self, clip_id: str, query: str, match: bool, confidence: float, model: str
    ) -> None:
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
            outcome = CustomQueryOutcome(
                clip_id,
                query,
                "succeeded",
                match,
                confidence,
                model,
            )
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
                receipt = cache.results[clip_id]
                if not receipt.matches(outcome):
                    raise ValueError(
                        "Queued custom query differs from its recorded result"
                    )
            accepted = self.application.apply(window.project, outcome)
            if accepted and receipt is not None:
                window.project.record_job_result(receipt.result_id, receipt.digest)
        except Exception as exc:
            window._on_custom_query_error(f"Could not apply custom query: {exc}")
            return
        if accepted:
            window._on_custom_query_ready(clip_id, query, match, confidence, model)
        else:
            window._on_custom_query_error(
                "Custom query discarded because the target changed. Run the query again."
            )

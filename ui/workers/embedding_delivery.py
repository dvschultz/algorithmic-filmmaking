"""Publish embedding outcomes only into their current owner-thread context."""

from typing import Any
from dataclasses import asdict

from PySide6.QtCore import Slot

from ui.workers.qt_lifetime import RetiringQObject

from core.operations.embeddings import EmbeddingApplication, EmbeddingOutcome
from ui.workers.analysis_pipeline_delivery import pipeline_can_continue


class EmbeddingDelivery(RetiringQObject):
    application_type: type = EmbeddingApplication
    worker_attribute = "_embeddings_worker"
    error_method = "_on_embeddings_error"

    def __init__(self, window: Any, worker: Any, *, pipeline: bool = False) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.application = self.application_type(window.project, worker.tasks)
        self.pipeline = pipeline
        self.run = getattr(window, "_analysis_run", None) if pipeline else None
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.delivered: set[str] = set()
        worker.finished.connect(self.retire)
        worker.outcome_ready.connect(self.result)

    @Slot(object)
    def result(self, outcome: EmbeddingOutcome) -> None:
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
            or outcome.clip_id in self.delivered
        ):
            return
        self.delivered.add(outcome.clip_id)
        try:
            receipt = None
            cache = getattr(self.worker, "cache", None)
            if cache is not None:
                if (
                    window.project.path is None
                    or window.project.path.resolve() != cache.path
                ):
                    raise ValueError(
                        "Project save location changed during embedding analysis"
                    )
                if outcome.status == "skipped" and outcome.record_json is not None:
                    if getattr(cache, "reused_outcomes", {}).get(outcome.clip_id) != asdict(outcome):
                        raise ValueError("Queued embedding differs from its verified reusable result")
                else:
                    receipt = cache.results[outcome.clip_id]
                    if not receipt.matches(outcome):
                        raise ValueError(
                            "Queued embedding differs from its recorded result"
                        )
            accepted = self.application.apply(window.project, outcome)
            if accepted and receipt is not None:
                window.project.record_job_result(receipt.result_id, receipt.digest)
        except Exception as exc:
            getattr(window, self.error_method)(f"Could not apply embedding: {exc}")
            return
        if not accepted:
            getattr(window, self.error_method)(
                "Embedding discarded because the target changed. Run analysis again."
            )
        else:
            window._on_embedding_ready(outcome.clip_id)


class BoundaryEmbeddingDelivery(EmbeddingDelivery):
    from core.operations.boundary_embeddings import BoundaryEmbeddingApplication

    application_type = BoundaryEmbeddingApplication
    worker_attribute = "_boundary_embeddings_worker"
    error_method = "_on_boundary_embeddings_error"

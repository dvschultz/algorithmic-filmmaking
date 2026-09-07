"""Publish embedding outcomes only into their current owner-thread context."""

from typing import Any

from PySide6.QtCore import QObject, Slot

from core.operations.embeddings import EmbeddingApplication, EmbeddingOutcome
from ui.workers.analysis_pipeline_delivery import pipeline_can_continue


class EmbeddingDelivery(QObject):
    def __init__(self, window: Any, worker: Any, *, pipeline: bool = False) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.application = EmbeddingApplication(window.project, worker.tasks)
        self.pipeline = pipeline
        self.run = getattr(window, "_analysis_run", None) if pipeline else None
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.delivered: set[str] = set()
        worker.finished.connect(self.deleteLater)
        worker.outcome_ready.connect(self.result)

    @Slot(object)
    def result(self, outcome: EmbeddingOutcome) -> None:
        window = self.window
        if (
            getattr(window, "_embeddings_worker", None) is not self.worker
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
            accepted = self.application.apply(window.project, outcome)
        except Exception as exc:
            window._on_embeddings_error(f"Could not apply embedding: {exc}")
            return
        if not accepted:
            window._on_embeddings_error(
                "Embedding discarded because the target changed. Run analysis again."
            )
        else:
            window._on_embedding_ready(outcome.clip_id)

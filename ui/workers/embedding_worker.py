"""Compatibility QThread for detached shared DINOv2 extraction."""

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Signal, Slot

from core.operations.embeddings import (
    EmbeddingOptions,
    EmbeddingOutcome,
    EmbeddingTask,
    run_embeddings,
)
from ui.workers.base import CancellableWorker, summarize_clip_errors

if TYPE_CHECKING:
    from models.clip import Clip, Source

DEFAULT_CHUNK_SIZE = 16


class EmbeddingAnalysisWorker(CancellableWorker):
    """Emit immutable vectors; publication belongs to the project owner."""

    progress = Signal(int, int)
    embedding_ready = Signal(str)
    outcome_ready = Signal(object)
    analysis_completed = Signal()

    def __init__(
        self,
        clips: list["Clip"],
        sources_by_id: dict[str, "Source"] | None = None,
        skip_existing: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.options = EmbeddingOptions(max(1, chunk_size))
        self.tasks = tuple(
            EmbeddingTask(
                c.id,
                Path(c.thumbnail_path) if c.thumbnail_path else None,
                skip_existing and c.embedding is not None,
            )
            for c in clips
        )
        self.result: tuple[EmbeddingOutcome, ...] = ()

    @Slot()
    def run(self) -> None:
        self._log_start()
        self.progress.emit(0, len(self.tasks))
        errors = []

        def deliver(outcome: EmbeddingOutcome) -> None:
            if outcome.status == "succeeded":
                self.outcome_ready.emit(outcome)
                self.embedding_ready.emit(outcome.clip_id)
            elif outcome.status == "failed":
                errors.append(
                    (
                        outcome.clip_id,
                        outcome.message or outcome.code or "Embedding failed",
                    )
                )

        try:
            self.result = run_embeddings(
                self.tasks,
                self.options,
                cancel_event=self._cancel_event,
                on_outcome=deliver,
                progress=self.progress.emit,
            )
        except Exception as exc:
            self.error.emit(f"Embedding extraction failed: {exc}")
        finally:
            if errors and not self.is_cancelled():
                self.error.emit(
                    summarize_clip_errors(
                        errors, operation_label="Embedding extraction"
                    )
                )
            self.analysis_completed.emit()
            self._log_complete()

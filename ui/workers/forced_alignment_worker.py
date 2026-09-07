"""Qt progress and dependency adapter for shared serial word alignment."""

from __future__ import annotations

from PySide6.QtCore import Signal, Slot

from core.operations.alignment import (
    AlignmentOutcome,
    run_alignment,
    snapshot_alignment_tasks,
)
from ui.workers.base import CancellableWorker, summarize_clip_errors


class ForcedAlignmentWorker(CancellableWorker):
    """Capture detached inputs on construction; emit words without mutating clips."""

    progress = Signal(int, int)
    clip_aligned = Signal(str, list)
    alignment_completed = Signal()

    def __init__(
        self, clips: list, sources_by_id: dict, skip_existing: bool = True, parent=None
    ) -> None:
        super().__init__(parent)
        self.tasks = tuple(
            task
            for task in snapshot_alignment_tasks(
                clips or [], sources_by_id or {}, skip_existing=skip_existing
            )
            if task.skip_reason is None and task.target.source_path is not None
        )
        self.result: tuple[AlignmentOutcome, ...] = ()

    @Slot()
    def run(self) -> None:
        self._log_start()
        try:
            if not self.tasks:
                return
            if self.is_cancelled():
                self.result = run_alignment(self.tasks, cancel_event=self._cancel_event)
                return
            # Preserve the existing GUI feature-install flow during this
            # computation cutover; explicit capability jobs remain U14 work.
            from core import feature_registry

            try:
                ready, _ = feature_registry.check_feature_ready("word_alignment")
                if not ready:
                    if self.is_cancelled():
                        return
                    if not feature_registry.install_for_feature("word_alignment"):
                        self.error.emit(
                            "Could not install word-level alignment dependencies. Check Settings > Dependencies and try again."
                        )
                        return
            except Exception as exc:
                self.error.emit(f"Word alignment dependencies unavailable: {exc}")
                return

            def deliver(outcome: AlignmentOutcome) -> None:
                if outcome.status == "succeeded":
                    self.clip_aligned.emit(outcome.clip_id, list(outcome.words))

            self.result = run_alignment(
                self.tasks,
                cancel_event=self._cancel_event,
                on_outcome=deliver,
                progress=self.progress.emit,
            )
            errors = [
                (o.clip_id, o.message or o.code or "Alignment failed")
                for o in self.result
                if o.status == "failed"
            ]
            if errors:
                self.error.emit(
                    summarize_clip_errors(errors, operation_label="Word alignment")
                )
        except Exception as exc:
            self._log_error(str(exc))
            self.error.emit(str(exc))
        finally:
            self.alignment_completed.emit()
            self._log_complete()


__all__ = ["ForcedAlignmentWorker"]

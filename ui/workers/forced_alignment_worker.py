"""Qt progress and dependency adapter for shared serial word alignment."""

from __future__ import annotations

from dataclasses import asdict
from queue import Empty, Queue
from typing import TYPE_CHECKING

from PySide6.QtCore import Signal, Slot

from core.jobs import JobRuntime
from core.jobs.alignment import alignment_operation_spec
from core.jobs.gui_alignment import GuiAlignmentCache
from core.jobs.media import media_stamp
from core.transcription_models import WordTimestamp
from core.operations.alignment import (
    AlignmentOutcome,
    run_alignment,
    snapshot_alignment_tasks,
)
from ui.workers.base import CancellableWorker, summarize_clip_errors

if TYPE_CHECKING:
    from core.project import Project


class ForcedAlignmentWorker(CancellableWorker):
    """Capture detached inputs on construction; emit words without mutating clips."""

    progress = Signal(int, int)
    clip_aligned = Signal(str, list)
    alignment_completed = Signal()

    def __init__(
        self,
        clips: list,
        sources_by_id: dict,
        skip_existing: bool = True,
        parent=None,
        *,
        project: Project | None = None,
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
        self.operation = alignment_operation_spec(
            self.tasks,
            force=not skip_existing,
            arguments={
                "clip_ids": [task.clip_id for task in self.tasks],
                "force": not skip_existing,
            },
            persistence="session_only",
            session_id=project.session.session_id if project is not None else None,
            input_revision=str(project.mutation_generation)
            if project is not None
            else None,
        )
        self.task_id: str | None = None
        self.job_status: str | None = None
        self._runtime: JobRuntime | None = None
        self.cache = (
            GuiAlignmentCache(
                project.path,
                project.metadata.id,
                {clip.id: clip.source_id for clip in clips},
                project.metadata.job_results,
                force=not skip_existing,
                media_stamps={
                    task.target.source_path: media_stamp(task.target.source_path)
                    for task in self.tasks
                    if task.target.source_path is not None
                },
            )
            if project is not None and project.path is not None
            else None
        )

    def cancel(self) -> None:
        super().cancel()
        runtime = self._runtime
        if runtime is not None and self.task_id is not None:
            runtime.cancel(self.task_id)

    def _prepare(self) -> bool:
        if self.is_cancelled() or not self.tasks:
            return False
        # Preserve the existing GUI install flow until explicit capability jobs.
        from core import feature_registry

        ready, _ = feature_registry.check_feature_ready("word_alignment")
        if self.is_cancelled():
            return False
        if not ready and not feature_registry.install_for_feature("word_alignment"):
            raise RuntimeError(
                "Could not install word-level alignment dependencies. Check Settings > Dependencies and try again."
            )
        return not self.is_cancelled()

    @Slot()
    def run(self) -> None:
        self._log_start()
        runtime = None
        events: Queue[tuple[str, tuple]] = Queue()

        def emit_event(event: tuple[str, tuple]) -> None:
            kind, args = event
            if kind == "progress":
                self.progress.emit(*args)
            elif kind == "aligned":
                self.clip_aligned.emit(*args)

        def compute(progress, cancel):
            try:

                def deliver(outcome: AlignmentOutcome) -> None:
                    if outcome.status == "succeeded":
                        events.put(("aligned", (outcome.clip_id, list(outcome.words))))

                def report(current: int, total: int) -> None:
                    progress(
                        current / total if total else 1.0,
                        f"Aligning words ({current}/{total})",
                    )
                    events.put(("progress", (current, total)))

                if self.cache is not None:
                    outcomes = self.cache.run(
                        self.tasks, cancel, self._prepare, deliver, report
                    )
                elif not self._prepare():
                    outcomes = tuple(
                        AlignmentOutcome(task.clip_id, "unprocessed", code="cancelled")
                        for task in self.tasks
                    )
                else:
                    outcomes = run_alignment(
                        self.tasks,
                        cancel_event=cancel,
                        on_outcome=deliver,
                        progress=report,
                    )
                return {"outcomes": [asdict(outcome) for outcome in outcomes]}
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        try:
            runtime = JobRuntime.for_session(max_workers=1)
            self._runtime = runtime
            submission = runtime.submit(
                kind="align_words",
                args=self.operation.arguments,
                operation=self.operation,
                run=compute,
                cancellation_event=self._cancel_event,
            )
            self.task_id = submission["task_id"]
            while runtime.is_handle_live(self.task_id):
                try:
                    emit_event(events.get(timeout=0.05))
                except Empty:
                    pass
            runtime.shutdown()
            while not events.empty():
                emit_event(events.get_nowait())
            row = runtime.store.get(self.task_id)
            self.job_status = row.status
            self.result = tuple(
                AlignmentOutcome(
                    **{
                        **item,
                        "words": tuple(
                            WordTimestamp.from_dict(word) for word in item["words"]
                        ),
                    }
                )
                for item in (row.result or {}).get("outcomes", [])
            )
            if row.status == "cancelled" and not self.result:
                self.result = tuple(
                    AlignmentOutcome(task.clip_id, "unprocessed", code="cancelled")
                    for task in self.tasks
                )
            if row.status == "failed":
                raise RuntimeError(row.error or "Word alignment failed")
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
            try:
                if runtime is not None:
                    runtime.close_session()
            finally:
                self._runtime = None
                self.alignment_completed.emit()
                self._log_complete()


__all__ = ["ForcedAlignmentWorker"]

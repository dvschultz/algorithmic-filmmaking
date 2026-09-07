"""Boundary-pair computation using the shared embedding worker lifecycle."""

from typing import TYPE_CHECKING
from dataclasses import asdict
from queue import Queue

from PySide6.QtCore import Signal

from core.jobs.boundary_embeddings import _runtime, _target, _values
from core.jobs.gui_boundary_embeddings import GuiBoundaryEmbeddingCache
from core.jobs.media import media_stamp
from core.jobs.spec import OperationSpec
from core.operations.boundary_embeddings import (
    BoundaryEmbeddingTask,
    BoundaryEmbeddingOutcome,
    run_boundary_embeddings,
)
from core.operations.embeddings import embedding_model_session
from ui.workers.base import CancellableWorker
from ui.workers.embedding_worker import _run_embedding_job
from ui.workers.job_adapter import gui_job_operation

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip


class BoundaryEmbeddingWorker(CancellableWorker):
    progress = Signal(int, int)
    embedding_ready = Signal(str)
    outcome_ready = Signal(object)
    analysis_completed = Signal()

    def __init__(
        self,
        clips: list["Clip"],
        *,
        project: "Project",
        skip_existing: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        project.session.assert_owner()
        self.tasks = tuple(
            BoundaryEmbeddingTask(
                clip.id,
                project.sources_by_id[clip.source_id].file_path
                if clip.source_id in project.sources_by_id
                else None,
                clip.start_frame,
                clip.end_frame,
                project.sources_by_id[clip.source_id].fps
                if clip.source_id in project.sources_by_id
                else 0.0,
                skip_existing
                and clip.first_frame_embedding is not None
                and clip.last_frame_embedding is not None,
            )
            for clip in clips
        )
        self.result: tuple[BoundaryEmbeddingOutcome, ...] = ()
        self._media_stamps = {
            task.source_path: media_stamp(task.source_path)
            for task in self.tasks
            if task.source_path and not task.skip
        }
        self.runtime_identity = _runtime()
        self.operation = gui_job_operation(
            OperationSpec.build(
                kind="boundary_embeddings",
                version=1,
                arguments={"clip_ids": [task.clip_id for task in self.tasks]},
                inputs={
                    "targets": [_target(project, task.clip_id) for task in self.tasks],
                    "previous": {
                        task.clip_id: _values(project, task.clip_id)
                        for task in self.tasks
                    },
                    "runtime": self.runtime_identity,
                },
                persistence="session_only",
                session_id=project.session.session_id,
                input_revision=str(project.mutation_generation),
            ),
            project.path,
        )
        from core.jobs import JobRuntime

        self.task_id: str | None = None
        self.job_status: str | None = None
        self._runtime: JobRuntime | None = None
        self.cache = (
            GuiBoundaryEmbeddingCache(project, self.tasks) if project.path else None
        )

    def _prepare(self) -> bool:
        if _runtime() != self.runtime_identity or any(
            media_stamp(path) != stamp for path, stamp in self._media_stamps.items()
        ):
            raise RuntimeError("Boundary embedding runtime changed while queued")
        return not self.is_cancelled()

    def cancel(self) -> None:
        super().cancel()
        if self._runtime is not None and self.task_id is not None:
            self._runtime.cancel(self.task_id)

    def run(self) -> None:
        self._log_start()
        self.progress.emit(0, len(self.tasks))
        events: Queue = Queue()
        errors: list[tuple[str, str]] = []

        def relay(event):
            if self.is_cancelled():
                return
            kind, value = event
            if kind == "progress":
                self.progress.emit(*value)
            elif value.status == "succeeded":
                self.outcome_ready.emit(value)
                self.embedding_ready.emit(value.clip_id)
            elif value.status == "failed":
                errors.append(
                    (
                        value.clip_id,
                        value.message or value.code or "Boundary analysis failed",
                    )
                )

        def compute(progress, cancel):
            collected = {}

            def deliver(outcome):
                collected[outcome.clip_id] = outcome
                events.put(("outcome", outcome))

            def report(current, total):
                progress(
                    current / total if total else 1.0,
                    f"Boundary embeddings ({current}/{total})",
                )
                events.put(("progress", (current, total)))

            try:
                if self.cache is not None:
                    outcomes = self.cache.run(
                        self.tasks, cancel, self._prepare, deliver, report
                    )
                elif self._prepare():
                    outcomes = self._run_uncached(cancel, deliver, report)
                else:
                    outcomes = tuple(
                        BoundaryEmbeddingOutcome(
                            task.clip_id, "unprocessed", code="cancelled"
                        )
                        for task in self.tasks
                    )
                return {"outcomes": [asdict(outcome) for outcome in outcomes]}
            except Exception as exc:
                for task in self.tasks:
                    if task.clip_id not in collected:
                        deliver(
                            BoundaryEmbeddingOutcome(
                                task.clip_id,
                                "failed",
                                code="embedding_failed",
                                message=str(exc),
                            )
                        )
                return {
                    "success": False,
                    "error": str(exc),
                    "outcomes": [
                        asdict(collected[task.clip_id]) for task in self.tasks
                    ],
                }

        _run_embedding_job(
            self, compute, relay, events, BoundaryEmbeddingOutcome, errors
        )

    def _run_uncached(self, cancel, deliver, report):
        outcomes = []
        with embedding_model_session() as session:
            for index, task in enumerate(self.tasks):
                if cancel.is_set() or session.failed:
                    outcomes.append(
                        BoundaryEmbeddingOutcome(
                            task.clip_id,
                            "unprocessed",
                            code="cancelled" if cancel.is_set() else "embedding_failed",
                        )
                    )
                    continue
                outcome = run_boundary_embeddings(
                    (task,), cancel_event=cancel, model_session=session
                )[0]
                outcomes.append(outcome)
                if not cancel.is_set():
                    deliver(outcome)
                    report(index + 1, len(self.tasks))
        return tuple(outcomes)

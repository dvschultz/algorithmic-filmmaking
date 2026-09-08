"""Compatibility QThread for shared, recoverable thumbnail embeddings."""

from dataclasses import asdict
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Callable

from PySide6.QtCore import Signal

from core.jobs import JobRuntime
from core.jobs.embeddings import _runtime, _target
from core.jobs.gui_embeddings import GuiEmbeddingCache
from core.jobs.media import media_stamp
from core.jobs.spec import OperationSpec
from core.operations.embeddings import (
    EmbeddingOptions,
    EmbeddingOutcome,
    embedding_task,
    run_embeddings,
)
from ui.workers.base import CancellableWorker, summarize_clip_errors
from ui.workers.job_adapter import (
    gui_job_operation,
    gui_job_runtime,
    close_gui_job_runtime,
)

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip, Source

DEFAULT_CHUNK_SIZE = 16


class EmbeddingAnalysisWorker(CancellableWorker):
    """Compute detached vectors; model publication belongs to the project owner."""

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
        *,
        project: "Project | None" = None,
    ) -> None:
        super().__init__(parent)
        self.options = EmbeddingOptions(max(1, chunk_size))
        self.tasks = tuple(
            embedding_task(c, (project.sources_by_id if project is not None else (sources_by_id or {})).get(c.source_id), skip_existing=skip_existing)
            for c in clips
        )
        self.result: tuple[EmbeddingOutcome, ...] = ()
        targets = {
            c.id: _target(project, c.id)
            if project is not None
            else {
                "clip_id": c.id,
                "source_id": c.source_id,
                "thumbnail_path": str(c.thumbnail_path) if c.thumbnail_path else None,
                "start_frame": c.start_frame,
                "end_frame": c.end_frame,
                "source_path": None,
                "fps": None,
            }
            for c in clips
        }
        previous = {
            c.id: {"vector": c.embedding, "model": c.embedding_model} for c in clips
        }
        paths = {
            task.thumbnail_path
            for task in self.tasks
            if task.thumbnail_path
        }
        paths.update(
            Path(targets[t.clip_id]["source_path"])
            for t in self.tasks
            if targets[t.clip_id]["source_path"]
        )
        self._media_stamps = {path: media_stamp(path) for path in paths}
        self.operation = gui_job_operation(
            OperationSpec.build(
                kind="embeddings",
                version=2,
                arguments={"clip_ids": [t.clip_id for t in self.tasks]},
                inputs={
                    "targets": targets,
                    "options": asdict(self.options),
                    "runtime": _runtime(),
                    "previous": previous,
                },
                persistence="session_only",
                session_id=project.session.session_id if project else None,
                input_revision=str(project.mutation_generation) if project else None,
            ),
            project.path if project else None,
        )
        self.task_id: str | None = None
        self.job_status: str | None = None
        self._runtime: JobRuntime | None = None
        self.cache: GuiEmbeddingCache | None = None
        if project is not None and project.path is not None:
            self.cache = GuiEmbeddingCache(
                project.path,
                project.metadata.id,
                {c.id: c.source_id for c in clips},
                project.metadata.job_results,
                options=self.options,
                targets=targets,
                previous_results=previous,
                media_stamps=self._media_stamps,
            )

    def cancel(self) -> None:
        super().cancel()
        if self._runtime is not None and self.task_id is not None:
            self._runtime.cancel(self.task_id)

    def _prepare(self) -> bool:
        if any(
            media_stamp(path) != stamp for path, stamp in self._media_stamps.items()
        ):
            raise RuntimeError("Embedding extraction media changed while queued")
        return not self.is_cancelled()

    def run(self) -> None:
        """Relay shared-job results on the QThread; publish models on their owner."""
        self._log_start()
        self.progress.emit(0, len(self.tasks))
        events: Queue = Queue()
        errors: list[tuple[str, str]] = []

        def emit(event):
            kind, value = event
            if kind == "progress":
                self.progress.emit(*value)
            elif value.status == "succeeded" or (value.status == "skipped" and value.record_json is not None):
                self.outcome_ready.emit(value)
                self.embedding_ready.emit(value.clip_id)
            elif value.status == "failed":
                errors.append(
                    (value.clip_id, value.message or value.code or "Analysis failed")
                )

        def compute(progress, cancel):
            collected = {}

            def deliver(outcome):
                collected[outcome.clip_id] = outcome
                events.put(("outcome", outcome))

            def report(current, total):
                progress(
                    current / total if total else 1.0,
                    f"Embedding extraction ({current}/{total})",
                )
                events.put(("progress", (current, total)))

            try:
                if self.cache is not None:
                    outcomes = self.cache.run(
                        self.tasks, cancel, self._prepare, deliver, report
                    )
                elif self._prepare():
                    outcomes = run_embeddings(
                        self.tasks,
                        self.options,
                        cancel_event=cancel,
                        on_outcome=deliver,
                        progress=report,
                    )
                else:
                    outcomes = tuple(
                        EmbeddingOutcome(task.clip_id, "unprocessed", code="cancelled")
                        for task in self.tasks
                    )
                return {"outcomes": [asdict(outcome) for outcome in outcomes]}
            except Exception as exc:
                for task in self.tasks:
                    if task.clip_id not in collected:
                        deliver(
                            EmbeddingOutcome(
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

        _run_embedding_job(self, compute, emit, events, EmbeddingOutcome, errors)


def _run_embedding_job(
    worker: Any,
    compute: Callable,
    relay: Callable,
    events: Queue,
    outcome_type: type,
    errors: list[tuple[str, str]],
) -> None:
    """Run the shared lifecycle while each adapter owns its task/result types."""
    runtime = None
    try:
        runtime = gui_job_runtime(worker.operation)
        worker._runtime = runtime
        submission = runtime.submit(
            kind=worker.operation.kind,
            args=worker.operation.arguments,
            operation=worker.operation,
            run=compute,
            cancellation_event=worker._cancel_event,
            project_path=worker.operation.arguments.get("project_path"),
        )
        worker.task_id = submission["task_id"]
        while runtime.is_handle_live(worker.task_id):
            try:
                relay(events.get(timeout=0.05))
            except Empty:
                pass
        runtime.shutdown()
        while not events.empty():
            relay(events.get_nowait())
        row = runtime.store.get(worker.task_id)
        worker.job_status = row.status
        worker.result = tuple(
            outcome_type.from_dict(value)
            for value in (row.result or {}).get("outcomes", [])
        )
        if row.status == "cancelled" and not worker.result:
            worker.result = tuple(
                outcome_type(task.clip_id, "unprocessed", code="cancelled")
                for task in worker.tasks
            )
        if row.status == "failed" and not worker.result:
            raise RuntimeError(row.error or "Embedding extraction failed")
    except Exception as exc:
        worker.error.emit(str(exc))
    finally:
        try:
            if runtime is not None:
                close_gui_job_runtime(runtime)
        finally:
            worker._runtime = None
            if errors and not worker.is_cancelled():
                worker.error.emit(
                    summarize_clip_errors(
                        errors, operation_label="Embedding extraction"
                    )
                )
            worker.analysis_completed.emit()
            worker._log_complete()

"""Compatibility QThread for shared, detached gaze analysis."""

from typing import TYPE_CHECKING
from dataclasses import asdict
from queue import Empty, Queue
from core.jobs import JobRuntime
from core.jobs.gaze import _runtime, _task_data
from core.jobs.gui_gaze import GuiGazeCache
from core.jobs.media import media_stamp
from core.jobs.spec import OperationSpec
from ui.workers.job_adapter import (
    gui_job_operation,
    gui_job_runtime,
    close_gui_job_runtime,
)
from PySide6.QtCore import Signal
from core.operations.gaze import GazeOptions, GazeOutcome, gaze_task, run_gaze
from ui.workers.base import CancellableWorker, summarize_clip_errors

if TYPE_CHECKING:
    from models.clip import Clip, Source
    from core.project import Project


class GazeAnalysisWorker(CancellableWorker):
    """Compute immutable gaze outcomes; model publication belongs to the owner."""

    progress = Signal(int, int)
    gaze_ready = Signal(str, float, float, str)
    observation_ready = Signal(object)
    detection_completed = Signal()

    def __init__(
        self,
        clips: list["Clip"],
        sources_by_id: dict[str, "Source"],
        sample_interval: float = 1.0,
        skip_existing: bool = True,
        parent=None,
        *,
        project: "Project | None" = None,
    ) -> None:
        super().__init__(parent)
        self.options = GazeOptions(sample_interval)
        self.tasks = tuple(
            gaze_task(c, sources_by_id.get(c.source_id), skip_existing=skip_existing)
            for c in clips
        )
        self.result: tuple[GazeOutcome, ...] = ()

        previous = {
            c.id: {
                "gaze_yaw": c.gaze_yaw,
                "gaze_pitch": c.gaze_pitch,
                "gaze_category": c.gaze_category,
            }
            for c in clips
        }
        self._media_stamps = {
            task.source_path: media_stamp(task.source_path)
            for task in self.tasks
            if task.source_path is not None
        }
        self.operation = gui_job_operation(
            OperationSpec.build(
                kind="gaze",
                version=2,
                arguments={"clip_ids": [task.clip_id for task in self.tasks]},
                inputs={
                    "tasks": [_task_data(task) for task in self.tasks],
                    "options": asdict(self.options),
                    "runtime": _runtime(),
                    "previous": previous,
                },
                persistence="session_only",
                session_id=project.session.session_id if project is not None else None,
                input_revision=str(project.mutation_generation)
                if project is not None
                else None,
            ),
            project.path if project is not None else None,
        )
        self.task_id: str | None = None
        self.job_status: str | None = None
        self._runtime: JobRuntime | None = None
        self.cache: GuiGazeCache | None = None
        if project is not None and project.path is not None:
            self.cache = GuiGazeCache(
                project.path,
                project.metadata.id,
                {task.clip_id: task.source_id for task in self.tasks},
                project.metadata.job_results,
                options=self.options,
                previous_results=previous,
                media_stamps=self._media_stamps,
                skip_existing=skip_existing,
            )

    def cancel(self) -> None:
        super().cancel()
        if self._runtime is not None and self.task_id is not None:
            self._runtime.cancel(self.task_id)

    def _prepare(self) -> bool:
        if any(
            media_stamp(path) != stamp for path, stamp in self._media_stamps.items()
        ):
            raise RuntimeError("Gaze detection media changed while queued")
        return not self.is_cancelled()

    def run(self) -> None:
        """Relay shared-job results on the QThread; publish models on their owner."""
        self._log_start()
        self.progress.emit(0, len(self.tasks))
        runtime = None
        events: Queue = Queue()
        errors: list[tuple[str, str]] = []

        def emit(event):
            kind, value = event
            if kind == "progress":
                self.progress.emit(*value)
            else:
                if value.can_apply:
                    self.observation_ready.emit(value)
                if value.status == "succeeded" and value.category is not None:
                    self.gaze_ready.emit(
                        value.clip_id, value.yaw, value.pitch, value.category
                    )
                elif value.status == "failed":
                    errors.append(
                        (
                            value.clip_id,
                            value.message or value.code or "Analysis failed",
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
                    f"Gaze detection ({current}/{total})",
                )
                events.put(("progress", (current, total)))

            try:
                if self.cache is not None:
                    outcomes = self.cache.run(
                        self.tasks, cancel, self._prepare, deliver, report
                    )
                elif self._prepare():
                    outcomes = run_gaze(
                        self.tasks,
                        self.options,
                        cancel_event=cancel,
                        on_outcome=deliver,
                        progress=report,
                    )
                else:
                    outcomes = tuple(
                        GazeOutcome(task.clip_id, "unprocessed", code="cancelled")
                        for task in self.tasks
                    )
                return {"outcomes": [asdict(outcome) for outcome in outcomes]}
            except Exception as exc:
                for task in self.tasks:
                    if task.clip_id not in collected:
                        deliver(
                            GazeOutcome(
                                task.clip_id,
                                "failed",
                                code="gaze_failed",
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

        try:
            runtime = gui_job_runtime(self.operation)
            self._runtime = runtime
            submission = runtime.submit(
                kind=self.operation.kind,
                args=self.operation.arguments,
                operation=self.operation,
                run=compute,
                cancellation_event=self._cancel_event,
                project_path=self.operation.arguments.get("project_path"),
            )
            self.task_id = submission["task_id"]
            while runtime.is_handle_live(self.task_id):
                try:
                    emit(events.get(timeout=0.05))
                except Empty:
                    pass
            runtime.shutdown()
            while not events.empty():
                emit(events.get_nowait())
            row = runtime.store.get(self.task_id)
            self.job_status = row.status
            self.result = tuple(
                GazeOutcome.from_dict(value)
                for value in (row.result or {}).get("outcomes", [])
            )
            if row.status == "cancelled" and not self.result:
                self.result = tuple(
                    GazeOutcome(task.clip_id, "unprocessed", code="cancelled")
                    for task in self.tasks
                )
            if row.status == "failed" and not self.result:
                raise RuntimeError(row.error or "Gaze detection failed")
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            try:
                if runtime is not None:
                    close_gui_job_runtime(runtime)
            finally:
                self._runtime = None
                if errors and not self.is_cancelled():
                    model_failure = next(
                        (
                            outcome
                            for outcome in self.result
                            if outcome.code == "model_load_failed"
                        ),
                        None,
                    )
                    self.error.emit(
                        "Failed to load gaze detection model: "
                        + (model_failure.message or "unavailable")
                        if model_failure is not None
                        else summarize_clip_errors(
                            errors, operation_label="Gaze detection"
                        )
                    )
                self.detection_completed.emit()
                self._log_complete()

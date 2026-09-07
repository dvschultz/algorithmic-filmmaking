"""Qt adapter for detached batch frame extraction."""

from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING

from PySide6.QtCore import Signal

from core.operations.frame_extraction import (
    FrameExtractionTask,
    FrameExtractionOutcome,
    run_frame_extraction,
)
from core.jobs import JobRuntime
from core.jobs.spec import OperationSpec, encode_object
from core.jobs import gui_frame_extraction as frame_jobs
from models.clip import Clip, Source
from ui.workers.base import CancellableWorker
from ui.workers.job_adapter import (
    gui_job_operation,
    gui_job_runtime,
    close_gui_job_runtime,
)

if TYPE_CHECKING:
    from core.project import Project


class FrameExtractionWorker(CancellableWorker):
    progress = Signal(int, int)
    frame_ready = Signal(str, str)
    extraction_completed = Signal(list)
    outcome_ready = Signal(object)
    job_started = Signal(str, str)

    def __init__(
        self,
        source: Source,
        clip: Clip | None,
        mode: str,
        interval: int,
        output_dir: Path,
        parent=None,
        *,
        project: "Project | None" = None,
    ):
        super().__init__(parent)
        self.task = FrameExtractionTask.from_source(
            source, clip, mode, interval, output_dir
        )
        self.result: FrameExtractionOutcome | None = None
        if project is not None:
            project.session.assert_owner()
        runtime_identity = frame_jobs.frame_extraction_runtime()
        self.runtime_json = encode_object(runtime_identity)
        self.operation = gui_job_operation(
            OperationSpec.build(
                kind="extract_frames",
                version=1,
                arguments={"mode": mode, "interval": interval},
                inputs={"task": self.task.to_dict(), "runtime": self.runtime_json},
                persistence="session_only",
                session_id=project.session.session_id if project is not None else None,
                input_revision=str(project.mutation_generation)
                if project is not None
                else None,
            ),
            project.path if project is not None else None,
        )
        self.cache = (
            frame_jobs.GuiFrameExtractionCache(project, self.task, runtime_identity)
            if project is not None and project.path is not None
            else None
        )
        self.task_id: str | None = None
        self.job_status: str | None = None
        self._runtime: JobRuntime | None = None

    def cancel(self) -> None:
        super().cancel()
        if self._runtime is not None and self.task_id is not None:
            self._runtime.cancel(self.task_id)

    def run(self) -> None:
        self._log_start()
        runtime = None
        events: Queue[tuple[int, int]] = Queue()

        def compute(progress, cancel):
            def report(current, total):
                progress(current / total if total else 1.0, "Extracting frames")
                events.put((current, total))

            if (
                encode_object(frame_jobs.frame_extraction_runtime())
                != self.runtime_json
            ):
                raise ValueError("Frame extraction runtime changed while queued")
            outcome = (
                self.cache.run(self.task, cancel, report)
                if self.cache is not None
                else run_frame_extraction(
                    self.task, cancel_event=cancel, progress=report
                )
            )
            return {
                "success": outcome.status != "failed",
                "error": outcome.message,
                "outcome": outcome.to_dict(),
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
            self.job_started.emit(self.task_id, runtime.store.persistence)
            while runtime.is_handle_live(self.task_id):
                try:
                    current, total = events.get(timeout=0.05)
                    if not self.is_cancelled():
                        self.progress.emit(current, total)
                except Empty:
                    pass
            runtime.shutdown()
            while not events.empty():
                current, total = events.get_nowait()
                if not self.is_cancelled():
                    self.progress.emit(current, total)
            row = runtime.store.get(self.task_id)
            self.job_status = row.status
            if row.status == "cancelled" or self.is_cancelled():
                self.result = FrameExtractionOutcome(
                    self.task.request_id, "unprocessed"
                )
            else:
                payload = (row.result or {}).get("outcome")
                if payload is None:
                    raise RuntimeError(
                        row.error or "Frame extraction produced no result"
                    )
                self.result = FrameExtractionOutcome.from_dict(payload)
        except Exception as exc:
            self.result = FrameExtractionOutcome(
                self.task.request_id,
                "unprocessed" if self.is_cancelled() else "failed",
                message=str(exc),
            )
        finally:
            try:
                if runtime is not None:
                    close_gui_job_runtime(runtime)
            finally:
                self._runtime = None
        self.outcome_ready.emit(self.result)
        frames = []
        if self.result.status == "succeeded" and not self.is_cancelled():
            frames = [frame.to_model(self.task) for frame in self.result.frames]
            for frame in frames:
                if frame.thumbnail_path is not None:
                    self.frame_ready.emit(frame.id, str(frame.thumbnail_path))
        elif self.result.status == "failed":
            self.error.emit(self.result.message or "Frame extraction failed")
        self.extraction_completed.emit(frames)
        self._log_complete()

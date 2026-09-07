"""Qt adapter for shared detached audio probing."""

from pathlib import Path
from dataclasses import asdict
from queue import Empty, Queue
from typing import TYPE_CHECKING
from PySide6.QtCore import Signal
from core.jobs import JobRuntime
from core.jobs.spec import OperationSpec, encode_object
from core.jobs import gui_audio_import as audio_jobs
from core.operations.audio_import import (
    AudioImportTask,
    AudioImportOutcome,
    run_audio_import,
)
from ui.workers.base import CancellableWorker
from ui.workers.job_adapter import (
    gui_job_operation,
    gui_job_runtime,
    close_gui_job_runtime,
)

if TYPE_CHECKING:
    from core.project import Project


class AudioImportWorker(CancellableWorker):
    progress = Signal(int, int)
    audio_ready = Signal(object)
    outcome_ready = Signal(object)
    finished_signal = Signal()
    job_started = Signal(str, str)

    def __init__(
        self,
        file_path: Path,
        parent=None,
        *,
        session_id: str | None = None,
        project: "Project | None" = None,
    ):
        super().__init__(parent)
        if project is not None:
            project.session.assert_owner()
        self.task = AudioImportTask.from_path(file_path)
        self.session_id = (
            project.session.session_id if project is not None else session_id
        )
        runtime_identity = audio_jobs.audio_import_runtime()
        self.runtime_json = encode_object(runtime_identity)
        self.operation = gui_job_operation(
            OperationSpec.build(
                kind="audio_import",
                version=1,
                arguments={},
                inputs={"task": self.task.to_dict(), "runtime": runtime_identity},
                persistence="session_only",
                session_id=self.session_id,
                input_revision=str(project.mutation_generation)
                if project is not None
                else None,
            ),
            project.path if project is not None else None,
        )
        self.cache = (
            audio_jobs.GuiAudioImportCache(project, self.task, runtime_identity)
            if project is not None and project.path is not None
            else None
        )
        self.task_id: str | None = None
        self.job_status: str | None = None
        self.result: AudioImportOutcome | None = None
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
                progress(current / total if total else 1.0, "Importing audio")
                events.put((current, total))

            if encode_object(audio_jobs.audio_import_runtime()) != self.runtime_json:
                raise ValueError("Audio import runtime changed while queued")
            outcome = (
                self.cache.run(self.task, cancel, report)
                if self.cache is not None
                else run_audio_import(self.task, cancel_event=cancel, progress=report)
            )
            return {
                "success": outcome.status != "failed",
                "error": outcome.message,
                "outcome": asdict(outcome),
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
                self.result = AudioImportOutcome(
                    self.task.audio_source_id, "unprocessed"
                )
                self._log_cancelled()
                return
            payload = (row.result or {}).get("outcome")
            if payload is None:
                raise RuntimeError(row.error or "Audio import produced no result")
            self.result = AudioImportOutcome.from_dict(payload)
            self.outcome_ready.emit(self.result)
            if self.result.status == "succeeded" and not self.is_cancelled():
                self.audio_ready.emit(self.result.to_model(self.task))
                self._log_complete()
            elif self.result.status == "failed":
                self.error.emit(self.result.message or "Audio import failed")
        except Exception as exc:
            self.result = AudioImportOutcome(
                self.task.audio_source_id, "failed", message=str(exc)
            )
            if not self.is_cancelled():
                self.outcome_ready.emit(self.result)
                self.error.emit(str(exc))
        finally:
            try:
                if runtime is not None:
                    close_gui_job_runtime(runtime)
            finally:
                self._runtime = None
                self.finished_signal.emit()

"""Qt owner-thread notifications over the shared job lifecycle."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from core.jobs import JobRuntime, JobNotFoundError, JobStore
from core.jobs.spec import OperationSpec, encode_object
from core.jobs.store import STATUS_CANCELLED, STATUS_COMPLETED, TERMINAL_STATUSES


def gui_job_operation(operation: OperationSpec, path: Path | None) -> OperationSpec:
    """Persist saved-project computation history, retaining explicit GUI saves."""
    if path is None:
        return operation
    return replace(
        operation,
        kind=f"gui_{operation.kind}",
        persistence="job_history",
        publication="owner_thread",
        arguments_json=encode_object(
            {**operation.arguments, "project_path": str(path.expanduser().resolve())}
        ),
    )


def gui_job_runtime(operation: OperationSpec) -> JobRuntime:
    if operation.persistence == "session_only":
        return JobRuntime.for_session(max_workers=1)
    from core.settings import load_settings

    store = JobStore(load_settings().cache_dir / "jobs.db")
    store.mark_running_jobs_as_crashed()
    return JobRuntime(store, max_workers=1)


def close_gui_job_runtime(runtime: JobRuntime) -> None:
    if runtime.store.persistence == "session_only":
        runtime.close_session()
    else:
        runtime.shutdown()


class JobAdapter(QObject):
    """One active job per adapter; all signals identify their originating task."""

    progress = Signal(str, float, str)
    started = Signal(str, str)
    result_ready = Signal(str, dict)  # available terminal output, before outcome
    completed = Signal(str, dict)
    failed = Signal(str, str)
    cancelled = Signal(str)
    settled = Signal(str)

    def __init__(self, runtime: JobRuntime, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.task_id: str | None = None
        self._terminal = True
        self._last_progress: tuple[float, str] | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)

    def _assert_owner(self) -> None:
        if QThread.currentThread() != self.thread():
            raise RuntimeError("Job adapter must run on its owning Qt thread")

    def start(self, **submission) -> dict:
        self._assert_owner()
        if not self._terminal:
            raise RuntimeError("Job adapter already has an active job")
        result = self.runtime.submit(**submission)
        self.task_id = result["task_id"]
        self._terminal = False
        self._last_progress = None
        self._timer.start()
        self.started.emit(self.task_id, self.runtime.store.persistence)
        return result

    def cancel(self) -> bool:
        self._assert_owner()
        return bool(
            self.task_id and not self._terminal and self.runtime.cancel(self.task_id)
        )

    def _poll(self) -> None:
        self._assert_owner()
        if self._terminal or self.task_id is None:
            return
        try:
            row = self.runtime.store.get(self.task_id)
        except JobNotFoundError:
            task_id = self.task_id
            self._terminal = True
            self._timer.stop()
            self.failed.emit(task_id, "Job history is no longer available")
            self.settled.emit(task_id)
            return
        terminal = row.status in TERMINAL_STATUSES
        if terminal:
            self._terminal = True
            self._timer.stop()
        progress = (row.progress or 0.0, row.status_message or "")
        if progress != self._last_progress:
            self._last_progress = progress
            self.progress.emit(row.id, *progress)
        if not terminal:
            return
        payload = row.result
        if payload is not None:
            self.result_ready.emit(row.id, payload)
        if row.status == STATUS_COMPLETED:
            self.completed.emit(row.id, payload or {})
        elif row.status == STATUS_CANCELLED:
            self.cancelled.emit(row.id)
        else:
            self.failed.emit(row.id, row.error or row.status_message or row.status)
        self.settled.emit(row.id)

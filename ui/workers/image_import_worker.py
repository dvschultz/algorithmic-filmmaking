"""Qt adapter for shared detached still-image import."""

from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING
from PySide6.QtCore import Signal

from core.jobs import JobRuntime
from core.jobs.spec import OperationSpec, encode_object
from core.jobs import gui_image_import as image_jobs

from core.operations.image_import import (
    ImageImportTask,
    ImageImportOutcome,
    run_image_import,
)
from ui.workers.base import CancellableWorker
from ui.workers.job_adapter import (
    gui_job_operation,
    gui_job_runtime,
    close_gui_job_runtime,
)

if TYPE_CHECKING:
    from core.project import Project


class ImageImportWorker(CancellableWorker):
    progress = Signal(int, int)
    outcome_ready = Signal(object)
    job_started = Signal(str, str)

    def __init__(
        self,
        paths: list[Path],
        output_dir: Path,
        parent=None,
        *,
        copy_files: bool = False,
        validate_paths: bool = False,
        project: "Project | None" = None,
    ) -> None:
        super().__init__(parent)
        if project is not None:
            project.session.assert_owner()
        self.task = ImageImportTask.from_paths(
            paths, output_dir, copy_files=copy_files, validate_paths=validate_paths
        )
        self.result: ImageImportOutcome | None = None
        runtime_identity = image_jobs.image_import_runtime()
        self.runtime_json = encode_object(runtime_identity)
        self.operation = gui_job_operation(
            OperationSpec.build(
                kind="import_images",
                version=1,
                arguments={"copy_files": copy_files},
                inputs={"task": self.task.to_dict(), "runtime": runtime_identity},
                persistence="session_only",
                session_id=project.session.session_id if project else None,
                input_revision=str(project.mutation_generation) if project else None,
            ),
            project.path if project else None,
        )
        self.cache = (
            image_jobs.GuiImageImportCache(project, self.task, runtime_identity)
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
        runtime = None
        events: Queue[tuple[int, int]] = Queue()

        def compute(progress, cancel):
            def report(current, total):
                progress(current / total if total else 1.0, "Importing images")
                events.put((current, total))

            if encode_object(image_jobs.image_import_runtime()) != self.runtime_json:
                raise ValueError("Image import runtime changed while queued")
            outcome = (
                self.cache.run(self.task, cancel, report)
                if self.cache is not None
                else run_image_import(self.task, cancel_event=cancel, progress=report)
            )
            return {
                "success": outcome.status != "failed",
                "error": "; ".join(outcome.errors) or None,
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
                self.result = ImageImportOutcome(self.task.request_id, "unprocessed")
            else:
                payload = (row.result or {}).get("outcome")
                if payload is None:
                    raise RuntimeError(row.error or "Image import produced no result")
                self.result = ImageImportOutcome.from_dict(payload)
        except Exception as exc:
            self.result = ImageImportOutcome(
                self.task.request_id,
                "unprocessed" if self.is_cancelled() else "failed",
                errors=(str(exc),),
            )
        finally:
            try:
                if runtime is not None:
                    close_gui_job_runtime(runtime)
            finally:
                self._runtime = None
        self.outcome_ready.emit(self.result)

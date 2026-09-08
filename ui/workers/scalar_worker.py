"""Scalar analysis worker for the shared GUI/agent analysis controller."""

from dataclasses import asdict
from typing import TYPE_CHECKING, Callable
from threading import Event

from PySide6.QtCore import Signal

from core.analysis_records import AnalysisSnapshot
from core.jobs import JobRuntime
from core.jobs.gui_scalars import GuiScalarCache
from core.jobs.spec import OperationSpec
from core.operations.scalars import (
    ScalarBatchApplication,
    ScalarOperation,
    ScalarOutcome,
    run_scalars,
    scalar_task,
    scalar_runtime,
)
from ui.workers.base import CancellableWorker
from ui.workers.job_adapter import (
    gui_job_operation,
    gui_job_runtime,
    close_gui_job_runtime,
)

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip


class ScalarAnalysisWorker(CancellableWorker):
    progress = Signal(int, int)

    def __init__(
        self,
        clips: list["Clip"],
        *,
        project: "Project",
        operation: ScalarOperation,
        skip_existing: bool = True,
        num_samples: int = 5,
    ) -> None:
        super().__init__()
        project.session.assert_owner()
        self.tasks = tuple(
            scalar_task(
                clip,
                project.sources_by_id.get(clip.source_id),
                operation,
                skip_existing=skip_existing,
                num_samples=num_samples,
            )
            for clip in clips
        )
        self.application = ScalarBatchApplication(project, self.tasks)
        self.result: tuple[ScalarOutcome, ...] = ()
        self.cache: GuiScalarCache | None = None
        self.task_id: str | None = None
        self.job_status: str | None = None
        self._runtime: JobRuntime | None = None
        self.operation = gui_job_operation(
            OperationSpec.build(
                kind=f"scalar_{operation}",
                version=1,
                arguments={"clip_ids": [task.clip_id for task in self.tasks]},
                inputs={
                    "tasks": [asdict(task) for task in self.tasks],
                    "runtime": scalar_runtime(operation),
                },
                persistence="session_only",
                session_id=project.session.session_id,
                input_revision=str(project.mutation_generation),
            ),
            project.path,
        )
        if project.path is not None:
            self.cache = GuiScalarCache(
                project.path,
                project.metadata.id,
                {clip.id: clip.source_id for clip in clips},
                project.metadata.job_results,
                operation=operation,
                media_stamps={
                    path: stamp
                    for task in self.tasks
                    for _, path, stamp in AnalysisSnapshot.from_json(
                        task.snapshot_json
                    ).inputs.files
                },
            )

    def cancel(self) -> None:
        super().cancel()
        if self._runtime is not None and self.task_id is not None:
            self._runtime.cancel(self.task_id)

    def run(self) -> None:
        runtime = None

        def compute(report: Callable[[float, str], None], cancel: Event) -> dict:
            collected: dict[str, ScalarOutcome] = {}

            def deliver(outcome: ScalarOutcome) -> None:
                collected[outcome.clip_id] = outcome

            def progress(current: int, total: int) -> None:
                report(current / total if total else 1.0, "Measuring clip scalars")
                self.progress.emit(current, total)

            try:
                if self.cache is not None:
                    outcomes = self.cache.run(self.tasks, cancel, deliver, progress)
                else:

                    def receive(outcome: ScalarOutcome) -> None:
                        deliver(outcome)
                        progress(len(collected), len(self.tasks))

                    outcomes = run_scalars(
                        self.tasks, cancel_event=cancel, on_outcome=receive
                    )
                return {"outcomes": [asdict(outcome) for outcome in outcomes]}
            except Exception as exc:
                return {
                    "success": False,
                    "error": str(exc),
                    "outcomes": [
                        asdict(
                            collected.get(
                                task.clip_id,
                                ScalarOutcome(
                                    task.clip_id,
                                    task.operation,
                                    "failed",
                                    message=str(exc),
                                ),
                            )
                        )
                        for task in self.tasks
                    ],
                }

        try:
            runtime = gui_job_runtime(self.operation)
            self._runtime = runtime
            submitted = runtime.submit(
                kind=self.operation.kind,
                args=self.operation.arguments,
                operation=self.operation,
                run=compute,
                cancellation_event=self._cancel_event,
                project_path=self.operation.arguments.get("project_path"),
            )
            self.task_id = submitted["task_id"]
            runtime.shutdown()
            row = runtime.store.get(self.task_id)
            self.job_status = row.status
            self.result = tuple(
                ScalarOutcome(**outcome)
                for outcome in (row.result or {}).get("outcomes", [])
            )
            if row.status == "failed" and not self.is_cancelled():
                self.error.emit(row.error or "Scalar analysis failed")
        except Exception as exc:
            if not self.is_cancelled():
                self.error.emit(str(exc))
        finally:
            if runtime is not None:
                close_gui_job_runtime(runtime)
            self._runtime = None

"""Qt scheduling and signals for the shared color operation."""

from typing import TYPE_CHECKING, Optional
from dataclasses import asdict
from queue import Empty, Queue

from PySide6.QtCore import Signal

from core.analysis_target import AnalysisTarget
from core.jobs import JobRuntime
from core.jobs.colors import color_job_spec
from core.operations.colors import (
    ColorApplication,
    compute_colors,
    request_from_targets,
)
from core.operations.contracts import ColorOutcome, ColorResult
from ui.workers.base import CancellableWorker, summarize_clip_errors

if TYPE_CHECKING:
    from core.project import Project


class ColorAnalysisWorker(CancellableWorker):
    """Snapshot on construction; compute off-thread; apply on the project thread.

    ``color_ready`` remains available for consumers of individual palettes.
    The main window uses ``result_ready`` to commit a whole batch exactly once.
    """

    progress = Signal(int, int)
    color_ready = Signal(str, list)
    result_ready = Signal(object, object)  # ColorApplication, ColorResult
    analysis_completed = Signal()
    job_started = Signal(str, str)  # task ID, persistence

    def __init__(
        self,
        clips: list,
        parallelism: int = 4,
        skip_existing: bool = True,
        analysis_targets: Optional[list] = None,
        sources_by_id: Optional[dict] = None,
        parent=None,
        *,
        project: Optional["Project"] = None,
    ) -> None:
        super().__init__(parent)
        self._parallelism = min(max(1, parallelism), 8)
        targets = analysis_targets
        if targets is None:
            sources = sources_by_id or {}
            targets = []
            for clip in clips:
                target = AnalysisTarget.from_clip(clip, sources.get(clip.source_id))
                target.image_path = None
                targets.append(target)
        self.request = request_from_targets(
            targets, skip_existing=skip_existing, skip_empty=True
        )
        self.application = (
            ColorApplication(project, self.request) if project is not None else None
        )
        self.operation = color_job_spec(
            self.request,
            arguments={
                "num_colors": self.request.num_colors,
                "skip_existing": self.request.skip_existing,
                "skip_empty": self.request.skip_empty,
                "parallelism": self._parallelism,
            },
            persistence="session_only",
            session_id=project.session.session_id if project is not None else None,
            input_revision=str(project.mutation_generation)
            if project is not None
            else None,
        )
        self.result: Optional[ColorResult] = None
        self.task_id: str | None = None
        self.job_status: str | None = None
        self._runtime: JobRuntime | None = None

    def cancel(self) -> None:
        super().cancel()
        runtime = self._runtime
        if runtime is not None and self.task_id is not None:
            runtime.cancel(self.task_id)

    def _on_progress(self, completed: int, total: int, outcome: ColorOutcome) -> None:
        if outcome.status == "succeeded":
            self.color_ready.emit(outcome.target_id, list(outcome.colors))
        self.progress.emit(completed, total)

    def run(self) -> None:
        self._log_start()
        runtime: JobRuntime | None = None
        events: Queue[tuple[int, int, ColorOutcome]] = Queue()

        def compute(progress, cancel):
            def report(completed, total, outcome):
                progress(completed / total if total else 1.0, outcome.target_id)
                events.put((completed, total, outcome))

            return asdict(
                compute_colors(
                    self.request,
                    parallelism=self._parallelism,
                    cancel_event=cancel,
                    progress_callback=report,
                )
            )

        try:
            runtime = JobRuntime.for_session(max_workers=1)
            self._runtime = runtime
            submission = runtime.submit(
                kind="analyze_colors",
                args=self.operation.arguments,
                operation=self.operation,
                run=compute,
                cancellation_event=self._cancel_event,
            )
            self.task_id = submission["task_id"]
            self.job_started.emit(self.task_id, submission["persistence"])
            while runtime.is_handle_live(self.task_id):
                try:
                    self._on_progress(*events.get(timeout=0.05))
                except Empty:
                    pass
            runtime.shutdown()
            while not events.empty():
                self._on_progress(*events.get_nowait())
            row = runtime.store.get(self.task_id)
            self.job_status = row.status
            if row.status == "failed":
                raise RuntimeError(row.error or "Color analysis failed")
            payload = row.result
            self.result = ColorResult(
                self.request.request_id,
                tuple(
                    ColorOutcome(
                        target_id=o["target_id"],
                        status=o["status"],
                        colors=tuple(tuple(c) for c in o["colors"]),
                        code=o["code"],
                        message=o["message"],
                        record_json=o.get("record_json"),
                    )
                    for o in payload["outcomes"]
                )
                if payload
                else tuple(
                    ColorOutcome(target.target_id, "unprocessed", code="cancelled")
                    for target in self.request.targets
                ),
            )
            errors = [
                (o.target_id, o.message or o.code or "Color extraction failed")
                for o in self.result.outcomes
                if o.status == "failed"
            ]
            if errors:
                self.error.emit(
                    summarize_clip_errors(errors, operation_label="Color extraction")
                )
            self.result_ready.emit(self.application, self.result)
        except Exception as exc:
            self._log_error(str(exc))
            self.error.emit(str(exc))
        finally:
            try:
                if runtime is not None:
                    runtime.close_session()
            finally:
                self._runtime = None
                self.analysis_completed.emit()
                self._log_complete()

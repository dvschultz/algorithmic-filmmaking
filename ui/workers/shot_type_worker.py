"""Background worker for shot type classification.

Runs the shared detached operation on a background thread.
"""

import logging
from dataclasses import asdict
from queue import Empty, Queue
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import Signal

from ui.workers.base import CancellableWorker
from core.jobs import JobRuntime
from core.jobs.gui_shots import GuiShotCache, shot_target_snapshot
from core.jobs.shots import _runtime, _task_data
from core.jobs.spec import OperationSpec
from ui.workers.job_adapter import (
    gui_job_operation,
    gui_job_runtime,
    close_gui_job_runtime,
)

if TYPE_CHECKING:
    from core.project import Project

from core.operations.shots import (
    ShotTypeTask,
    shot_task,
    ShotTypeOptions,
    ShotTypeOutcome,
    run_shot_types,
)

logger = logging.getLogger(__name__)


class ShotTypeWorker(CancellableWorker):
    """Background worker for local SigLIP or cloud vision shot classification.

    Supports tiered processing:
    - Local: pinned SigLIP zero-shot classification from thumbnails
    - Cloud: configured vision model with local fallback

    Supports both Clip and Frame inputs via AnalysisTarget.

    Inference is serialized across jobs because the local model is shared.
    The parallelism argument is retained for existing callers.

    Signals:
        progress: Emitted with (current, total) during processing
        shot_type_ready: Emitted with (clip_id, shot_type, confidence)
        analysis_completed: Emitted when all clips are processed
        error: Emitted with error message string on failure (inherited)
    """

    progress = Signal(int, int)  # current, total
    shot_type_ready = Signal(str, str, float)  # clip_id, shot_type, confidence
    analysis_completed = Signal()
    outcome_ready = Signal(object)

    @staticmethod
    def _summarize_errors(errors: list[tuple[str, str]]) -> str:
        """Create a readable summary for one or more clip failures."""
        preview = "\n".join(
            f"- {clip_id}: {message}" for clip_id, message in errors[:3]
        )
        if len(errors) == 1:
            return f"Shot type classification failed:\n\n{preview}"

        remaining = len(errors) - 3
        extra = f"\n- ...and {remaining} more clip(s)" if remaining > 0 else ""
        return (
            f"Shot type classification failed for {len(errors)} clips:\n\n"
            f"{preview}{extra}"
        )

    def __init__(
        self,
        clips: list,
        sources_by_id: dict,
        parallelism: int = 1,
        skip_existing: bool = True,
        analysis_targets: Optional[list] = None,
        parent=None,
        *,
        project: Optional["Project"] = None,
        options: Optional[ShotTypeOptions] = None,
    ) -> None:
        super().__init__(parent)
        self._parallelism = 1
        self.options = options or ShotTypeOptions.from_settings()
        self.result: tuple[ShotTypeOutcome, ...] = ()
        if analysis_targets:
            self._tasks = self._build_tasks_from_targets(
                analysis_targets, skip_existing
            )
        else:
            self._tasks = self._build_tasks(clips, sources_by_id, skip_existing)
        self.operation = gui_job_operation(
            OperationSpec.build(
                kind="shots",
                version=2,
                arguments={
                    "targets": [[task.target_type, task.clip_id] for task in self.tasks]
                },
                inputs={
                    "tasks": [_task_data(task) for task in self.tasks],
                    "options": asdict(self.options),
                    "runtime": _runtime(),
                    "previous": [
                        shot_target_snapshot(project, task) for task in self.tasks
                    ]
                    if project
                    else [],
                },
                persistence="session_only",
                session_id=project.session.session_id if project else None,
                input_revision=str(project.mutation_generation) if project else None,
            ),
            project.path if project else None,
        )
        self.cache = (
            GuiShotCache(project, self.tasks, self.options)
            if project is not None and project.path is not None
            else None
        )
        self.task_id: str | None = None
        self.job_status: str | None = None
        self._runtime: JobRuntime | None = None

    def _build_tasks(
        self, clips: list, sources_by_id: dict, skip_existing: bool
    ) -> list[ShotTypeTask]:
        """Build immutable task list from clips."""
        tasks = []
        for clip in clips:
            if not clip.thumbnail_path or not clip.thumbnail_path.exists():
                logger.warning(f"Skipping clip {clip.id}: thumbnail not found")
                continue

            source = sources_by_id.get(clip.source_id)
            tasks.append(
                shot_task(clip, source, skip_existing=skip_existing)
            )
        return tasks

    def _build_tasks_from_targets(
        self, targets: list, skip_existing: bool
    ) -> list[ShotTypeTask]:
        """Build immutable task list from AnalysisTarget objects."""
        tasks = []
        for target in targets:
            image_path = target.image_path
            if not image_path or not image_path.exists():
                logger.warning(f"Skipping target {target.id}: image not found")
                continue
            tasks.append(
                shot_task(target, skip_existing=skip_existing)
            )
        return tasks

    @property
    def tasks(self) -> tuple[ShotTypeTask, ...]:
        return tuple(self._tasks)

    def cancel(self) -> None:
        super().cancel()
        if self._runtime is not None and self.task_id is not None:
            self._runtime.cancel(self.task_id)

    def _prepare(self) -> bool:
        if any(not task.media_current() for task in self.tasks):
            raise RuntimeError("Shot media changed while queued")
        return not self.is_cancelled()

    def run(self) -> None:
        """Relay shared-job outcomes; model publication stays on the owner thread."""
        self._log_start()
        runtime = None
        events: Queue = Queue()
        errors: list[tuple[str, str]] = []

        def emit(event) -> None:
            kind, outcome = event
            if kind == "progress":
                self.progress.emit(*outcome)
                return
            if outcome.can_apply:
                self.outcome_ready.emit(outcome)
            if outcome.status == "succeeded":
                self.shot_type_ready.emit(
                    outcome.clip_id, outcome.shot_type, outcome.confidence
                )
            elif outcome.status == "failed":
                errors.append(
                    (
                        outcome.clip_id,
                        outcome.message or outcome.code or "Analysis failed",
                    )
                )

        def compute(progress, cancel):
            collected = {}

            def deliver(outcome):
                collected[outcome.target_type, outcome.clip_id] = outcome
                events.put(("outcome", outcome))

            def report(current, total):
                progress(
                    current / total if total else 1.0,
                    f"Classifying shots ({current}/{total})",
                )
                events.put(("progress", (current, total)))

            try:
                if self.cache is not None:
                    outcomes = self.cache.run(
                        self.tasks, cancel, self._prepare, deliver, report
                    )
                elif self._prepare():
                    outcomes = run_shot_types(
                        self.tasks,
                        self.options,
                        cancel_event=cancel,
                        on_outcome=deliver,
                        progress=report,
                    )
                else:
                    outcomes = tuple(
                        ShotTypeOutcome(
                            task.clip_id,
                            "unprocessed",
                            code="cancelled",
                            target_type=task.target_type,
                        )
                        for task in self.tasks
                    )
                return {"outcomes": [asdict(outcome) for outcome in outcomes]}
            except Exception as exc:
                for task in self.tasks:
                    if (task.target_type, task.clip_id) not in collected:
                        deliver(
                            ShotTypeOutcome(
                                task.clip_id,
                                "failed",
                                code="classification_failed",
                                message=str(exc),
                                target_type=task.target_type,
                            )
                        )
                return {
                    "success": False,
                    "error": str(exc),
                    "outcomes": [
                        asdict(collected[task.target_type, task.clip_id])
                        for task in self.tasks
                    ],
                }

        try:
            self.progress.emit(0, len(self.tasks))
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
                ShotTypeOutcome.from_dict(value)
                for value in (row.result or {}).get("outcomes", [])
            )
            if row.status == "cancelled" and not self.result:
                self.result = tuple(
                    ShotTypeOutcome(
                        task.clip_id,
                        "unprocessed",
                        code="cancelled",
                        target_type=task.target_type,
                    )
                    for task in self.tasks
                )
            if row.status == "failed" and not self.result:
                raise RuntimeError(row.error or "Shot classification failed")
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            try:
                if runtime is not None:
                    close_gui_job_runtime(runtime)
            finally:
                self._runtime = None
                if errors and not self.is_cancelled():
                    self.error.emit(self._summarize_errors(errors))
                self.analysis_completed.emit()
                self._log_complete()

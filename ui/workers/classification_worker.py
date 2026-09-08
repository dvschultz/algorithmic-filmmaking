"""Background worker for frame classification using MobileNet.

Runs image classification on multiple clips in a background thread,
using shared serial inference.
"""

import logging
from dataclasses import asdict
from queue import Empty, Queue
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import Signal

from core.jobs import JobRuntime
from core.jobs.classification import _runtime, _task_data
from core.jobs.gui_classification import GuiClassificationCache
from core.jobs.media import media_stamp
from core.jobs.spec import OperationSpec
from ui.workers.job_adapter import (
    gui_job_operation,
    gui_job_runtime,
    close_gui_job_runtime,
)

if TYPE_CHECKING:
    from core.project import Project

from core.operations.classification import (
    ClassificationTask,
    ClassificationOptions,
    ClassificationOutcome,
    compute_classification,
    classification_task,
    run_classification,
)

from ui.workers.base import CancellableWorker, summarize_clip_errors

logger = logging.getLogger(__name__)


def _summarize_errors(errors: list[tuple[str, str]]) -> str:
    """Return a compact user-facing summary for a batch failure."""
    return summarize_clip_errors(errors, operation_label="Content classification")


class ClassificationWorker(CancellableWorker):
    """Background worker for frame classification using MobileNet.

    MobileNet inference is serialized across shared classification jobs.
    The parallelism argument is retained for existing callers.

    Supports both Clip and Frame inputs via AnalysisTarget.

    Signals:
        progress: Emitted with (current, total) during processing
        labels_ready: Emitted with (clip_id, labels) when a clip finishes
        classification_completed: Emitted when all clips are processed
        error: Emitted with error message string on failure (inherited)
    """

    progress = Signal(int, int)  # current, total
    labels_ready = Signal(str, list)  # clip_id, [(label, confidence), ...]
    outcome_ready = Signal(object)
    classification_completed = Signal()

    def __init__(
        self,
        clips: list,
        top_k: int = 5,
        threshold: float = 0.1,
        parallelism: int = 1,
        skip_existing: bool = True,
        analysis_targets: Optional[list] = None,
        parent=None,
        *,
        project: Optional["Project"] = None,
    ) -> None:
        super().__init__(parent)
        self._top_k = top_k
        self._threshold = threshold
        self._parallelism = 1
        self.options = ClassificationOptions(top_k, threshold)
        self.project = project
        self.result: tuple[ClassificationOutcome, ...] = ()
        if analysis_targets:
            self._tasks = self._build_tasks_from_targets(
                analysis_targets, skip_existing
            )
        else:
            self._tasks = self._build_tasks(clips, skip_existing)

        targets = (
            {
                task.clip_id: (
                    project.frames_by_id
                    if task.target_type == "frame"
                    else project.clips_by_id
                )[task.clip_id]
                for task in self.tasks
            }
            if project is not None
            else {}
        )
        sources_by_id = project.sources_by_id if project is not None else {}
        previous = {}
        paths = {
            task.thumbnail_path
            for task in self.tasks
            if task.thumbnail_path is not None
        }
        for cid, target in targets.items():
            source = sources_by_id.get(target.source_id or "")
            if source is not None:
                paths.add(source.file_path)
            previous[cid] = {
                "labels": target.object_labels,
                "source_path": str(source.file_path) if source else None,
                "start_frame": getattr(target, "start_frame", None),
                "end_frame": getattr(target, "end_frame", None),
                "fps": source.fps if source else None,
                "frame_clip_id": getattr(target, "clip_id", None),
                "frame_number": getattr(target, "frame_number", None),
            }
        self._media_stamps = {path: media_stamp(path) for path in paths}
        self.operation = gui_job_operation(
            OperationSpec.build(
                kind="classification",
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
        self.cache: GuiClassificationCache | None = None
        if project is not None and project.path is not None:
            self.cache = GuiClassificationCache(
                project.path,
                project.metadata.id,
                {cid: target.source_id or "" for cid, target in targets.items()},
                project.metadata.job_results,
                options=self.options,
                previous_results=previous,
                media_stamps=self._media_stamps,
            )

    def _build_tasks(
        self, clips: list, skip_existing: bool
    ) -> list[ClassificationTask]:
        """Build immutable task list from clips."""
        tasks = []
        for clip in clips:
            if not clip.thumbnail_path or not clip.thumbnail_path.exists():
                logger.warning(f"Skipping clip {clip.id}: thumbnail not found")
                continue
            tasks.append(
                classification_task(
                    clip,
                    self.project.sources_by_id.get(clip.source_id)
                    if self.project
                    else None,
                    skip_existing=skip_existing,
                )
            )
        return tasks

    def _build_tasks_from_targets(
        self, targets: list, skip_existing: bool
    ) -> list[ClassificationTask]:
        """Build immutable task list from AnalysisTarget objects."""
        tasks = []
        for target in targets:
            image_path = target.image_path
            if not image_path or not image_path.exists():
                logger.warning(f"Skipping target {target.id}: image not found")
                continue
            tasks.append(
                classification_task(
                    target,
                    self.project.sources_by_id.get(target.source_id)
                    if self.project
                    else None,
                    skip_existing=skip_existing,
                )
            )
        return tasks

    @property
    def tasks(self) -> tuple[ClassificationTask, ...]:
        return tuple(self._tasks)

    def _process_task(
        self, task: ClassificationTask
    ) -> tuple[str, Optional[list], Optional[str]]:
        outcome = compute_classification(task, self.options, self._cancel_event)
        if outcome.status == "succeeded":
            return task.clip_id, list(outcome.labels), None
        return (
            task.clip_id,
            None,
            "Cancelled"
            if outcome.status == "unprocessed"
            else outcome.message or outcome.code,
        )

    def cancel(self) -> None:
        super().cancel()
        if self._runtime is not None and self.task_id is not None:
            self._runtime.cancel(self.task_id)

    def _prepare(self) -> bool:
        if any(
            media_stamp(path) != stamp for path, stamp in self._media_stamps.items()
        ):
            raise RuntimeError("Classification media changed while queued")
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
                    self.outcome_ready.emit(value)
                if value.status == "succeeded":
                    self.labels_ready.emit(value.clip_id, list(value.labels))
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
                    f"Classification ({current}/{total})",
                )
                events.put(("progress", (current, total)))

            try:
                if self.cache is not None:
                    outcomes = self.cache.run(
                        self.tasks, cancel, self._prepare, deliver, report
                    )
                elif self._prepare():
                    outcomes = run_classification(
                        self.tasks,
                        self.options,
                        cancel_event=cancel,
                        on_outcome=deliver,
                        progress=report,
                    )
                else:
                    outcomes = tuple(
                        ClassificationOutcome(
                            task.clip_id, "unprocessed", code="cancelled"
                        )
                        for task in self.tasks
                    )
                return {"outcomes": [asdict(outcome) for outcome in outcomes]}
            except Exception as exc:
                for task in self.tasks:
                    if task.clip_id not in collected:
                        deliver(
                            ClassificationOutcome(
                                task.clip_id,
                                "failed",
                                code="classification_failed",
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
                ClassificationOutcome.from_dict(value)
                for value in (row.result or {}).get("outcomes", [])
            )
            if row.status == "cancelled" and not self.result:
                self.result = tuple(
                    ClassificationOutcome(task.clip_id, "unprocessed", code="cancelled")
                    for task in self.tasks
                )
            if row.status == "failed" and not self.result:
                raise RuntimeError(row.error or "Classification failed")
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            try:
                if runtime is not None:
                    close_gui_job_runtime(runtime)
            finally:
                self._runtime = None
                if errors and not self.is_cancelled():
                    self.error.emit(_summarize_errors(errors))
                self.classification_completed.emit()
                self._log_complete()

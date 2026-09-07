"""Background worker for frame classification using MobileNet.

Runs image classification on multiple clips in a background thread,
using shared serial inference.
"""

import logging
from typing import Optional

from PySide6.QtCore import Signal

from core.operations.classification import (
    ClassificationTask,
    ClassificationOptions,
    ClassificationOutcome,
    compute_classification,
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
    ):
        super().__init__(parent)
        self._top_k = top_k
        self._threshold = threshold
        self._parallelism = 1
        self.options = ClassificationOptions(top_k, threshold)
        self.result: tuple[ClassificationOutcome, ...] = ()
        if analysis_targets:
            self._tasks = self._build_tasks_from_targets(
                analysis_targets, skip_existing
            )
        else:
            self._tasks = self._build_tasks(clips, skip_existing)

    def _build_tasks(
        self, clips: list, skip_existing: bool
    ) -> list[ClassificationTask]:
        """Build immutable task list from clips."""
        tasks = []
        for clip in clips:
            if skip_existing and clip.object_labels is not None:
                continue
            if not clip.thumbnail_path or not clip.thumbnail_path.exists():
                logger.warning(f"Skipping clip {clip.id}: thumbnail not found")
                continue
            tasks.append(
                ClassificationTask(
                    clip_id=clip.id,
                    thumbnail_path=clip.thumbnail_path,
                )
            )
        return tasks

    def _build_tasks_from_targets(
        self, targets: list, skip_existing: bool
    ) -> list[ClassificationTask]:
        """Build immutable task list from AnalysisTarget objects."""
        tasks = []
        for target in targets:
            if skip_existing and target.object_labels is not None:
                continue
            image_path = target.image_path
            if not image_path or not image_path.exists():
                logger.warning(f"Skipping target {target.id}: image not found")
                continue
            tasks.append(
                ClassificationTask(
                    clip_id=target.id,
                    thumbnail_path=image_path,
                    target_type=target.target_type,
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

    def run(self) -> None:
        """Run shared computation; always settle the compatibility completion signal."""
        self._log_start()
        errors: list[tuple[str, str]] = []
        try:
            self.progress.emit(0, len(self.tasks))

            def deliver(outcome: ClassificationOutcome) -> None:
                if outcome.status == "succeeded":
                    self.labels_ready.emit(outcome.clip_id, list(outcome.labels))
                elif outcome.status == "failed":
                    errors.append(
                        (
                            outcome.clip_id,
                            outcome.message or outcome.code or "Classification failed",
                        )
                    )

            self.result = run_classification(
                self.tasks,
                self.options,
                cancel_event=self._cancel_event,
                on_outcome=deliver,
                progress=self.progress.emit,
            )
            if errors and not self.is_cancelled():
                self.error.emit(_summarize_errors(errors))
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.classification_completed.emit()
            self._log_complete()

"""Background worker for object detection using YOLOv8.

Runs object detection on multiple clips in a background thread,
using shared serial inference.
"""

import logging
from typing import Optional

from PySide6.QtCore import Signal

from core.operations.object_detection import (
    ObjectDetectionTask,
    ObjectDetectionOptions,
    ObjectDetectionOutcome,
    compute_object_detection,
    run_object_detection,
)
from ui.workers.base import CancellableWorker, summarize_clip_errors

logger = logging.getLogger(__name__)


def _summarize_errors(errors: list[tuple[str, str]]) -> str:
    """Return a compact user-facing summary for a batch failure."""
    return summarize_clip_errors(errors, operation_label="Object detection")


class ObjectDetectionWorker(CancellableWorker):
    """Background worker for object detection using YOLOv8.

    Inference is serialized across callers sharing the YOLO singleton.
    The parallelism argument remains accepted for compatibility.

    Supports both Clip and Frame inputs via AnalysisTarget.

    Signals:
        progress: Emitted with (current, total) during processing
        objects_ready: Emitted with (clip_id, detections, person_count)
        detection_completed: Emitted when all clips are processed
        error: Emitted with error message string on failure (inherited)
    """

    progress = Signal(int, int)  # current, total
    objects_ready = Signal(str, list, int)  # clip_id, detections, person_count
    detection_completed = Signal()

    def __init__(
        self,
        clips: list,
        confidence: float = 0.5,
        detect_all: bool = True,
        parallelism: int = 1,
        skip_existing: bool = True,
        analysis_targets: Optional[list] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._confidence = confidence
        self._detect_all = detect_all
        self._parallelism = 1
        self.options = ObjectDetectionOptions(confidence, detect_all)
        self.result: tuple[ObjectDetectionOutcome, ...] = ()
        if analysis_targets:
            self._tasks = self._build_tasks_from_targets(
                analysis_targets, skip_existing
            )
        else:
            self._tasks = self._build_tasks(clips, skip_existing)

    def _build_tasks(
        self, clips: list, skip_existing: bool
    ) -> list[ObjectDetectionTask]:
        """Build immutable task list from clips."""
        tasks = []
        for clip in clips:
            existing = (
                clip.detected_objects if self.options.detect_all else clip.person_count
            )
            if skip_existing and existing is not None:
                continue
            if not clip.thumbnail_path or not clip.thumbnail_path.exists():
                logger.warning(f"Skipping clip {clip.id}: thumbnail not found")
                continue
            tasks.append(
                ObjectDetectionTask(
                    clip_id=clip.id,
                    thumbnail_path=clip.thumbnail_path,
                )
            )
        return tasks

    def _build_tasks_from_targets(
        self, targets: list, skip_existing: bool
    ) -> list[ObjectDetectionTask]:
        """Build immutable task list from AnalysisTarget objects."""
        tasks = []
        for target in targets:
            existing = (
                target.detected_objects
                if self.options.detect_all
                else target.person_count
            )
            if skip_existing and existing is not None:
                continue
            image_path = target.image_path
            if not image_path or not image_path.exists():
                logger.warning(f"Skipping target {target.id}: image not found")
                continue
            tasks.append(
                ObjectDetectionTask(
                    clip_id=target.id,
                    thumbnail_path=image_path,
                    target_type=target.target_type,
                )
            )
        return tasks

    @property
    def tasks(self) -> tuple[ObjectDetectionTask, ...]:
        return tuple(self._tasks)

    def _process_task(
        self, task: ObjectDetectionTask
    ) -> tuple[str, Optional[list], Optional[int], Optional[str]]:
        outcome = compute_object_detection(task, self.options, self._cancel_event)
        if outcome.status == "succeeded":
            return task.clip_id, outcome.detection_dicts(), outcome.person_count, None
        return (
            task.clip_id,
            None,
            None,
            (
                "Cancelled"
                if outcome.status == "unprocessed"
                else outcome.message or outcome.code
            ),
        )

    def run(self) -> None:
        """Run shared inference and always settle the compatibility signal."""
        self._log_start()
        errors: list[tuple[str, str]] = []
        try:
            self.progress.emit(0, len(self.tasks))

            def deliver(outcome: ObjectDetectionOutcome) -> None:
                if outcome.status == "succeeded":
                    self.objects_ready.emit(
                        outcome.clip_id, outcome.detection_dicts(), outcome.person_count
                    )
                elif outcome.status == "failed":
                    errors.append(
                        (
                            outcome.clip_id,
                            outcome.message
                            or outcome.code
                            or "Object detection failed",
                        )
                    )

            self.result = run_object_detection(
                self.tasks,
                self.options,
                cancel_event=self._cancel_event,
                on_outcome=deliver,
                progress=self.progress.emit,
            )
            if errors and not self.is_cancelled():
                model_failure = next(
                    (
                        outcome
                        for outcome in self.result
                        if outcome.code == "model_load_failed"
                    ),
                    None,
                )
                if model_failure is not None:
                    self.error.emit(
                        model_failure.message or "Object detection model unavailable"
                    )
                else:
                    self.error.emit(_summarize_errors(errors))
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.detection_completed.emit()
            self._log_complete()

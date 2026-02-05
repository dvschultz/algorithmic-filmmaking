"""Background worker for object detection using YOLOv8.

Runs object detection on multiple clips in a background thread,
using ThreadPoolExecutor for parallelism.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal

from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectDetectionTask:
    """Immutable task data for thread pool execution."""

    clip_id: str
    thumbnail_path: Path


class ObjectDetectionWorker(CancellableWorker):
    """Background worker for object detection using YOLOv8.

    Uses ThreadPoolExecutor for parallel processing. Default parallelism is 1
    because the YOLO model singleton is not thread-safe for inference.

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
        parent=None,
    ):
        super().__init__(parent)
        self._confidence = confidence
        self._detect_all = detect_all
        self._parallelism = min(max(1, parallelism), 4)
        self._tasks = self._build_tasks(clips, skip_existing)

    def _build_tasks(
        self, clips: list, skip_existing: bool
    ) -> list[ObjectDetectionTask]:
        """Build immutable task list from clips."""
        tasks = []
        for clip in clips:
            if skip_existing and clip.detected_objects is not None:
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

    def _process_task(
        self, task: ObjectDetectionTask
    ) -> tuple[str, Optional[list], Optional[int], Optional[str]]:
        """Process a single task (runs in thread pool).

        Returns:
            Tuple of (clip_id, detections, person_count, error_message)
        """
        if self.is_cancelled():
            return task.clip_id, None, None, "Cancelled"

        try:
            from core.analysis.detection import detect_objects, count_people

            if self._detect_all:
                detections = detect_objects(
                    task.thumbnail_path,
                    confidence_threshold=self._confidence,
                )
                person_count = sum(
                    1 for d in detections if d["label"] == "person"
                )
            else:
                detections = []
                person_count = count_people(
                    task.thumbnail_path,
                    confidence_threshold=self._confidence,
                )
            return task.clip_id, detections, person_count, None
        except Exception as e:
            return task.clip_id, None, None, str(e)

    def run(self):
        """Execute object detection on all clips."""
        self._log_start()

        total = len(self._tasks)
        if total == 0:
            logger.info("No clips to process for object detection")
            self.detection_completed.emit()
            self._log_complete()
            return

        logger.info(
            f"Starting object detection: {total} clips, "
            f"parallelism={self._parallelism}"
        )

        completed = 0

        with ThreadPoolExecutor(max_workers=self._parallelism) as executor:
            future_to_task = {
                executor.submit(self._process_task, task): task
                for task in self._tasks
            }

            for future in as_completed(future_to_task):
                if self.is_cancelled():
                    self._log_cancelled()
                    for f in future_to_task:
                        f.cancel()
                    break

                task = future_to_task[future]
                completed += 1

                try:
                    clip_id, detections, person_count, error_msg = (
                        future.result()
                    )

                    if error_msg and error_msg != "Cancelled":
                        self._log_error(error_msg, clip_id)
                    elif detections is not None:
                        self.objects_ready.emit(
                            clip_id, detections, person_count
                        )
                except Exception as e:
                    self._log_error(str(e), task.clip_id)

                self.progress.emit(completed, total)

        self.detection_completed.emit()
        self._log_complete()

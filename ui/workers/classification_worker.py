"""Background worker for frame classification using MobileNet.

Runs image classification on multiple clips in a background thread,
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
class ClassificationTask:
    """Immutable task data for thread pool execution."""

    clip_id: str
    thumbnail_path: Path


class ClassificationWorker(CancellableWorker):
    """Background worker for frame classification using MobileNet.

    Uses ThreadPoolExecutor for parallel processing. Default parallelism is 1
    because the MobileNet model singleton is not thread-safe for inference.

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
        parent=None,
    ):
        super().__init__(parent)
        self._top_k = top_k
        self._threshold = threshold
        self._parallelism = min(max(1, parallelism), 4)
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

    def _process_task(
        self, task: ClassificationTask
    ) -> tuple[str, Optional[list], Optional[str]]:
        """Process a single task (runs in thread pool).

        Returns:
            Tuple of (clip_id, labels, error_message)
        """
        if self.is_cancelled():
            return task.clip_id, None, "Cancelled"

        try:
            from core.analysis.classification import classify_frame

            results = classify_frame(
                task.thumbnail_path,
                top_k=self._top_k,
                threshold=self._threshold,
            )
            return task.clip_id, results, None
        except Exception as e:
            return task.clip_id, None, str(e)

    def run(self):
        """Execute frame classification on all clips."""
        self._log_start()

        total = len(self._tasks)
        if total == 0:
            logger.info("No clips to process for classification")
            self.classification_completed.emit()
            self._log_complete()
            return

        logger.info(
            f"Starting classification: {total} clips, "
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
                    clip_id, labels, error_msg = future.result()

                    if error_msg and error_msg != "Cancelled":
                        self._log_error(error_msg, clip_id)
                    elif labels is not None:
                        self.labels_ready.emit(clip_id, labels)
                except Exception as e:
                    self._log_error(str(e), task.clip_id)

                self.progress.emit(completed, total)

        self.classification_completed.emit()
        self._log_complete()

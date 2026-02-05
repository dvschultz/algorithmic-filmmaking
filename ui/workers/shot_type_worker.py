"""Background worker for shot type classification.

Runs shot type classification on multiple clips in a background thread,
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
class ShotTypeTask:
    """Immutable task data for thread pool execution."""

    clip_id: str
    thumbnail_path: Path
    source_path: Optional[Path]
    start_frame: int
    end_frame: int
    fps: Optional[float]


class ShotTypeWorker(CancellableWorker):
    """Background worker for shot type classification using CLIP or VideoMAE.

    Supports tiered processing:
    - CPU: CLIP zero-shot classification from thumbnails (free, local)
    - Cloud: VideoMAE model on Replicate for video-based classification (paid)

    Uses ThreadPoolExecutor for parallel processing. Default parallelism is 1
    because the CLIP model singleton is not thread-safe for inference.

    Signals:
        progress: Emitted with (current, total) during processing
        shot_type_ready: Emitted with (clip_id, shot_type, confidence)
        analysis_completed: Emitted when all clips are processed
        error: Emitted with error message string on failure (inherited)
    """

    progress = Signal(int, int)  # current, total
    shot_type_ready = Signal(str, str, float)  # clip_id, shot_type, confidence
    analysis_completed = Signal()

    def __init__(
        self,
        clips: list,
        sources_by_id: dict,
        parallelism: int = 1,
        skip_existing: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._parallelism = min(max(1, parallelism), 4)
        self._tasks = self._build_tasks(clips, sources_by_id, skip_existing)

    def _build_tasks(
        self, clips: list, sources_by_id: dict, skip_existing: bool
    ) -> list[ShotTypeTask]:
        """Build immutable task list from clips."""
        tasks = []
        for clip in clips:
            if skip_existing and clip.shot_type is not None:
                continue
            if not clip.thumbnail_path or not clip.thumbnail_path.exists():
                logger.warning(f"Skipping clip {clip.id}: thumbnail not found")
                continue

            source = sources_by_id.get(clip.source_id)
            source_path = source.file_path if source else None
            fps = source.fps if source else None

            tasks.append(
                ShotTypeTask(
                    clip_id=clip.id,
                    thumbnail_path=clip.thumbnail_path,
                    source_path=source_path,
                    start_frame=clip.start_frame,
                    end_frame=clip.end_frame,
                    fps=fps,
                )
            )
        return tasks

    def _process_task(
        self, task: ShotTypeTask
    ) -> tuple[str, Optional[str], Optional[float], Optional[str]]:
        """Process a single task (runs in thread pool).

        Returns:
            Tuple of (clip_id, shot_type, confidence, error_message)
        """
        if self.is_cancelled():
            return task.clip_id, None, None, "Cancelled"

        try:
            from core.analysis.shots import classify_shot_type_tiered

            shot_type, confidence = classify_shot_type_tiered(
                image_path=task.thumbnail_path,
                source_path=task.source_path,
                start_frame=task.start_frame,
                end_frame=task.end_frame,
                fps=task.fps,
            )
            return task.clip_id, shot_type, confidence, None
        except Exception as e:
            return task.clip_id, None, None, str(e)

    def run(self):
        """Execute shot type classification on all clips."""
        self._log_start()

        total = len(self._tasks)
        if total == 0:
            logger.info("No clips to process for shot type classification")
            self.analysis_completed.emit()
            self._log_complete()
            return

        logger.info(
            f"Starting shot type classification: {total} clips, "
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
                    clip_id, shot_type, confidence, error_msg = future.result()

                    if error_msg and error_msg != "Cancelled":
                        self._log_error(error_msg, clip_id)
                    elif shot_type is not None:
                        self.shot_type_ready.emit(clip_id, shot_type, confidence)
                except Exception as e:
                    self._log_error(str(e), task.clip_id)

                self.progress.emit(completed, total)

        self.analysis_completed.emit()
        self._log_complete()

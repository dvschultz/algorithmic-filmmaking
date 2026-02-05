"""Background worker for color extraction from thumbnails.

Runs dominant color extraction on multiple clips in a background thread,
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
class ColorAnalysisTask:
    """Immutable task data for thread pool execution."""

    clip_id: str
    thumbnail_path: Path


class ColorAnalysisWorker(CancellableWorker):
    """Background worker for color extraction from thumbnails.

    Uses ThreadPoolExecutor for parallel processing. All signal emissions
    happen on the QThread, not from pool worker threads.

    Signals:
        progress: Emitted with (current, total) during processing
        color_ready: Emitted with (clip_id, colors) when a clip finishes
        analysis_completed: Emitted when all clips are processed
        error: Emitted with error message string on failure (inherited)
    """

    progress = Signal(int, int)  # current, total
    color_ready = Signal(str, list)  # clip_id, colors (list of RGB tuples)
    analysis_completed = Signal()

    def __init__(
        self,
        clips: list,
        parallelism: int = 4,
        skip_existing: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._parallelism = min(max(1, parallelism), 8)
        self._tasks = self._build_tasks(clips, skip_existing)

    def _build_tasks(
        self, clips: list, skip_existing: bool
    ) -> list[ColorAnalysisTask]:
        """Build immutable task list from clips."""
        tasks = []
        for clip in clips:
            if skip_existing and clip.dominant_colors is not None:
                continue
            if not clip.thumbnail_path or not clip.thumbnail_path.exists():
                logger.warning(f"Skipping clip {clip.id}: thumbnail not found")
                continue
            tasks.append(
                ColorAnalysisTask(
                    clip_id=clip.id,
                    thumbnail_path=clip.thumbnail_path,
                )
            )
        return tasks

    def _process_task(
        self, task: ColorAnalysisTask
    ) -> tuple[str, Optional[list], Optional[str]]:
        """Process a single task (runs in thread pool).

        Returns:
            Tuple of (clip_id, colors, error_message)
        """
        if self.is_cancelled():
            return task.clip_id, None, "Cancelled"

        try:
            from core.analysis.color import extract_dominant_colors

            colors = extract_dominant_colors(task.thumbnail_path)
            return task.clip_id, colors, None
        except Exception as e:
            return task.clip_id, None, str(e)

    def run(self):
        """Execute color extraction on all clips."""
        self._log_start()

        total = len(self._tasks)
        if total == 0:
            logger.info("No clips to process for color analysis")
            self.analysis_completed.emit()
            self._log_complete()
            return

        logger.info(
            f"Starting color analysis: {total} clips, "
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
                    clip_id, colors, error_msg = future.result()

                    if error_msg and error_msg != "Cancelled":
                        self._log_error(error_msg, clip_id)
                    elif colors is not None:
                        self.color_ready.emit(clip_id, colors)
                except Exception as e:
                    self._log_error(str(e), task.clip_id)

                self.progress.emit(completed, total)

        self.analysis_completed.emit()
        self._log_complete()

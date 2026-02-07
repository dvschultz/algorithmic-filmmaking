"""Background worker for generating video descriptions.

Runs VLM description on multiple clips in a background thread,
using ThreadPoolExecutor for parallelism with retry logic for cloud APIs.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal

from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)

# Retry configuration for cloud API rate limits
_MAX_RETRIES = 3
_RETRY_DELAYS = [2, 5, 10]  # seconds


@dataclass(frozen=True)
class DescriptionTask:
    """Immutable task data for thread pool execution."""

    clip_id: str
    thumbnail_path: Path
    source_path: Optional[Path]
    start_frame: int
    end_frame: int
    fps: Optional[float]


class DescriptionWorker(CancellableWorker):
    """Background worker for generating video descriptions.

    Uses ThreadPoolExecutor for parallel processing. Default parallelism is 3
    since cloud VLM calls are I/O-bound.

    Includes retry logic with exponential backoff for rate-limit (429) errors.

    Supports both Clip and Frame inputs via AnalysisTarget.

    Signals:
        progress: Emitted with (current, total) during processing
        description_ready: Emitted with (clip_id, description, model_name)
        error: Emitted with (clip_id, error_message)
        description_completed: Emitted when all clips are processed
    """

    progress = Signal(int, int)  # current, total
    description_ready = Signal(str, str, str)  # clip_id, description, model_name
    error = Signal(str, str)  # clip_id, error_message (shadows base class)
    description_completed = Signal()

    def __init__(
        self,
        clips: list,
        tier: Optional[str] = None,
        prompt: Optional[str] = None,
        sources: Optional[dict] = None,
        parallelism: int = 3,
        skip_existing: bool = True,
        analysis_targets: Optional[list] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._tier = tier
        self._prompt = prompt or (
            "Describe this video frame in 3 sentences or less. "
            "Focus on the main subjects, action, and setting."
        )
        self._parallelism = min(max(1, parallelism), 5)
        self.error_count = 0
        self.success_count = 0
        self.last_error = None
        if analysis_targets:
            self._tasks = self._build_tasks_from_targets(
                analysis_targets, skip_existing
            )
        else:
            self._tasks = self._build_tasks(clips, sources or {}, skip_existing)

    def _build_tasks(
        self, clips: list, sources: dict, skip_existing: bool
    ) -> list[DescriptionTask]:
        """Build immutable task list from clips."""
        tasks = []
        for clip in clips:
            if skip_existing and clip.description is not None:
                continue
            if not clip.thumbnail_path or not clip.thumbnail_path.exists():
                logger.warning(f"Skipping clip {clip.id}: thumbnail not found")
                continue

            source = sources.get(clip.source_id)
            tasks.append(
                DescriptionTask(
                    clip_id=clip.id,
                    thumbnail_path=clip.thumbnail_path,
                    source_path=source.file_path if source else None,
                    start_frame=clip.start_frame,
                    end_frame=clip.end_frame,
                    fps=source.fps if source else None,
                )
            )
        return tasks

    def _build_tasks_from_targets(
        self, targets: list, skip_existing: bool
    ) -> list[DescriptionTask]:
        """Build immutable task list from AnalysisTarget objects."""
        tasks = []
        for target in targets:
            if skip_existing and target.description is not None:
                continue
            image_path = target.image_path
            if not image_path or not image_path.exists():
                logger.warning(
                    f"Skipping target {target.id}: image not found"
                )
                continue
            tasks.append(
                DescriptionTask(
                    clip_id=target.id,
                    thumbnail_path=image_path,
                    source_path=target.video_path,
                    start_frame=target.start_frame or 0,
                    end_frame=target.end_frame or 0,
                    fps=target.fps,
                )
            )
        return tasks

    def _process_task(
        self, task: DescriptionTask
    ) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
        """Process a single task with retry logic (runs in thread pool).

        Returns:
            Tuple of (clip_id, description, model_name, error_message)
        """
        if self.is_cancelled():
            return task.clip_id, None, None, "Cancelled"

        from core.analysis.description import describe_frame

        last_error = None
        for attempt in range(_MAX_RETRIES + 1):
            if self.is_cancelled():
                return task.clip_id, None, None, "Cancelled"

            try:
                description, model = describe_frame(
                    task.thumbnail_path,
                    tier=self._tier,
                    prompt=self._prompt,
                    source_path=task.source_path,
                    start_frame=task.start_frame,
                    end_frame=task.end_frame,
                    fps=task.fps,
                )

                if description and not description.startswith("Error"):
                    return task.clip_id, description, model, None
                else:
                    return task.clip_id, None, None, description
            except Exception as e:
                last_error = str(e)
                # Retry on rate-limit errors
                is_rate_limit = "429" in last_error or "rate" in last_error.lower()
                if is_rate_limit and attempt < _MAX_RETRIES:
                    delay = _RETRY_DELAYS[attempt]
                    logger.warning(
                        f"Rate limit for {task.clip_id}, "
                        f"retry {attempt + 1}/{_MAX_RETRIES} in {delay}s"
                    )
                    time.sleep(delay)
                    continue
                break

        return task.clip_id, None, None, last_error

    def run(self):
        """Execute description generation on all clips."""
        self._log_start()

        total = len(self._tasks)
        if total == 0:
            logger.info("No clips to process for descriptions")
            self.description_completed.emit()
            self._log_complete()
            return

        logger.info(
            f"Starting description generation: {total} clips, "
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
                    clip_id, description, model, error_msg = future.result()

                    if error_msg and error_msg != "Cancelled":
                        self._log_error(error_msg, clip_id)
                        self.error_count += 1
                        self.last_error = error_msg
                        self.error.emit(clip_id, error_msg)
                    elif description:
                        self.description_ready.emit(clip_id, description, model)
                        self.success_count += 1
                except Exception as e:
                    error_msg = str(e)
                    self._log_error(error_msg, task.clip_id)
                    self.error_count += 1
                    self.last_error = error_msg
                    self.error.emit(task.clip_id, error_msg)

                self.progress.emit(completed, total)

        logger.info(
            f"DescriptionWorker.run() completed: "
            f"{self.success_count} success, {self.error_count} errors"
        )
        self.description_completed.emit()
        self._log_complete()

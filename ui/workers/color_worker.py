"""Background worker for color extraction from video clips.

Runs dominant color extraction on multiple clips in a background thread,
using ThreadPoolExecutor for parallelism. Each clip is analyzed by sampling
multiple frames from the source video.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal

from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)


def _summarize_errors(errors: list[tuple[str, str]]) -> str:
    """Return a compact user-facing summary for a batch failure."""
    preview = "\n".join(f"- {clip_id}: {message}" for clip_id, message in errors[:3])
    if len(errors) == 1:
        return errors[0][1]

    remaining = len(errors) - 3
    summary = f"Color extraction failed for {len(errors)} clips:\n\n{preview}"
    if remaining > 0:
        summary += f"\n\n... and {remaining} more"
    return summary


@dataclass(frozen=True)
class ColorAnalysisTask:
    """Immutable task data for thread pool execution."""

    clip_id: str
    video_path: Path
    start_frame: int
    end_frame: int
    image_path: Optional[Path] = None  # Single-image fallback for frame targets


class ColorAnalysisWorker(CancellableWorker):
    """Background worker for color extraction from video clips.

    Uses ThreadPoolExecutor for parallel processing. All signal emissions
    happen on the QThread, not from pool worker threads.

    Supports both Clip and Frame inputs via AnalysisTarget.

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
        analysis_targets: Optional[list] = None,
        sources_by_id: Optional[dict] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._parallelism = min(max(1, parallelism), 8)
        if analysis_targets:
            self._tasks = self._build_tasks_from_targets(
                analysis_targets, skip_existing
            )
        else:
            self._tasks = self._build_tasks(clips, skip_existing, sources_by_id or {})

    def _build_tasks(
        self, clips: list, skip_existing: bool, sources_by_id: dict
    ) -> list[ColorAnalysisTask]:
        """Build immutable task list from clips."""
        tasks = []
        for clip in clips:
            if skip_existing and clip.dominant_colors is not None:
                continue
            source = sources_by_id.get(clip.source_id)
            if not source or not source.file_path.exists():
                logger.warning(f"Skipping clip {clip.id}: source video not found")
                continue
            tasks.append(
                ColorAnalysisTask(
                    clip_id=clip.id,
                    video_path=source.file_path,
                    start_frame=clip.start_frame,
                    end_frame=clip.end_frame,
                )
            )
        return tasks

    def _build_tasks_from_targets(
        self, targets: list, skip_existing: bool
    ) -> list[ColorAnalysisTask]:
        """Build immutable task list from AnalysisTarget objects."""
        tasks = []
        for target in targets:
            if skip_existing and target.dominant_colors is not None:
                continue
            if target.target_type == "clip" and target.video_path and target.start_frame is not None and target.end_frame is not None:
                if not target.video_path.exists():
                    logger.warning(f"Skipping target {target.id}: video not found")
                    continue
                tasks.append(
                    ColorAnalysisTask(
                        clip_id=target.id,
                        video_path=target.video_path,
                        start_frame=target.start_frame,
                        end_frame=target.end_frame,
                    )
                )
            else:
                # Frame target — single image fallback
                image_path = target.image_path
                if not image_path or not image_path.exists():
                    logger.warning(f"Skipping target {target.id}: image not found")
                    continue
                tasks.append(
                    ColorAnalysisTask(
                        clip_id=target.id,
                        video_path=Path(),  # unused in image_path mode
                        start_frame=0,
                        end_frame=0,
                        image_path=image_path,
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

            colors = extract_dominant_colors(
                video_path=task.video_path,
                start_frame=task.start_frame,
                end_frame=task.end_frame,
                image_path=task.image_path,
            )
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
        errors: list[tuple[str, str]] = []

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
                        errors.append((clip_id, error_msg))
                    elif colors is not None:
                        self.color_ready.emit(clip_id, colors)
                except Exception as e:
                    self._log_error(str(e), task.clip_id)
                    errors.append((task.clip_id, str(e)))

                self.progress.emit(completed, total)

        if errors:
            self.error.emit(_summarize_errors(errors))
        self.analysis_completed.emit()
        self._log_complete()

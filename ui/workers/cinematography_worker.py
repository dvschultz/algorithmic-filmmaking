"""Background worker for rich cinematography analysis.

Runs cinematography analysis on multiple clips in a background thread,
emitting progress signals to keep the UI responsive.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal

from core.analysis.cinematography import analyze_cinematography
from models.cinematography import CinematographyAnalysis
from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClipAnalysisTask:
    """Immutable task data for thread pool execution.

    Using frozen dataclass ensures thread safety - no shared mutable state
    is passed into the thread pool.
    """

    clip_id: str
    thumbnail_path: Path
    source_path: Optional[Path]
    start_frame: int
    end_frame: int
    fps: float


class CinematographyWorker(CancellableWorker):
    """Analyze cinematography for multiple clips in background.

    Uses ThreadPoolExecutor for parallel VLM requests while keeping
    the UI responsive. All signal emissions happen on the QThread,
    not from pool worker threads.

    Supports both Clip and Frame inputs via AnalysisTarget.

    Signals:
        progress: Emitted with (current, total, clip_id) during processing
        clip_completed: Emitted with (clip_id, CinematographyAnalysis) when done
        finished: Emitted with {clip_id: CinematographyAnalysis} on completion
        error: Emitted with error message string on failure (inherited)
    """

    progress = Signal(int, int, str)  # current, total, clip_id
    clip_completed = Signal(str, object)  # clip_id, CinematographyAnalysis
    finished = Signal(dict)  # {clip_id: CinematographyAnalysis}

    def __init__(
        self,
        clips: list,
        sources_by_id: dict,
        mode: Optional[str] = None,
        model: Optional[str] = None,
        parallelism: int = 2,
        skip_existing: bool = True,
        analysis_targets: Optional[list] = None,
        parent=None,
    ):
        """Initialize the cinematography analysis worker.

        Args:
            clips: List of Clip objects to process
            sources_by_id: Dict mapping source_id to Source objects
            mode: Input mode ("frame" or "video"). None uses settings default.
            model: VLM model to use (default: from settings)
            parallelism: Number of concurrent VLM requests (1-5, default: 2)
            skip_existing: Skip clips that already have cinematography data
            analysis_targets: Optional list of AnalysisTarget objects (alternative to clips)
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self._mode = mode
        self._model = model
        self._parallelism = min(max(1, parallelism), 5)

        # Build immutable task list upfront - no mutable state in thread pool
        if analysis_targets:
            self._tasks = self._build_tasks_from_targets(
                analysis_targets, skip_existing
            )
        else:
            self._tasks = self._build_tasks(clips, sources_by_id, skip_existing)

    def _build_tasks(
        self, clips: list, sources_by_id: dict, skip_existing: bool
    ) -> list[ClipAnalysisTask]:
        """Build immutable task list from clips.

        Filters and validates clips, creating frozen dataclass instances
        that can safely be passed to the thread pool.
        """
        tasks = []

        for clip in clips:
            # Skip if already analyzed
            if skip_existing and clip.cinematography is not None:
                continue

            # Validate source exists
            source = sources_by_id.get(clip.source_id)
            if not source:
                logger.warning(f"Skipping clip {clip.id}: source not found")
                continue

            # Validate thumbnail exists
            if not clip.thumbnail_path or not clip.thumbnail_path.exists():
                logger.warning(f"Skipping clip {clip.id}: thumbnail not found")
                continue

            # Create immutable task
            tasks.append(
                ClipAnalysisTask(
                    clip_id=clip.id,
                    thumbnail_path=clip.thumbnail_path,
                    source_path=source.file_path if source.file_path.exists() else None,
                    start_frame=clip.start_frame,
                    end_frame=clip.end_frame,
                    fps=source.fps,
                )
            )

        return tasks

    def _build_tasks_from_targets(
        self, targets: list, skip_existing: bool
    ) -> list[ClipAnalysisTask]:
        """Build immutable task list from AnalysisTarget objects."""
        tasks = []

        for target in targets:
            if skip_existing and target.cinematography is not None:
                continue

            image_path = target.image_path
            if not image_path or not image_path.exists():
                logger.warning(
                    f"Skipping target {target.id}: image not found"
                )
                continue

            # For frame targets, use "frame" mode since there's no video
            source_path = None
            if target.video_path and target.video_path.exists():
                source_path = target.video_path

            tasks.append(
                ClipAnalysisTask(
                    clip_id=target.id,
                    thumbnail_path=image_path,
                    source_path=source_path,
                    start_frame=target.start_frame or 0,
                    end_frame=target.end_frame or 0,
                    fps=target.fps or 30.0,
                )
            )

        return tasks

    def _analyze_task(
        self, task: ClipAnalysisTask
    ) -> tuple[str, Optional[CinematographyAnalysis], Optional[str]]:
        """Analyze a single task (runs in thread pool).

        Args:
            task: Immutable task data

        Returns:
            Tuple of (clip_id, analysis_result, error_message)
        """
        # Check cancellation before expensive work
        if self.is_cancelled():
            return task.clip_id, None, "Cancelled"

        try:
            analysis = analyze_cinematography(
                thumbnail_path=task.thumbnail_path,
                source_path=task.source_path,
                start_frame=task.start_frame,
                end_frame=task.end_frame,
                fps=task.fps,
                mode=self._mode,
                model=self._model,
            )
            return task.clip_id, analysis, None
        except Exception as e:
            return task.clip_id, None, str(e)

    def run(self):
        """Execute cinematography analysis on all clips.

        Uses ThreadPoolExecutor for parallelism. All signal emissions
        happen here on the QThread, not from pool worker threads.
        """
        self._log_start()

        total = len(self._tasks)
        if total == 0:
            logger.info("No clips to process for cinematography analysis")
            self.finished.emit({})
            self._log_complete()
            return

        logger.info(
            f"Starting cinematography analysis: {total} clips, "
            f"parallelism={self._parallelism}"
        )

        results: dict[str, CinematographyAnalysis] = {}
        completed = 0

        with ThreadPoolExecutor(max_workers=self._parallelism) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._analyze_task, task): task
                for task in self._tasks
            }

            # Process results as they complete (on QThread, not pool threads)
            for future in as_completed(future_to_task):
                if self.is_cancelled():
                    self._log_cancelled()
                    # Cancel pending futures (won't stop running tasks)
                    for f in future_to_task:
                        f.cancel()
                    break

                task = future_to_task[future]
                completed += 1

                try:
                    clip_id, analysis, error_msg = future.result()

                    if error_msg and error_msg != "Cancelled":
                        self._log_error(error_msg, clip_id)
                        self.error.emit(f"Error analyzing {clip_id}: {error_msg}")
                    elif analysis:
                        results[clip_id] = analysis
                        # Emit on QThread (this is safe)
                        self.clip_completed.emit(clip_id, analysis)
                        logger.debug(
                            f"Cinematography: {clip_id} -> "
                            f"{analysis.shot_size}, {analysis.camera_angle}"
                        )

                except Exception as e:
                    self._log_error(str(e), task.clip_id)
                    self.error.emit(f"Error analyzing {task.clip_id}: {e}")

                self.progress.emit(completed, total, task.clip_id)

        if not self.is_cancelled():
            logger.info(
                f"Cinematography analysis complete: "
                f"{len(results)}/{total} clips processed"
            )
            self.finished.emit(results)
            self._log_complete()

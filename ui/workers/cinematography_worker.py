"""Background worker for rich cinematography analysis.

Runs cinematography analysis on multiple clips in a background thread,
emitting progress signals to keep the UI responsive.
"""

import logging
from typing import Optional

from PySide6.QtCore import Signal

from core.operations.cinematography import (
    CinematographyTask as ClipAnalysisTask,
    CinematographyOutcome,
    compute_cinematography,
    resolve_options,
    run_cinematography,
)
from models.cinematography import CinematographyAnalysis
from ui.workers.base import CancellableWorker, summarize_clip_errors

logger = logging.getLogger(__name__)


def _summarize_errors(errors: list[tuple[str, str]]) -> str:
    """Return a compact user-facing summary for a batch failure."""
    return summarize_clip_errors(errors, operation_label="Cinematography analysis")


class CinematographyWorker(CancellableWorker):
    """Analyze cinematography for multiple clips in background.

    Uses ThreadPoolExecutor for parallel VLM requests while keeping
    the UI responsive. All signal emissions happen on the QThread,
    not from pool worker threads.

    Supports both Clip and Frame inputs via AnalysisTarget.

    Signals:
        progress: Emitted with (current, total, clip_id) during processing
        clip_completed: Emitted with (clip_id, CinematographyAnalysis) when done
        analysis_completed: Emitted with {clip_id: CinematographyAnalysis} on completion
        error: Emitted with error message string on failure (inherited)
    """

    progress = Signal(int, int, str)  # current, total, clip_id
    clip_completed = Signal(str, object)  # clip_id, CinematographyAnalysis
    analysis_completed = Signal(dict)  # {clip_id: CinematographyAnalysis}

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
    ) -> None:
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
        self.options = resolve_options(mode, model, parallelism)
        self._parallelism = (
            1 if self.options.tier == "local" else min(max(1, parallelism), 5)
        )
        self.result: tuple[CinematographyOutcome, ...] = ()

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
                logger.warning(f"Skipping target {target.id}: image not found")
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
                    target_type=target.target_type,
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
        outcome = compute_cinematography(task, self.options, self._cancel_event)
        error = (
            "Cancelled"
            if outcome.status == "unprocessed"
            else outcome.message or outcome.code
        )
        return task.clip_id, outcome.analysis, error

    @property
    def tasks(self) -> tuple[ClipAnalysisTask, ...]:
        return tuple(self._tasks)

    def run(self) -> None:
        """Run shared computation and settle completion even after cancellation."""
        self._log_start()
        results: dict[str, CinematographyAnalysis] = {}
        errors: list[tuple[str, str]] = []
        try:

            def deliver(outcome: CinematographyOutcome) -> None:
                if outcome.status == "succeeded":
                    analysis = outcome.analysis
                    if analysis is not None:
                        results[outcome.clip_id] = analysis
                        self.clip_completed.emit(outcome.clip_id, outcome.analysis)
                elif outcome.status == "failed":
                    errors.append(
                        (
                            outcome.clip_id,
                            outcome.message or outcome.code or "Analysis failed",
                        )
                    )

            self.result = run_cinematography(
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
            self.analysis_completed.emit(results)
            self._log_complete()

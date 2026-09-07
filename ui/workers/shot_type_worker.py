"""Background worker for shot type classification.

Runs the shared detached operation on a background thread.
"""

import logging
from typing import Optional

from PySide6.QtCore import Signal

from ui.workers.base import CancellableWorker

from core.operations.shots import (
    ShotTypeTask,
    ShotTypeOptions,
    ShotTypeOutcome,
    run_shot_types,
)

logger = logging.getLogger(__name__)


class ShotTypeWorker(CancellableWorker):
    """Background worker for shot type classification using CLIP or VideoMAE.

    Supports tiered processing:
    - CPU: CLIP zero-shot classification from thumbnails (free, local)
    - Cloud: VideoMAE model on Replicate for video-based classification (paid)

    Supports both Clip and Frame inputs via AnalysisTarget.

    Inference is serialized across jobs because the local model is shared.
    The parallelism argument is retained for existing callers.

    Signals:
        progress: Emitted with (current, total) during processing
        shot_type_ready: Emitted with (clip_id, shot_type, confidence)
        analysis_completed: Emitted when all clips are processed
        error: Emitted with error message string on failure (inherited)
    """

    progress = Signal(int, int)  # current, total
    shot_type_ready = Signal(str, str, float)  # clip_id, shot_type, confidence
    analysis_completed = Signal()
    outcome_ready = Signal(object)

    @staticmethod
    def _summarize_errors(errors: list[tuple[str, str]]) -> str:
        """Create a readable summary for one or more clip failures."""
        preview = "\n".join(
            f"- {clip_id}: {message}" for clip_id, message in errors[:3]
        )
        if len(errors) == 1:
            return f"Shot type classification failed:\n\n{preview}"

        remaining = len(errors) - 3
        extra = f"\n- ...and {remaining} more clip(s)" if remaining > 0 else ""
        return (
            f"Shot type classification failed for {len(errors)} clips:\n\n"
            f"{preview}{extra}"
        )

    def __init__(
        self,
        clips: list,
        sources_by_id: dict,
        parallelism: int = 1,
        skip_existing: bool = True,
        analysis_targets: Optional[list] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._parallelism = 1
        self.options = ShotTypeOptions.from_settings()
        self.result: tuple[ShotTypeOutcome, ...] = ()
        if analysis_targets:
            self._tasks = self._build_tasks_from_targets(
                analysis_targets, skip_existing
            )
        else:
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

    def _build_tasks_from_targets(
        self, targets: list, skip_existing: bool
    ) -> list[ShotTypeTask]:
        """Build immutable task list from AnalysisTarget objects."""
        tasks = []
        for target in targets:
            if skip_existing and target.shot_type is not None:
                continue
            image_path = target.image_path
            if not image_path or not image_path.exists():
                logger.warning(f"Skipping target {target.id}: image not found")
                continue
            tasks.append(
                ShotTypeTask(
                    clip_id=target.id,
                    thumbnail_path=image_path,
                    source_path=target.video_path,
                    start_frame=target.start_frame or 0,
                    end_frame=target.end_frame or 0,
                    fps=target.fps,
                    target_type=target.target_type,
                )
            )
        return tasks

    @property
    def tasks(self) -> tuple[ShotTypeTask, ...]:
        return tuple(self._tasks)

    def run(self) -> None:
        """Compute detached results and always report batch termination."""
        self._log_start()
        errors: list[tuple[str, str]] = []

        def deliver(outcome: ShotTypeOutcome) -> None:
            if outcome.status == "succeeded":
                self.outcome_ready.emit(outcome)
                self.shot_type_ready.emit(
                    outcome.clip_id, outcome.shot_type, outcome.confidence
                )
            elif outcome.status == "failed":
                errors.append(
                    (
                        outcome.clip_id,
                        outcome.message or outcome.code or "Analysis failed",
                    )
                )

        try:
            if self.tasks:
                self.progress.emit(0, len(self.tasks))
            self.result = run_shot_types(
                self.tasks,
                self.options,
                cancel_event=self._cancel_event,
                on_outcome=deliver,
                progress=self.progress.emit,
            )
            if errors and not self.is_cancelled():
                self.error.emit(self._summarize_errors(errors))
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.analysis_completed.emit()
            self._log_complete()

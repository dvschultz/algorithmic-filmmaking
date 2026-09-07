"""Background worker for generating video descriptions.

Runs VLM description on multiple clips in a background thread,
using ThreadPoolExecutor for parallelism with retry logic for cloud APIs.
"""

import logging
from typing import Optional

from PySide6.QtCore import Signal

from core.settings import load_settings
from ui.workers.base import CancellableWorker
from core.operations.description import (
    DEFAULT_PROMPT,
    DescriptionTask,
    DescriptionOptions,
    DescriptionOutcome,
    compute_description,
    run_description,
)

logger = logging.getLogger(__name__)


class DescriptionWorker(CancellableWorker):
    """Background worker for generating video descriptions.

    Uses ThreadPoolExecutor for parallel processing. Cloud requests can run
    concurrently, but local VLM inference is forced to serial execution
    because the underlying runtimes are not thread-safe.

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
        self._tier = self._resolve_tier(tier)
        self._prompt = prompt or DEFAULT_PROMPT
        requested_parallelism = min(max(1, parallelism), 5)
        # Local MLX/Moondream inference shares model state and can crash native
        # backends if multiple descriptions run at once.
        self._parallelism = 1 if self._tier == "local" else requested_parallelism
        self.result: tuple[DescriptionOutcome, ...] = ()
        self.error_count = 0
        self.success_count = 0
        self.last_error: Optional[str] = None
        if analysis_targets:
            self._tasks = self._build_tasks_from_targets(
                analysis_targets, skip_existing
            )
        else:
            self._tasks = self._build_tasks(clips, sources or {}, skip_existing)

    @staticmethod
    def _resolve_tier(tier: Optional[str]) -> str:
        """Normalize the effective description tier."""
        resolved = tier or load_settings().description_model_tier
        if resolved in ("cpu", "gpu"):
            return "local"
        return resolved

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
                logger.warning(f"Skipping target {target.id}: image not found")
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
        """Compatibility wrapper for callers testing one detached task."""
        outcome = compute_description(
            task, DescriptionOptions(self._tier, self._prompt), self._cancel_event
        )
        error = "Cancelled" if outcome.status == "unprocessed" else outcome.message
        return outcome.clip_id, outcome.description, outcome.model, error

    def run(self) -> None:
        """Execute description generation on all clips."""
        self._log_start()
        try:
            self._run_descriptions()
        finally:
            self.description_completed.emit()
            self._log_complete()

    def _run_descriptions(self) -> None:
        if self.is_cancelled():
            self._log_cancelled()
            return

        total = len(self._tasks)
        if total == 0:
            logger.info("No clips to process for descriptions")
            return

        logger.info(
            f"Starting description generation: {total} clips, "
            f"parallelism={self._parallelism}"
        )

        # Pre-load local VLM model so user sees download status
        if self._tier == "local":
            try:
                from core.analysis.description import is_model_loaded, _load_local_model

                if not is_model_loaded():
                    self.progress.emit(0, total)
                    _load_local_model()
            except Exception as e:
                if self.is_cancelled():
                    self._log_cancelled()
                    return
                message = f"Failed to load local VLM: {e}"
                self.last_error = message
                for task in self._tasks:
                    self.error_count += 1
                    self._log_error(message, task.clip_id)
                    self.error.emit(task.clip_id, message)
                return

        if self.is_cancelled():
            self._log_cancelled()
            return

        self.result = run_description(
            tuple(self._tasks),
            DescriptionOptions(self._tier, self._prompt, self._parallelism),
            cancel_event=self._cancel_event,
            on_outcome=self._on_outcome,
            progress=self.progress.emit,
        )

    def _on_outcome(self, outcome: DescriptionOutcome) -> None:
        if outcome.status == "failed":
            message = outcome.message or outcome.code or "Description failed"
            self._log_error(message, outcome.clip_id)
            self.error_count += 1
            self.last_error = message
            self.error.emit(outcome.clip_id, message)
        elif outcome.status == "succeeded":
            self.description_ready.emit(
                outcome.clip_id, outcome.description, outcome.model
            )
            self.success_count += 1

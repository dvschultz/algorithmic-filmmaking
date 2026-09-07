"""Background worker for custom visual query evaluation.

Runs a yes/no VLM query across multiple clips in a background thread.
Cloud APIs use ThreadPoolExecutor for parallelism and retry logic; local
VLM inference runs serially in the worker thread because MLX stream state is
thread-local.
"""

import logging
from typing import Optional

from PySide6.QtCore import Signal

from core.settings import load_settings
from ui.workers.base import CancellableWorker
from core.operations.custom_query import (
    CustomQueryTask,
    CustomQueryOutcome,
    compute_custom_query,
    resolve_options,
    run_custom_query,
)

logger = logging.getLogger(__name__)


class CustomQueryWorker(CancellableWorker):
    """Background worker for evaluating custom visual queries.

    Cloud tier uses ThreadPoolExecutor for parallel processing. Local tier
    runs serially inside this worker thread because MLX stream state is
    thread-local and local VLM model state is not thread-safe.

    Includes retry logic with exponential backoff for rate-limit (429) errors.

    Signals:
        progress: Emitted with (current, total) during processing
        query_result_ready: Emitted with (clip_id, query, match, confidence, model)
        analysis_completed: Emitted when all clips are processed
        error: Emitted with error message string on failure (inherited)
    """

    progress = Signal(int, int)  # current, total
    query_result_ready = Signal(
        str, str, bool, float, str
    )  # clip_id, query, match, confidence, model
    analysis_completed = Signal()

    @staticmethod
    def _summarize_errors(errors: list[tuple[str, str]]) -> str:
        """Create a readable summary for one or more clip failures."""
        preview = "\n".join(
            f"- {clip_id}: {message}" for clip_id, message in errors[:3]
        )
        if len(errors) == 1:
            return f"Custom query failed:\n\n{preview}"

        remaining = len(errors) - 3
        extra = f"\n- ...and {remaining} more clip(s)" if remaining > 0 else ""
        return f"Custom query failed for {len(errors)} clips:\n\n{preview}{extra}"

    def __init__(
        self,
        clips: list,
        query: str,
        sources_by_id: dict,
        tier: Optional[str] = None,
        parallelism: int = 3,
        skip_existing: bool = False,
        analysis_targets: Optional[list] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        query = query.strip()
        self._query = query
        self._tier = self._resolve_tier(tier)
        requested_parallelism = min(max(1, parallelism), 5)
        self._parallelism = 1 if self._tier == "local" else requested_parallelism
        self.options = resolve_options(self._tier, self._parallelism)
        self.result: tuple[CustomQueryOutcome, ...] = ()
        if analysis_targets:
            self._tasks = self._build_tasks_from_targets(
                analysis_targets, query, skip_existing
            )
        else:
            self._tasks = self._build_tasks(clips, query, skip_existing)

    @staticmethod
    def _resolve_tier(tier: Optional[str]) -> str:
        """Normalize the effective custom-query model tier."""
        resolved = tier or load_settings().description_model_tier
        if resolved in ("cpu", "gpu"):
            return "local" if resolved == "cpu" else "cloud"
        return resolved

    def _build_tasks(
        self, clips: list, query: str, skip_existing: bool
    ) -> list[CustomQueryTask]:
        """Build immutable task list from clips."""
        tasks = []
        for clip in clips:
            if skip_existing and self._has_query_result(clip, query):
                continue
            if not clip.thumbnail_path or not clip.thumbnail_path.exists():
                logger.warning(f"Skipping clip {clip.id}: thumbnail not found")
                continue
            tasks.append(
                CustomQueryTask(
                    clip_id=clip.id,
                    thumbnail_path=clip.thumbnail_path,
                    query=query,
                )
            )
        return tasks

    def _build_tasks_from_targets(
        self, targets: list, query: str, skip_existing: bool
    ) -> list[CustomQueryTask]:
        """Build immutable task list from AnalysisTarget objects."""
        tasks = []
        for target in targets:
            if skip_existing and self._has_query_result_target(target, query):
                continue
            image_path = target.image_path
            if not image_path or not image_path.exists():
                logger.warning(f"Skipping target {target.id}: image not found")
                continue
            tasks.append(
                CustomQueryTask(
                    clip_id=target.id,
                    thumbnail_path=image_path,
                    query=query,
                    target_type=target.target_type,
                )
            )
        return tasks

    @staticmethod
    def _has_query_result(clip, query: str) -> bool:
        """Check if a clip already has a result for this exact query."""
        if not clip.custom_queries:
            return False
        return any(q.get("query") == query for q in clip.custom_queries)

    @staticmethod
    def _has_query_result_target(target, query: str) -> bool:
        """Check if an analysis target already has a result for this query."""
        custom_queries = getattr(target, "custom_queries", None)
        if not custom_queries:
            return False
        return any(q.get("query") == query for q in custom_queries)

    def _process_task(
        self, task: CustomQueryTask
    ) -> tuple[str, str, Optional[bool], Optional[float], Optional[str], Optional[str]]:
        """Process a single task with retry logic.

        Returns:
            Tuple of (clip_id, query, match, confidence, model, error_message)
        """
        outcome = compute_custom_query(task, self.options, self._cancel_event)
        error = (
            "Cancelled"
            if outcome.status == "unprocessed"
            else outcome.message or outcome.code
        )
        return (
            outcome.clip_id,
            outcome.query,
            outcome.match,
            outcome.confidence,
            outcome.model,
            error,
        )

    def _preload_local_model(self, total: int) -> bool:
        """Load local VLM model in this worker thread before local inference."""
        try:
            from core.analysis.description import is_model_loaded, _load_local_model

            if not is_model_loaded(self.options.model):
                self.progress.emit(0, total)
                _load_local_model(self.options.model)
            return True
        except Exception as e:
            self.error.emit(f"Failed to load local VLM: {e}")
            return False

    def _handle_task_result(
        self,
        result: tuple[
            str,
            str,
            Optional[bool],
            Optional[float],
            Optional[str],
            Optional[str],
        ],
        errors: list[tuple[str, str]],
    ) -> None:
        """Emit success or collect an error from a processed task result."""
        clip_id, query, match, confidence, model, error_msg = result

        if error_msg and error_msg != "Cancelled":
            self._log_error(error_msg, clip_id)
            errors.append((clip_id, error_msg))
        elif match is not None:
            self.query_result_ready.emit(clip_id, query, match, confidence, model)

    @property
    def tasks(self) -> tuple[CustomQueryTask, ...]:
        return tuple(self._tasks)

    def run(self) -> None:
        """Compute detached outcomes and always settle the worker lifecycle."""
        self._log_start()
        errors: list[tuple[str, str]] = []
        try:
            if not self.tasks or self.is_cancelled():
                return
            if self._tier == "local" and not self._preload_local_model(len(self.tasks)):
                return

            def deliver(outcome: CustomQueryOutcome) -> None:
                self._handle_task_result(
                    (
                        outcome.clip_id,
                        outcome.query,
                        outcome.match,
                        outcome.confidence,
                        outcome.model,
                        outcome.message or outcome.code,
                    ),
                    errors,
                )

            self.result = run_custom_query(
                self.tasks,
                self.options,
                cancel_event=self._cancel_event,
                on_outcome=deliver,
                progress=self.progress.emit,
            )
            if errors:
                self.error.emit(self._summarize_errors(errors))
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.analysis_completed.emit()
            self._log_complete()

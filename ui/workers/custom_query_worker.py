"""Background worker for custom visual query evaluation.

Runs a yes/no VLM query across multiple clips in a background thread.
Cloud APIs use ThreadPoolExecutor for parallelism and retry logic; local
VLM inference runs serially in the worker thread because MLX stream state is
thread-local.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal

from core.settings import load_settings
from ui.workers.base import CancellableWorker, is_transient_provider_error

logger = logging.getLogger(__name__)

# Retry configuration for cloud API rate limits
_MAX_RETRIES = 3
_RETRY_DELAYS = [2, 5, 10]  # seconds


@dataclass(frozen=True)
class CustomQueryTask:
    """Immutable task data for thread pool execution."""

    clip_id: str
    thumbnail_path: Path
    query: str


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
    query_result_ready = Signal(str, str, bool, float, str)  # clip_id, query, match, confidence, model
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
        return (
            f"Custom query failed for {len(errors)} clips:\n\n"
            f"{preview}{extra}"
        )

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
    ):
        super().__init__(parent)
        self._query = query
        self._tier = self._resolve_tier(tier)
        requested_parallelism = min(max(1, parallelism), 5)
        self._parallelism = 1 if self._tier == "local" else requested_parallelism
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
        if self.is_cancelled():
            return task.clip_id, task.query, None, None, None, "Cancelled"

        from core.analysis.custom_query import evaluate_custom_query

        last_error = None
        for attempt in range(_MAX_RETRIES + 1):
            if self.is_cancelled():
                return task.clip_id, task.query, None, None, None, "Cancelled"

            try:
                match, confidence, model = evaluate_custom_query(
                    image_path=task.thumbnail_path,
                    query=task.query,
                    tier=self._tier,
                )
                return task.clip_id, task.query, match, confidence, model, None
            except Exception as e:
                last_error = str(e)
                if is_transient_provider_error(last_error) and attempt < _MAX_RETRIES:
                    delay = _RETRY_DELAYS[attempt]
                    logger.warning(
                        f"Transient query failure for {task.clip_id}, "
                        f"retry {attempt + 1}/{_MAX_RETRIES} in {delay}s"
                    )
                    time.sleep(delay)
                    continue
                break

        return task.clip_id, task.query, None, None, None, last_error

    def _preload_local_model(self, total: int) -> bool:
        """Load local VLM model in this worker thread before local inference."""
        try:
            from core.analysis.description import is_model_loaded, _load_local_model

            if not is_model_loaded():
                self.progress.emit(0, total)
                _load_local_model()
            return True
        except Exception as e:
            self.error.emit(f"Failed to load local VLM: {e}")
            self._log_complete()
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

    def _run_local_serial(self, total: int, errors: list[tuple[str, str]]) -> int:
        """Run local VLM inference serially in this worker thread."""
        completed = 0
        for task in self._tasks:
            if self.is_cancelled():
                self._log_cancelled()
                break

            completed += 1
            result = self._process_task(task)
            self._handle_task_result(result, errors)
            self.progress.emit(completed, total)
        return completed

    def _run_cloud_parallel(self, total: int, errors: list[tuple[str, str]]) -> int:
        """Run cloud VLM query tasks in a bounded thread pool."""
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
                    self._handle_task_result(future.result(), errors)
                except Exception as e:
                    self._log_error(str(e), task.clip_id)
                    errors.append((task.clip_id, str(e)))

                self.progress.emit(completed, total)
        return completed

    def run(self):
        """Execute custom query evaluation on all clips."""
        self._log_start()

        total = len(self._tasks)
        if total == 0:
            logger.info("No clips to process for custom query")
            self.analysis_completed.emit()
            self._log_complete()
            return

        logger.info(
            f"Starting custom query '{self._query}': {total} clips, "
            f"tier={self._tier}, parallelism={self._parallelism}"
        )

        errors: list[tuple[str, str]] = []

        if self._tier == "local":
            if not self._preload_local_model(total):
                return
            self._run_local_serial(total, errors)
        else:
            self._run_cloud_parallel(total, errors)

        if errors:
            self.error.emit(self._summarize_errors(errors))

        self.analysis_completed.emit()
        self._log_complete()

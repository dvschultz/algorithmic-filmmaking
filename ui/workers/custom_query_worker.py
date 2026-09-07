"""Background worker for custom visual query evaluation.

Runs a yes/no VLM query across multiple clips in a background thread.
Cloud APIs use ThreadPoolExecutor for parallelism and retry logic; local
VLM preparation and inference share one runtime thread because MLX stream state is
thread-local.
"""

import logging
from dataclasses import asdict
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Signal

from core.settings import load_settings
from core.jobs import JobRuntime
from core.jobs.spec import OperationSpec
from core.jobs.custom_query import _target, _provenance
from core.jobs.gui_custom_query import GuiCustomQueryCache
from core.jobs.media import media_stamp
from ui.workers.job_adapter import (
    gui_job_operation,
    gui_job_runtime,
    close_gui_job_runtime,
)
from ui.workers.base import CancellableWorker
from core.operations.custom_query import (
    CustomQueryTask,
    CustomQueryOutcome,
    compute_custom_query,
    resolve_options,
    run_custom_query,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.project import Project


class CustomQueryWorker(CancellableWorker):
    """Background worker for evaluating custom visual queries.

    Cloud tier uses ThreadPoolExecutor for parallel processing. Local tier
    runs serially inside the shared runtime worker because MLX stream state is
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
        *,
        project: Optional["Project"] = None,
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
        targets = {
            task.clip_id: _target(project, task.clip_id)
            for task in self.tasks
            if project is not None and task.target_type == "clip"
        }
        if project is not None and any(
            task.target_type != "clip" for task in self.tasks
        ):
            raise ValueError("Frame custom-query storage is not supported")
        self._media_stamps = {
            task.thumbnail_path: media_stamp(task.thumbnail_path)
            for task in self.tasks
            if task.thumbnail_path is not None
        }
        for target in targets.values():
            if target["source_path"]:
                path = Path(target["source_path"])
                self._media_stamps[path] = media_stamp(path)
        self.operation = gui_job_operation(
            OperationSpec.build(
                kind="custom_query",
                version=1,
                arguments={
                    "clip_ids": [task.clip_id for task in self.tasks],
                    "query": query,
                },
                inputs={
                    "targets": targets,
                    "options": asdict(self.options),
                    "runtime": _provenance(self.options),
                },
                persistence="session_only",
                session_id=project.session.session_id if project is not None else None,
                input_revision=str(project.mutation_generation)
                if project is not None
                else None,
            ),
            project.path if project is not None else None,
        )
        self.task_id: str | None = None
        self.job_status: str | None = None
        self._runtime: JobRuntime | None = None
        self._errors: list[tuple[str, str]] = []
        self.cache: GuiCustomQueryCache | None = None
        if project is not None and project.path is not None:
            self.cache = GuiCustomQueryCache(
                project.path,
                project.metadata.id,
                {cid: target["source_id"] for cid, target in targets.items()},
                project.metadata.job_results,
                options=self.options,
                previous_queries={
                    cid: project.clips_by_id[cid].custom_queries or []
                    for cid in targets
                },
                targets=targets,
                media_stamps=self._media_stamps,
            )

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

    @property
    def tasks(self) -> tuple[CustomQueryTask, ...]:
        return tuple(self._tasks)

    def cancel(self) -> None:
        super().cancel()
        if self._runtime is not None and self.task_id is not None:
            self._runtime.cancel(self.task_id)

    def _prepare(self) -> bool:
        if self.is_cancelled() or not self.tasks:
            return False
        if any(
            media_stamp(path) != stamp for path, stamp in self._media_stamps.items()
        ):
            raise RuntimeError("Custom-query media changed while queued")
        if self._tier == "local":
            try:
                from core.analysis.description import is_model_loaded, _load_local_model

                if not is_model_loaded(self.options.model):
                    _load_local_model(self.options.model)
            except Exception as exc:
                if self.is_cancelled():
                    return False
                raise RuntimeError(f"Failed to load local VLM: {exc}") from exc
        if any(
            media_stamp(path) != stamp for path, stamp in self._media_stamps.items()
        ):
            raise RuntimeError("Custom-query media changed during preparation")
        return not self.is_cancelled()

    def run(self) -> None:
        self._log_start()
        runtime = None
        events: Queue[tuple[str, object]] = Queue()

        def emit(event):
            kind, value = event
            if kind == "outcome":
                self._on_outcome(value)
            else:
                self.progress.emit(*value)

        def compute(progress, cancel):
            collected = {}

            def deliver(outcome):
                collected[outcome.clip_id] = outcome
                events.put(("outcome", outcome))

            def report(current, total):
                progress(
                    current / total if total else 1.0,
                    f"Custom query ({current}/{total})",
                )
                events.put(("progress", (current, total)))

            try:
                if self.cache is not None:
                    outcomes = self.cache.run(
                        self.tasks, cancel, self._prepare, deliver, report
                    )
                else:
                    report(0, len(self.tasks))
                    if self._prepare():
                        outcomes = run_custom_query(
                            self.tasks,
                            self.options,
                            cancel_event=cancel,
                            on_outcome=deliver,
                            progress=report,
                        )
                    else:
                        outcomes = tuple(
                            CustomQueryOutcome(
                                task.clip_id,
                                task.query,
                                "unprocessed",
                                code="cancelled",
                            )
                            for task in self.tasks
                        )
                return {"outcomes": [asdict(outcome) for outcome in outcomes]}
            except Exception as exc:
                for task in self.tasks:
                    if task.clip_id not in collected:
                        deliver(
                            CustomQueryOutcome(
                                task.clip_id,
                                task.query,
                                "failed",
                                code="custom_query_failed",
                                message=str(exc),
                            )
                        )
                return {
                    "success": False,
                    "error": str(exc),
                    "outcomes": [
                        asdict(collected[task.clip_id]) for task in self.tasks
                    ],
                }

        try:
            runtime = gui_job_runtime(self.operation)
            self._runtime = runtime
            submission = runtime.submit(
                kind=self.operation.kind,
                args=self.operation.arguments,
                operation=self.operation,
                run=compute,
                cancellation_event=self._cancel_event,
                project_path=self.operation.arguments.get("project_path"),
            )
            self.task_id = submission["task_id"]
            while runtime.is_handle_live(self.task_id):
                try:
                    emit(events.get(timeout=0.05))
                except Empty:
                    pass
            runtime.shutdown()
            while not events.empty():
                emit(events.get_nowait())
            row = runtime.store.get(self.task_id)
            self.job_status = row.status
            self.result = tuple(
                CustomQueryOutcome(**value)
                for value in (row.result or {}).get("outcomes", [])
            )
            if row.status == "cancelled" and not self.result:
                self.result = tuple(
                    CustomQueryOutcome(
                        task.clip_id, task.query, "unprocessed", code="cancelled"
                    )
                    for task in self.tasks
                )
            if row.status == "failed" and not self.result:
                raise RuntimeError(row.error or "Custom query failed")
        except Exception as exc:
            for task in self.tasks:
                self._on_outcome(
                    CustomQueryOutcome(
                        task.clip_id,
                        task.query,
                        "failed",
                        code="custom_query_failed",
                        message=str(exc),
                    )
                )
        finally:
            try:
                if runtime is not None:
                    close_gui_job_runtime(runtime)
            finally:
                self._runtime = None
                if self._errors:
                    self.error.emit(self._summarize_errors(self._errors))
                self.analysis_completed.emit()
                self._log_complete()

    def _on_outcome(self, outcome: CustomQueryOutcome) -> None:
        if outcome.status == "failed":
            message = outcome.message or outcome.code or "Custom query failed"
            self._log_error(message, outcome.clip_id)
            self._errors.append((outcome.clip_id, message))
        elif outcome.status == "succeeded":
            self.query_result_ready.emit(
                outcome.clip_id,
                outcome.query,
                outcome.match,
                outcome.confidence,
                outcome.model,
            )

"""Background worker for generating video descriptions.

Runs VLM description on multiple clips in a background thread,
using ThreadPoolExecutor for parallelism with retry logic for cloud APIs.
"""

import logging
from dataclasses import asdict, replace
from queue import Empty, Queue
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Signal

from core.settings import load_settings
from core.jobs import JobRuntime
from core.jobs.spec import OperationSpec
from core.jobs.description import _runtime, _task_data
from core.jobs.gui_description import GuiDescriptionCache
from core.jobs.media import media_stamp
from ui.workers.job_adapter import (
    gui_job_operation,
    gui_job_runtime,
    close_gui_job_runtime,
)
from ui.workers.base import CancellableWorker
from core.operations.description import (
    DEFAULT_PROMPT,
    DescriptionTask,
    DescriptionOutcome,
    DescriptionOptions,
    compute_description,
    run_description,
    resolve_options,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.project import Project


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
        *,
        project: Optional["Project"] = None,
        options: Optional[DescriptionOptions] = None,
    ) -> None:
        super().__init__(parent)
        self._tier = options.tier if options is not None else self._resolve_tier(tier)
        self._prompt = (
            options.prompt if options is not None else prompt or DEFAULT_PROMPT
        )
        requested_parallelism = min(
            max(1, options.parallelism if options is not None else parallelism), 5
        )
        # Local MLX/Moondream inference shares model state and can crash native
        # backends if multiple descriptions run at once.
        self._parallelism = 1 if self._tier == "local" else requested_parallelism
        self.options = options or resolve_options(
            self._tier, self._prompt, self._parallelism
        )
        self.options = replace(self.options, parallelism=self._parallelism)
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
        self._media_stamps = {
            path: media_stamp(path)
            for task in self.tasks
            for path in (task.thumbnail_path, task.source_path)
            if path is not None
        }
        self.operation = gui_job_operation(
            OperationSpec.build(
                kind="describe",
                version=1,
                arguments={
                    "clip_ids": [task.clip_id for task in self.tasks],
                    "force": not skip_existing,
                },
                inputs={
                    "targets": [_task_data(task) for task in self.tasks],
                    "options": asdict(self.options),
                    "runtime": _runtime(self.options),
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
        self.cache: GuiDescriptionCache | None = None
        if project is not None and project.path is not None:
            targets = {
                task.clip_id: (
                    project.frames_by_id
                    if task.target_type == "frame"
                    else project.clips_by_id
                )[task.clip_id]
                for task in self.tasks
            }
            self.cache = GuiDescriptionCache(
                project.path,
                project.metadata.id,
                {cid: target.source_id or "" for cid, target in targets.items()},
                project.metadata.job_results,
                options=self.options,
                previous_descriptions={
                    cid: {
                        "description": target.description,
                        "model": target.description_model,
                        "frames": getattr(target, "description_frames", None),
                        "clip_id": getattr(target, "clip_id", None),
                        "frame_number": getattr(target, "frame_number", None),
                    }
                    for cid, target in targets.items()
                },
                media_stamps=self._media_stamps,
            )

    @staticmethod
    def _resolve_tier(tier: Optional[str]) -> str:
        """Normalize the effective description tier."""
        resolved = tier or load_settings().description_model_tier
        if resolved in ("cpu", "gpu"):
            return "local"
        return resolved

    @property
    def tasks(self) -> tuple[DescriptionTask, ...]:
        return tuple(self._tasks)

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
                    target_type=target.target_type,
                )
            )
        return tasks

    def _process_task(
        self, task: DescriptionTask
    ) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
        """Compatibility wrapper for callers testing one detached task."""
        outcome = compute_description(task, self.options, self._cancel_event)
        error = "Cancelled" if outcome.status == "unprocessed" else outcome.message
        return outcome.clip_id, outcome.description, outcome.model, error

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
            raise RuntimeError("Description media changed while queued")
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
            raise RuntimeError("Description media changed during preparation")
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
                    current / total if total else 1.0, f"Describing ({current}/{total})"
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
                        outcomes = run_description(
                            self.tasks,
                            self.options,
                            cancel_event=cancel,
                            on_outcome=deliver,
                            progress=report,
                        )
                    else:
                        outcomes = tuple(
                            DescriptionOutcome(
                                task.clip_id, "unprocessed", code="cancelled"
                            )
                            for task in self.tasks
                        )
                return {"outcomes": [asdict(outcome) for outcome in outcomes]}
            except Exception as exc:
                for task in self.tasks:
                    if task.clip_id not in collected:
                        deliver(
                            DescriptionOutcome(
                                task.clip_id,
                                "failed",
                                code="description_failed",
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
                DescriptionOutcome(**value)
                for value in (row.result or {}).get("outcomes", [])
            )
            if row.status == "cancelled" and not self.result:
                self.result = tuple(
                    DescriptionOutcome(task.clip_id, "unprocessed", code="cancelled")
                    for task in self.tasks
                )
            if row.status == "failed" and not self.result:
                raise RuntimeError(row.error or "Description failed")
        except Exception as exc:
            for task in self.tasks:
                self._on_outcome(
                    DescriptionOutcome(
                        task.clip_id,
                        "failed",
                        code="description_failed",
                        message=str(exc),
                    )
                )
        finally:
            try:
                if runtime is not None:
                    close_gui_job_runtime(runtime)
            finally:
                self._runtime = None
                self.description_completed.emit()
                self._log_complete()

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

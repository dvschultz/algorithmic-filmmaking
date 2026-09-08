"""Background worker for rich cinematography analysis.

Runs cinematography analysis on multiple clips in a background thread,
emitting progress signals to keep the UI responsive.
"""

import logging
from dataclasses import asdict, replace
from queue import Empty, Queue
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import Signal

from core.jobs import JobRuntime
from core.jobs.cinematography import _task_data
from core.jobs.gui_cinematography import GuiCinematographyCache
from core.jobs.media import media_stamp
from core.jobs.spec import OperationSpec

from core.operations.cinematography import (
    CinematographyTask as ClipAnalysisTask,
    CinematographyOutcome,
    CinematographyOptions,
    compute_cinematography,
    cinematography_task,
    resolve_options,
    run_cinematography,
)
from models.cinematography import CinematographyAnalysis
from ui.workers.base import CancellableWorker, summarize_clip_errors
from ui.workers.job_adapter import (
    gui_job_operation,
    gui_job_runtime,
    close_gui_job_runtime,
)

if TYPE_CHECKING:
    from core.project import Project

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
    outcome_ready = Signal(object)
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
        *,
        project: Optional["Project"] = None,
        options: Optional[CinematographyOptions] = None,
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
        self.options = options or resolve_options(mode, model, parallelism)
        self._parallelism = (
            1
            if self.options.tier == "local"
            else min(max(1, self.options.parallelism), 5)
        )
        self.options = replace(self.options, parallelism=self._parallelism)
        self.result: tuple[CinematographyOutcome, ...] = ()

        # Build immutable task list upfront - no mutable state in thread pool
        if analysis_targets:
            self._tasks = self._build_tasks_from_targets(
                analysis_targets, skip_existing
            )
        else:
            self._tasks = self._build_tasks(clips, sources_by_id, skip_existing)

        self._media_stamps = {
            path: media_stamp(path)
            for task in self.tasks
            for path in (task.thumbnail_path, task.source_path)
            if path is not None
        }
        self.operation = gui_job_operation(
            OperationSpec.build(
                kind="cinematography",
                version=2,
                arguments={"clip_ids": [task.clip_id for task in self.tasks]},
                inputs={
                    "tasks": [_task_data(task) for task in self.tasks],
                    "options": asdict(self.options),
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
        self.cache: GuiCinematographyCache | None = None
        if project is not None and project.path is not None:
            targets = {
                task.clip_id: (
                    project.frames_by_id
                    if task.target_type == "frame"
                    else project.clips_by_id
                )[task.clip_id]
                for task in self.tasks
            }
            self.cache = GuiCinematographyCache(
                project.path,
                project.metadata.id,
                {cid: target.source_id or "" for cid, target in targets.items()},
                project.metadata.job_results,
                options=self.options,
                previous_results={
                    cid: {
                        "analysis": target.cinematography.to_dict()
                        if target.cinematography
                        else None,
                        "shot_type": target.shot_type,
                        "frame_clip_id": getattr(target, "clip_id", None),
                        "frame_number": getattr(target, "frame_number", None),
                        "source_path": str(
                            project.sources_by_id[target.source_id].file_path
                        )
                        if target.source_id in project.sources_by_id
                        else None,
                    }
                    for cid, target in targets.items()
                },
                media_stamps=self._media_stamps,
            )

    def _build_tasks(
        self, clips: list, sources_by_id: dict, skip_existing: bool
    ) -> list[ClipAnalysisTask]:
        """Build immutable task list from clips.

        Filters and validates clips, creating frozen dataclass instances
        that can safely be passed to the thread pool.
        """
        tasks = []

        for clip in clips:
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
            tasks.append(cinematography_task(clip, source, skip_existing=skip_existing))

        return tasks

    def _build_tasks_from_targets(
        self, targets: list, skip_existing: bool
    ) -> list[ClipAnalysisTask]:
        """Build immutable task list from AnalysisTarget objects."""
        tasks = []

        for target in targets:
            image_path = target.image_path
            if not image_path or not image_path.exists():
                logger.warning(f"Skipping target {target.id}: image not found")
                continue

            tasks.append(cinematography_task(target, skip_existing=skip_existing))

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

    def cancel(self) -> None:
        super().cancel()
        if self._runtime is not None and self.task_id is not None:
            self._runtime.cancel(self.task_id)

    def _prepare(self) -> bool:
        if any(
            media_stamp(path) != stamp for path, stamp in self._media_stamps.items()
        ):
            raise RuntimeError("Cinematography media changed while queued")
        return not self.is_cancelled()

    def run(self) -> None:
        """Relay shared-job results on the QThread; publish models on their owner."""
        self._log_start()
        runtime = None
        events: Queue = Queue()
        results: dict[str, CinematographyAnalysis] = {}
        errors: list[tuple[str, str]] = []

        def emit(event):
            kind, value = event
            if kind == "progress":
                self.progress.emit(*value)
                return
            if value.can_apply:
                self.outcome_ready.emit(value)
            if value.has_result:
                results[value.clip_id] = value.analysis
                self.clip_completed.emit(value.clip_id, value.analysis)
            elif value.status == "failed":
                errors.append(
                    (value.clip_id, value.message or value.code or "Analysis failed")
                )

        def compute(progress, cancel):
            collected = {}

            def deliver(outcome):
                collected[outcome.clip_id] = outcome
                events.put(("outcome", outcome))

            def report(current, total, cid):
                progress(
                    current / total if total else 1.0,
                    f"Cinematography ({current}/{total})",
                )
                events.put(("progress", (current, total, cid)))

            try:
                if self.cache is not None:
                    outcomes = self.cache.run(
                        self.tasks, cancel, self._prepare, deliver, report
                    )
                elif self._prepare():
                    outcomes = run_cinematography(
                        self.tasks,
                        self.options,
                        cancel_event=cancel,
                        on_outcome=deliver,
                        progress=report,
                    )
                else:
                    outcomes = tuple(
                        CinematographyOutcome(
                            task.clip_id, "unprocessed", code="cancelled"
                        )
                        for task in self.tasks
                    )
                return {"outcomes": [asdict(outcome) for outcome in outcomes]}
            except Exception as exc:
                for task in self.tasks:
                    if task.clip_id not in collected:
                        deliver(
                            CinematographyOutcome(
                                task.clip_id,
                                "failed",
                                code="cinematography_failed",
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
                CinematographyOutcome(**value)
                for value in (row.result or {}).get("outcomes", [])
            )
            if row.status == "cancelled" and not self.result:
                self.result = tuple(
                    CinematographyOutcome(task.clip_id, "unprocessed", code="cancelled")
                    for task in self.tasks
                )
            if row.status == "failed" and not self.result:
                raise RuntimeError(row.error or "Cinematography failed")
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            try:
                if runtime is not None:
                    close_gui_job_runtime(runtime)
            finally:
                self._runtime = None
                if errors and not self.is_cancelled():
                    self.error.emit(_summarize_errors(errors))
                self.analysis_completed.emit(results)
                self._log_complete()

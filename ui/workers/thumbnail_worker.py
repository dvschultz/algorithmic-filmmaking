"""Qt scheduling for detached thumbnail generation."""

from dataclasses import asdict
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING

from PySide6.QtCore import Signal

from core.jobs import JobRuntime
from core.jobs.spec import OperationSpec
from core.operations.thumbnails import (
    ThumbnailApplication,
    ThumbnailOptions,
    ThumbnailOutcome,
    ThumbnailTask,
    run_thumbnails,
)
from ui.workers.base import CancellableWorker

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip, Source


class ThumbnailWorker(CancellableWorker):
    progress = Signal(int, int)
    outcome_ready = Signal(object)

    def __init__(
        self,
        source: "Source | None",
        clips: list["Clip"],
        cache_dir: Path | None = None,
        sources_by_id: dict[str, "Source"] | None = None,
        *,
        project: "Project | None" = None,
    ) -> None:
        super().__init__()
        if cache_dir is None:
            from core.settings import load_settings

            cache_dir = load_settings().thumbnail_cache_dir
        sources = sources_by_id or {}
        self.tasks = tuple(
            ThumbnailTask.capture(
                clip,
                sources.get(clip.source_id)
                or (source if source and source.id == clip.source_id else None),
            )
            for clip in clips
        )
        self.options = ThumbnailOptions(cache_dir, width=160, height=90)
        self.application = (
            ThumbnailApplication(project, self.tasks, clips=clips)
            if project is not None
            else None
        )
        self.result: tuple[ThumbnailOutcome, ...] = ()
        self.task_id: str | None = None
        self._runtime: JobRuntime | None = None
        self.operation = OperationSpec.build(
            kind="thumbnails",
            version=1,
            arguments={"clip_ids": [t.clip_id for t in self.tasks]},
            inputs={
                "tasks": [
                    {
                        **asdict(t),
                        # Malformed timing still needs a per-item outcome;
                        # nonfinite numbers cannot be encoded in job JSON.
                        "fps": repr(t.fps),
                        "start_frame": repr(t.start_frame),
                        "end_frame": repr(t.end_frame),
                        "source_path": str(t.source_path) if t.source_path else None,
                        "previous_path": str(t.previous_path)
                        if t.previous_path
                        else None,
                    }
                    for t in self.tasks
                ],
                "options": {**asdict(self.options), "cache_dir": str(cache_dir)},
            },
            persistence="session_only",
            session_id=project.session.session_id if project is not None else None,
        )

    def cancel(self) -> None:
        super().cancel()
        if self._runtime is not None and self.task_id is not None:
            self._runtime.cancel(self.task_id)

    def run(self) -> None:
        runtime = None
        events: Queue[tuple[int, int]] = Queue()

        def compute(progress, cancel):
            def report(current, total, outcome):
                progress(current / total if total else 1.0, outcome.clip_id)
                events.put((current, total))

            return {
                "outcomes": [
                    asdict(o)
                    for o in run_thumbnails(self.tasks, self.options, cancel, report)
                ]
            }

        try:
            runtime = JobRuntime.for_session(max_workers=1)
            self._runtime = runtime
            submission = runtime.submit(
                kind="thumbnails",
                args=self.operation.arguments,
                operation=self.operation,
                run=compute,
                cancellation_event=self._cancel_event,
            )
            self.task_id = submission["task_id"]
            while runtime.is_handle_live(self.task_id):
                try:
                    self.progress.emit(*events.get(timeout=0.05))
                except Empty:
                    pass
            runtime.shutdown()
            while not events.empty():
                self.progress.emit(*events.get_nowait())
            row = runtime.store.get(self.task_id)
            if row.status == "failed":
                raise RuntimeError(row.error or "Thumbnail generation failed")
            self.result = (
                tuple(ThumbnailOutcome(**o) for o in row.result["outcomes"])
                if row.result
                else tuple(
                    ThumbnailOutcome(t.clip_id, "unprocessed", code="cancelled")
                    for t in self.tasks
                )
            )
            for outcome in self.result:
                self.outcome_ready.emit(outcome)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if runtime is not None:
                runtime.close_session()
            self._runtime = None

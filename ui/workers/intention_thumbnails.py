"""Bind shared thumbnail delivery and cancellation to one intention plan."""

from typing import Any

from PySide6.QtCore import Slot

from core.intention_workflow import WorkflowState
from core.jobs.media import media_stamp
from ui.workers.intention_run import IntentionRun
from ui.workers.thumbnail_delivery import ThumbnailDelivery
from ui.workers.thumbnail_worker import ThumbnailWorker


class IntentionThumbnailDelivery(ThumbnailDelivery):
    def __init__(self, window: Any, run: IntentionRun) -> None:
        clips = run.workflow.get_all_clips()
        sources = run.workflow.get_all_sources()
        source_map = {source.id: source for source in sources}
        if (
            not clips
            or any(run.project.clips_by_id.get(clip.id) is not clip for clip in clips)
            or any(run.project.sources_by_id.get(s.id) is not s for s in sources)
            or any(clip.source_id not in source_map for clip in clips)
        ):
            raise ValueError("Thumbnail inputs changed; start the workflow again")
        worker = ThumbnailWorker(
            source=sources[0] if sources else None,
            clips=clips,
            cache_dir=window.settings.thumbnail_cache_dir,
            sources_by_id=source_map,
            project=run.project,
        )
        self.run_identity = run
        self.sources = sources
        self.cancelled = False
        self._started = False
        self.errors: list[str] = []
        super().__init__(
            window,
            worker,
            ready=self._ready,
            completed=self._complete,
            progress=run.workflow.on_thumbnail_progress,
        )
        window.thumbnail_worker = worker
        window._intention_thumbnails = self
        if not hasattr(window, "_active_intention_thumbnails"):
            window._active_intention_thumbnails = set()
        window._active_intention_thumbnails.add(self)
        worker.error.connect(self._run_error)
        run.workflow.workflow_cancelled.connect(self.cancel)
        run.workflow.workflow_completed.connect(self._workflow_completed)

    def _current(self) -> bool:
        return bool(
            not self.cancelled
            and not self._cleaned
            and super()._current()
            and getattr(self.window, "_intention_thumbnails", None) is self
            and self.run_identity.is_current(self.window, WorkflowState.THUMBNAILS)
        )

    def _input_current(self, clip_id: str) -> bool:
        project = self.run_identity.project
        task = self.application.tasks[clip_id]
        clip, source = self.application.bindings[clip_id]
        return (
            clip is not None
            and source is not None
            and project.clips_by_id.get(clip_id) is clip
            and project.sources_by_id.get(task.source_id) is source
            and clip.source_id == task.source_id
            and (clip.start_frame, clip.end_frame) == (task.start_frame, task.end_frame)
            and source.file_path == task.source_path
            and source.fps == task.fps
            and media_stamp(source.file_path) == task.source_stamp
        )

    def _inputs_current(self) -> bool:
        return all(self._input_current(task.clip_id) for task in self.worker.tasks)

    def start(self) -> None:
        if self._started or self._cleaned:
            return
        self._started = True
        try:
            if not self._current():
                self.cancel()
                self._cleanup()
                return
            self.worker.start()
        except Exception as exc:
            if self._current():
                self.run_identity.workflow.fail(str(exc))
            self.cancel()
            if not self.worker.isRunning():
                self._cleanup()

    def _ready(self, clip_id: str, path: str) -> None:
        if self._current():
            try:
                if not self._input_current(clip_id):
                    raise ValueError(
                        "Thumbnail inputs changed during model publication"
                    )
                self.window._on_thumbnail_ready(clip_id, path)
            except Exception as exc:
                if self._current():
                    self.run_identity.workflow.fail(str(exc))

    @Slot(str)
    def _run_error(self, message: str) -> None:
        if self.sender() is self.worker and self._current():
            self.errors.append(message)

    def _complete(self) -> None:
        workflow = self.run_identity.workflow
        try:
            if not self._inputs_current():
                raise ValueError("Thumbnail inputs changed; start the workflow again")
            if self.errors:
                raise ValueError("\n".join(self.errors))
            if (
                tuple(o.clip_id for o in self.worker.result)
                != tuple(t.clip_id for t in self.worker.tasks)
                or any(
                    o.status not in ("succeeded", "skipped") for o in self.worker.result
                )
                or any(
                    not clip.thumbnail_path or not clip.thumbnail_path.is_file()
                    for clip, _source in self.application.bindings.values()
                )
            ):
                raise ValueError("Required thumbnails could not be generated")
            self.window._sync_intention_workflow_ui(
                sources=self.sources,
                valid=lambda: self._current() and self._inputs_current(),
            )
            if self._current():
                if not self._inputs_current():
                    raise ValueError("Thumbnail inputs changed during UI refresh")
                workflow.on_thumbnails_finished()
        except Exception as exc:
            if self._current():
                workflow.fail(str(exc))

    @Slot(object)
    def _workflow_completed(self, _result: Any) -> None:
        self.cancel()

    @Slot()
    def cancel(self) -> None:
        if self.cancelled or self._cleaned:
            return
        self.cancelled = True
        self.worker.cancel()
        if self.window.thumbnail_worker is self.worker:
            self.window.thumbnail_worker = None
        run = self.run_identity
        if (
            self.window.intention_workflow is run.workflow
            and run.workflow.plan is run.plan
            and run.workflow.is_running
        ):
            run.workflow.cancel()

    def _cleanup(self) -> None:
        if self._cleaned:
            return
        if (
            not self._current()
            and self.run_identity.workflow.state == WorkflowState.THUMBNAILS
        ):
            self.cancel()
        super()._cleanup()
        self.window._active_intention_thumbnails.discard(self)
        if getattr(self.window, "_intention_thumbnails", None) is self:
            self.window._intention_thumbnails = None

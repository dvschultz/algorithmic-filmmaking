"""Qt delivery adapter for detection jobs in the shared runtime."""

from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING

from PySide6.QtCore import Signal

from core.jobs import JobRuntime
from core.jobs.detection import detection_job_spec, run_detection_job
from core.operations.detection import DetectionGuard, DetectionRequest
from ui.workers.base import CancellableWorker

if TYPE_CHECKING:
    from core.project import Project
    from core.scene_detect import DetectionConfig, KaraokeDetectionConfig


class DetectionWorker(CancellableWorker):
    """Background worker for scene detection.

    Supports both visual detection (adaptive/content) and text-based
    detection (karaoke mode).
    """

    job_started = Signal(str, str)  # task ID, persistence
    progress = Signal(float, str)  # progress (0-1), status message
    result_ready = Signal(object, object, list)  # guard, source, clips
    detection_completed = Signal(
        object, list
    )  # source, clips (renamed from 'finished' to avoid shadowing QThread.finished)

    def __init__(
        self,
        video_path: Path,
        config: DetectionConfig | None = None,
        mode: str = "adaptive",
        karaoke_config: KaraokeDetectionConfig | None = None,
        project: Project | None = None,
        source_id: str | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            from core.scene_detect import DetectionConfig

            config = DetectionConfig()
        self.video_path = video_path
        self.config = config
        self.mode = mode
        self.karaoke_config = karaoke_config
        self.guard = (
            DetectionGuard.capture(project, video_path, source_id=source_id)
            if project is not None
            else None
        )
        self.request = DetectionRequest.build(
            self.video_path, self.config, mode=mode, karaoke_config=karaoke_config
        )

        self.operation = detection_job_spec(self.request, self.guard)
        self.task_id: str | None = None
        self.job_status: str | None = None
        self._runtime: JobRuntime | None = None

    def cancel(self) -> None:
        super().cancel()
        if self._runtime is not None and self.task_id is not None:
            self._runtime.cancel(self.task_id)

    def run(self) -> None:
        self._log_start()
        runtime = None
        events: Queue[tuple[float, str]] = Queue()

        def compute(progress, cancel):
            def report(fraction, message):
                progress(fraction, message)
                events.put((fraction, message))

            return run_detection_job(self.request, report, cancel)

        try:
            runtime = JobRuntime.for_session(max_workers=1)
            self._runtime = runtime
            submission = runtime.submit(
                kind=self.operation.kind,
                args=self.operation.arguments,
                operation=self.operation,
                run=compute,
                cancellation_event=self._cancel_event,
            )
            self.task_id = submission["task_id"]
            self.job_started.emit(self.task_id, submission["persistence"])
            while runtime.is_handle_live(self.task_id):
                try:
                    self.progress.emit(*events.get(timeout=0.05))
                except Empty:
                    pass
            runtime.shutdown()
            while not events.empty():
                self.progress.emit(*events.get_nowait())
            row = runtime.store.get(self.task_id)
            self.job_status = row.status
            if row.status == "cancelled" or self.is_cancelled():
                self._log_cancelled()
                return
            if row.status != "completed":
                raise RuntimeError(row.error or "Detection failed")

            from models.clip import Source, Clip

            payload = row.result
            if payload is None:
                raise RuntimeError("Detection completed without a result")
            source = Source.from_dict(payload["source"])
            clips = [Clip.from_dict(item) for item in payload["clips"]]
            if self.guard is not None:
                self.result_ready.emit(self.guard, source, clips)
            self.detection_completed.emit(source, clips)
            self._log_complete()
        except Exception as exc:
            self._log_error(str(exc))
            if not self.is_cancelled():
                self.error.emit(str(exc))
        finally:
            try:
                if runtime is not None:
                    runtime.close_session()
            finally:
                self._runtime = None

"""Run-bound serial scene detection, advancing only after native completion."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from PySide6.QtCore import QObject, Slot

from core.intention_workflow import WorkflowState
from core.operations.detection import DetectionApplication, DetectionGuard
from core.scene_detect import DetectionConfig, KaraokeDetectionConfig
from ui.workers.detection_worker import DetectionWorker
from ui.workers.intention_run import IntentionRun

if TYPE_CHECKING:
    from models.clip import Source, Clip


class IntentionDetectionController(QObject):
    """Own every detection worker for one intention plan's detection phase."""

    def __init__(self, window: Any) -> None:
        super().__init__(window)
        self.window = window
        self.run_identity = IntentionRun.capture(window)
        self.project = self.run_identity.project
        self.workflow = self.run_identity.workflow
        self.plan = self.run_identity.plan
        self.worker: DetectionWorker | None = None
        self.application: DetectionApplication | None = None
        self.result: tuple[Source, list[Clip]] | None = None
        self.error: str | None = None
        self.cancelled = False
        self.disposed = False
        algorithm, _ = self.workflow.get_algorithm_with_direction()
        self.mode = "karaoke" if algorithm == "exquisite_corpus" else "adaptive"
        self.config = DetectionConfig(
            threshold=window.settings.default_sensitivity,
            min_scene_length=15,
            use_adaptive=True,
        )
        self.karaoke_config = (
            KaraokeDetectionConfig(
                roi_top_percent=0.0,
                text_similarity_threshold=60.0,
                confirm_frames=3,
                cut_offset=5,
            )
            if self.mode == "karaoke"
            else None
        )
        previous = getattr(window, "_intention_detection", None)
        if previous is not None:
            previous.cancel()
        window._intention_detection = self
        if not hasattr(window, "_active_intention_detections"):
            window._active_intention_detections = set()
        window._active_intention_detections.add(self)
        self.workflow.workflow_cancelled.connect(self.cancel)
        self.workflow.workflow_completed.connect(self._workflow_completed)

    def _current(self) -> bool:
        return bool(
            not self.cancelled
            and not self.disposed
            and self.window._intention_detection is self
            and self.run_identity.is_current(self.window, WorkflowState.DETECTING)
        )

    def start(self) -> None:
        if self.worker is not None:
            return
        while self._current():
            path = self.workflow.get_current_source_path()
            if path is None:
                break
            existing = getattr(self.window, "detection_worker", None)
            if existing is not None and existing.isRunning():
                self.workflow.fail("Another scene detection is still running")
                break
            # Reject queued results from the previous standalone channel.
            self.window._active_detection_guard = None
            self.window._active_detection_reply = None
            try:
                worker = DetectionWorker(
                    path,
                    self.config,
                    mode=self.mode,
                    karaoke_config=self.karaoke_config,
                    project=self.project,
                )
                assert worker.guard is not None
                self.application = DetectionApplication(self.project, worker.guard)
                self.worker = worker
                self.result = None
                self.error = None
                self.window.detection_worker = worker
                worker.setParent(self)
                worker.progress.connect(self._progress)
                worker.result_ready.connect(self._result)
                worker.error.connect(self._error)
                worker.finished.connect(self._finished)
                worker.start()
                return
            except Exception as exc:
                if self.worker is not None:
                    if self.worker.isRunning():
                        self.cancel()
                        return
                    if self.window.detection_worker is self.worker:
                        self.window.detection_worker = None
                    self.worker.deleteLater()
                    self.worker = None
                self.workflow.on_detection_error(str(exc))
        self._dispose()

    @Slot(float, str)
    def _progress(self, value: float, message: str) -> None:
        if self.sender() is self.worker and self._current():
            self.workflow.on_detection_progress(value, message)

    @Slot(object, object, list)
    def _result(self, guard: DetectionGuard, source: Source, clips: list[Clip]) -> None:
        if (
            self.worker is not None
            and self.sender() is self.worker
            and guard is self.worker.guard
            and self._current()
            and self.result is None
            and self.error is None
        ):
            self.result = (source, clips)

    @Slot(str)
    def _error(self, message: str) -> None:
        if (
            self.sender() is self.worker
            and self._current()
            and self.result is None
            and self.error is None
        ):
            self.error = message

    @Slot()
    def _finished(self) -> None:
        worker = self.worker
        if worker is None or self.sender() is not worker or worker.isRunning():
            return
        current = self._current() and self.window.detection_worker is worker
        self.worker = None
        if self.window.detection_worker is worker:
            self.window.detection_worker = None
        worker.deleteLater()
        try:
            if not current or worker.is_cancelled():
                self.cancel()
                return
            if (
                self.error is not None
                or worker.job_status != "completed"
                or self.result is None
            ):
                self.workflow.on_detection_error(
                    self.error or "Detection completed without a result"
                )
            else:
                source, clips = self.result
                try:
                    assert self.application is not None
                    added = self.application.source is None
                    applied_source = self.application.apply(
                        source, clips, still_current=self._current
                    )
                except Exception as exc:
                    if self._current():
                        self.workflow.on_detection_error(str(exc))
                else:
                    if applied_source is not None and self._current():
                        if added:
                            self.window.collect_tab.add_source(applied_source)
                        if self._current():
                            self.window.cut_tab.set_source(applied_source)
                        if self._current():
                            self.workflow.on_detection_completed(applied_source, clips)
            self.start()
        except Exception as exc:
            if self._current():
                self.workflow.fail(str(exc))
        finally:
            if self.worker is None:
                self._dispose()

    @Slot(object)
    def _workflow_completed(self, _result: Any) -> None:
        self.cancel()

    @Slot()
    def cancel(self) -> None:
        if self.cancelled:
            return
        self.cancelled = True
        if self.worker is not None:
            self.worker.cancel()
            if self.window.detection_worker is self.worker:
                self.window.detection_worker = None
        else:
            self._dispose()
        if (
            self.window.intention_workflow is self.workflow
            and self.workflow.plan is self.plan
            and self.workflow.is_running
        ):
            self.workflow.cancel()

    def _dispose(self) -> None:
        if self.disposed or self.worker is not None:
            return
        self.disposed = True
        self.window._active_intention_detections.discard(self)
        if getattr(self.window, "_intention_detection", None) is self:
            self.window._intention_detection = None
        self.deleteLater()

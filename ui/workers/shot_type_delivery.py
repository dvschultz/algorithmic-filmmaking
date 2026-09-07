"""Owner-thread shot publication bound to its launching context."""

from typing import Any, Callable

from PySide6.QtCore import QObject, Slot

from core.operations.shots import ShotTypeApplication, ShotTypeOutcome
from ui.workers.analysis_pipeline_delivery import pipeline_can_continue


class ShotTypeDelivery(QObject):
    def __init__(
        self,
        window: Any,
        worker: Any,
        *,
        worker_attribute: str = "shot_type_worker",
        pipeline: bool = False,
        on_complete: Callable[[], None] | None = None,
        intention: bool = False,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.worker_attribute = worker_attribute
        self.application = ShotTypeApplication(window.project, worker.tasks)
        self.pipeline = pipeline
        self.run = getattr(window, "_analysis_run", None) if pipeline else None
        self.frame_run = getattr(window, "_frame_analysis_ops", None)
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.on_complete = on_complete
        self.delivered: set[tuple[str, str]] = set()
        self.intention = intention
        self.workflow = (
            getattr(window, "intention_workflow", None) if intention else None
        )
        if not hasattr(window, "_active_shot_workers"):
            window._active_shot_workers = set()
        window._active_shot_workers.add(worker)
        worker.outcome_ready.connect(self.result)
        if on_complete is not None:
            worker.finished.connect(self.completed)
        worker.finished.connect(self.cleanup)
        worker.finished.connect(self.deleteLater)

    def _is_current(self) -> bool:
        window = self.window
        return not (
            getattr(window, self.worker_attribute, None) is not self.worker
            or window.project is not self.application.project
            or window.project.session.session_id != self.application.session_id
            or window.project.path != self.application.path
            or self.worker.is_cancelled()
            or (self.reply is not None and not self.reply.is_current(window))
            or (
                self.intention
                and getattr(window, "intention_workflow", None) is not self.workflow
            )
            or (
                self.worker_attribute == "_frame_shot_worker"
                and getattr(window, "_frame_analysis_ops", None) is not self.frame_run
            )
            or (
                self.pipeline
                and (
                    getattr(window, "_analysis_run", None) is not self.run
                    or not pipeline_can_continue(window)
                )
            )
        )

    @Slot()
    def cleanup(self) -> None:
        self.window._active_shot_workers.discard(self.worker)
        if getattr(self.window, self.worker_attribute, None) is self.worker:
            setattr(self.window, self.worker_attribute, None)
        self.worker.deleteLater()

    @Slot()
    def completed(self) -> None:
        if self.on_complete is not None and self._is_current():
            callback, self.on_complete = self.on_complete, None
            callback()

    @Slot(object)
    def result(self, outcome: ShotTypeOutcome) -> None:
        window = self.window
        if not self._is_current():
            return
        try:
            if not isinstance(outcome, ShotTypeOutcome):
                raise ValueError("Invalid queued shot classification")
            key = outcome.target_type, outcome.clip_id
            if key in self.delivered or outcome.status != "succeeded":
                return
            self.delivered.add(key)
            accepted = self.application.apply(window.project, outcome)
        except Exception as exc:
            window._on_shot_type_error(f"Could not apply shot classification: {exc}")
            return
        if accepted:
            # The model is already updated. Only clip results need these widgets.
            if outcome.target_type == "clip":
                window._on_shot_type_ready(
                    outcome.clip_id, outcome.shot_type, outcome.confidence
                )
        elif outcome.status == "succeeded":
            window._on_shot_type_error(
                "Shot classification discarded because the target changed. Run analysis again."
            )

"""Owner-thread description delivery scoped to a project and worker."""

from typing import Any

from PySide6.QtCore import Slot

from ui.workers.qt_lifetime import RetiringQObject

from core.operations.description import DescriptionApplication, DescriptionOutcome
from ui.workers.analysis_pipeline_delivery import pipeline_can_continue


class DescriptionDelivery(RetiringQObject):
    def __init__(
        self,
        window: Any,
        worker: Any,
        *,
        worker_attribute: str = "description_worker",
        pipeline: bool = False,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.worker_attribute = worker_attribute
        self._delivered: set[str] = set()
        self.application = DescriptionApplication(window.project, worker.tasks)
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.pipeline = pipeline
        self.run = getattr(window, "_analysis_run", None) if pipeline else None
        worker.finished.connect(self.retire)
        worker.description_ready.connect(self.description)

    def _current(self) -> bool:
        return (
            getattr(self.window, self.worker_attribute, None) is self.worker
            and self.window.project is self.application.project
            and self.window.project.session.session_id == self.application.session_id
            and not self.worker.is_cancelled()
            and (self.reply is None or self.reply.is_current(self.window))
            and (
                not self.pipeline
                or (
                    getattr(self.window, "_analysis_run", None) is self.run
                    and pipeline_can_continue(self.window)
                )
            )
        )

    @Slot(str, str, str)
    def description(self, target_id: str, description: str, model: str) -> None:
        if not self._current() or target_id in self._delivered:
            return
        self._delivered.add(target_id)
        try:
            project = self.window.project
            cache = getattr(self.worker, "cache", None)
            outcome = DescriptionOutcome(target_id, "succeeded", description, model)
            receipt = None
            if cache is not None:
                if project.path is None or project.path.resolve() != cache.path:
                    raise ValueError("Project save location changed during description")
                receipt = cache.results[target_id]
                if not receipt.matches(outcome):
                    raise ValueError(
                        "Queued description differs from its recorded result"
                    )
            accepted = self.application.apply(
                project,
                outcome,
            )
            if accepted and receipt is not None:
                project.record_job_result(receipt.result_id, receipt.digest)
        except Exception as exc:
            self.window._on_description_error(
                target_id, f"Could not apply description: {exc}"
            )
            return
        if accepted:
            self.window._on_description_ready(target_id, description, model)
        else:
            self.window._on_description_error(
                target_id,
                "Description discarded because the target changed. Run description again.",
            )

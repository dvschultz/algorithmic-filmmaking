"""Owner-thread classification publication bound to its launch context."""

from typing import Any
from dataclasses import asdict

from PySide6.QtCore import Slot

from ui.workers.qt_lifetime import RetiringQObject

from core.operations.classification import (
    ClassificationApplication,
    ClassificationOutcome,
)
from ui.workers.analysis_pipeline_delivery import pipeline_can_continue


class ClassificationDelivery(RetiringQObject):
    def __init__(
        self,
        window: Any,
        worker: Any,
        *,
        worker_attribute: str = "classification_worker",
        pipeline: bool = False,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.worker_attribute = worker_attribute
        self.application = ClassificationApplication(
            window.project, worker.tasks, getattr(worker, "options", None)
        )
        self.pipeline = pipeline
        self.run = getattr(window, "_analysis_run", None) if pipeline else None
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.delivered: set[str] = set()
        worker.finished.connect(self.retire)
        if hasattr(worker, "outcome_ready"):
            worker.outcome_ready.connect(self.receive)
        else:
            worker.labels_ready.connect(self.result)

    @Slot(str, list)
    def result(self, target_id: str, labels: list) -> None:
        self.receive(
            ClassificationOutcome(
                target_id,
                "succeeded",
                tuple((label, float(confidence)) for label, confidence in labels),
            )
        )

    @Slot(object)
    def receive(self, outcome: ClassificationOutcome) -> None:
        window = self.window
        if not isinstance(outcome, ClassificationOutcome) or not outcome.can_apply:
            return
        target_id = outcome.clip_id
        if (
            getattr(window, self.worker_attribute, None) is not self.worker
            or window.project is not self.application.project
            or window.project.session.session_id != self.application.session_id
            or self.worker.is_cancelled()
            or (self.reply is not None and not self.reply.is_current(window))
            or (
                self.pipeline
                and (
                    getattr(window, "_analysis_run", None) is not self.run
                    or not pipeline_can_continue(window)
                )
            )
            or target_id in self.delivered
        ):
            return
        self.delivered.add(target_id)
        try:
            outcome = ClassificationOutcome.from_dict(asdict(outcome))
            receipt = None
            cache = getattr(self.worker, "cache", None)
            if cache is not None:
                if (
                    window.project.path is None
                    or window.project.path.resolve() != cache.path
                ):
                    raise ValueError(
                        "Project save location changed during classification"
                    )
                receipt = cache.results.get(target_id)
                if (receipt is not None and not receipt.matches(outcome)) or (
                    receipt is None
                    and cache.transient_outcomes.get(target_id) != asdict(outcome)
                ):
                    raise ValueError(
                        "Queued classification differs from its recorded result"
                    )
            accepted = self.application.apply(window.project, outcome)
            if accepted and receipt is not None:
                window.project.record_job_result(receipt.result_id, receipt.digest)
        except Exception as exc:
            window._on_classification_error(f"Could not apply classification: {exc}")
            return
        if not accepted:
            window._on_classification_error(
                "Classification discarded because the target changed. Run analysis again."
            )

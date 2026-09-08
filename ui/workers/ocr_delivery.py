"""Owner-thread OCR delivery bound to the launching worker and project."""

from dataclasses import asdict
from typing import Any, Callable

from PySide6.QtCore import Slot

from ui.workers.qt_lifetime import RetiringQObject

from core.operations.ocr import OcrApplication, OcrOutcome
from ui.workers.analysis_pipeline_delivery import pipeline_can_continue


class OcrDelivery(RetiringQObject):
    def __init__(
        self,
        window: Any,
        worker: Any,
        *,
        worker_attribute: str = "text_extraction_worker",
        pipeline: bool = False,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.worker_attribute = worker_attribute
        self.application = OcrApplication(
            window.project, worker.tasks, getattr(worker, "options", None)
        )
        self.pipeline = pipeline
        self.run = getattr(window, "_analysis_run", None) if pipeline else None
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.on_complete = on_complete
        self.frame_run = getattr(window, "_frame_analysis_ops", None)
        self.frame_workflow = on_complete is not None
        self.delivered: set[tuple[str, str]] = set()
        worker.outcome_ready.connect(self.result)
        worker.finished.connect(self.retire)
        if on_complete is not None:
            worker.extraction_completed.connect(self.completed)

    def _is_current(self) -> bool:
        window = self.window
        return (
            getattr(window, self.worker_attribute, None) is self.worker
            and window.project is self.application.project
            and window.project.session.session_id == self.application.session_id
            and not self.worker.is_cancelled()
            and (self.reply is None or self.reply.is_current(window))
            and (
                not self.frame_workflow
                or getattr(window, "_frame_analysis_ops", None) is self.frame_run
            )
            and (
                not self.pipeline
                or (
                    getattr(window, "_analysis_run", None) is self.run
                    and pipeline_can_continue(window)
                )
            )
        )

    @Slot(dict)
    def completed(self, _results: dict) -> None:
        if self.on_complete is not None and self._is_current():
            callback, self.on_complete = self.on_complete, None
            callback()

    @Slot(object)
    def result(self, outcome: OcrOutcome) -> None:
        window = self.window
        if not isinstance(outcome, OcrOutcome):
            window._on_text_extraction_error("Invalid queued OCR result")
            return
        key = outcome.target_type, outcome.clip_id
        if not outcome.can_apply or not self._is_current() or key in self.delivered:
            return
        self.delivered.add(key)
        try:
            outcome = OcrOutcome.from_dict(asdict(outcome))
            receipt = None
            cache = getattr(self.worker, "cache", None)
            if cache is not None:
                if (
                    window.project.path is None
                    or window.project.path.resolve() != cache.path
                ):
                    raise ValueError("Project save location changed during OCR")
                receipt = cache.results.get(key)
                if (receipt is not None and not receipt.matches(outcome)) or (
                    receipt is None
                    and cache.transient_outcomes.get(key) != asdict(outcome)
                ):
                    raise ValueError("Queued OCR differs from its recorded result")
            accepted = self.application.apply(window.project, outcome)
            if accepted and receipt is not None:
                window.project.record_job_result(receipt.result_id, receipt.digest)
        except Exception as exc:
            window._on_text_extraction_error(f"Could not apply OCR: {exc}")
            return
        if not accepted:
            window._on_text_extraction_error(
                "OCR discarded because the target changed. Run analysis again."
            )
        elif outcome.has_result and outcome.target_type == "clip":
            window.analyze_tab.update_clip_extracted_text(
                outcome.clip_id, outcome.to_models()
            )

"""Qt adapter for shared OCR computation on detached clip/frame inputs."""

from typing import Optional

from PySide6.QtCore import Signal

from core.operations.ocr import OcrTask, OcrOptions, OcrOutcome, run_ocr
from ui.workers.base import CancellableWorker, summarize_clip_errors


def _summarize_errors(errors: list[tuple[str, str]]) -> str:
    return summarize_clip_errors(errors, operation_label="Text extraction")


class TextExtractionWorker(CancellableWorker):
    """Preserve legacy signals while publishing typed outcomes to the owner."""

    progress = Signal(int, int, str)
    clip_completed = Signal(str, list)
    extraction_completed = Signal(dict)
    outcome_ready = Signal(object)

    def __init__(
        self,
        clips: list,
        sources_by_id: dict,
        num_keyframes: int = 3,
        use_vlm_fallback: bool = True,
        vlm_model: Optional[str] = None,
        vlm_only: bool = False,
        use_text_detection: bool = True,
        analysis_targets: Optional[list] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.options = OcrOptions(
            min(max(1, num_keyframes), 5),
            use_vlm_fallback,
            vlm_model,
            vlm_only,
            use_text_detection,
        )
        if analysis_targets:
            self.tasks = tuple(
                OcrTask(
                    target.id,
                    target.video_path
                    if getattr(target, "target_type", "frame") == "clip"
                    else target.image_path,
                    getattr(target, "target_type", "frame"),
                    getattr(target, "start_frame", None) or 0,
                    getattr(target, "end_frame", None) or 0,
                    getattr(target, "fps", None) or 0.0,
                )
                for target in analysis_targets
            )
        else:
            self.tasks = tuple(
                OcrTask.from_clip(clip, sources_by_id.get(clip.source_id))
                for clip in clips
            )
        self.result: tuple[OcrOutcome, ...] = ()

    def run(self) -> None:
        self._log_start()
        results = {}
        errors: list[tuple[str, str]] = []

        def deliver(outcome: OcrOutcome) -> None:
            self.outcome_ready.emit(outcome)
            if outcome.status == "succeeded":
                results[outcome.clip_id] = outcome.to_models()
                self.clip_completed.emit(outcome.clip_id, outcome.to_models())
            elif outcome.status == "failed":
                errors.append(
                    (outcome.clip_id, outcome.message or outcome.code or "OCR failed")
                )
                results[outcome.clip_id] = []

        self.result = run_ocr(
            self.tasks,
            self.options,
            cancel_event=self._cancel_event,
            on_outcome=deliver,
            progress=self.progress.emit,
        )
        if not self.is_cancelled():
            if errors:
                self.error.emit(_summarize_errors(errors))
            self.extraction_completed.emit(results)
            self._log_complete()
        else:
            self._log_cancelled()

"""Compatibility QThread for shared, detached gaze analysis."""

from typing import TYPE_CHECKING
from PySide6.QtCore import Signal
from core.operations.gaze import GazeOptions, GazeTask, GazeOutcome, run_gaze
from ui.workers.base import CancellableWorker, summarize_clip_errors

if TYPE_CHECKING:
    from models.clip import Clip, Source


class GazeAnalysisWorker(CancellableWorker):
    """Compute immutable gaze outcomes; model publication belongs to the owner."""

    progress = Signal(int, int)
    gaze_ready = Signal(str, float, float, str)
    detection_completed = Signal()

    def __init__(
        self,
        clips: list["Clip"],
        sources_by_id: dict[str, "Source"],
        sample_interval: float = 1.0,
        skip_existing: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.options = GazeOptions(sample_interval)
        self.tasks = tuple(
            GazeTask(
                c.id,
                c.source_id,
                sources_by_id[c.source_id].file_path
                if c.source_id in sources_by_id
                else None,
                c.start_frame,
                c.end_frame,
                sources_by_id[c.source_id].fps if c.source_id in sources_by_id else 0.0,
                skip=skip_existing and c.gaze_category is not None,
            )
            for c in clips
        )
        self.result: tuple[GazeOutcome, ...] = ()

    def run(self) -> None:
        self._log_start()
        errors = []
        try:
            self.progress.emit(0, len(self.tasks))

            def deliver(outcome: GazeOutcome) -> None:
                if outcome.status == "succeeded":
                    self.gaze_ready.emit(
                        outcome.clip_id, outcome.yaw, outcome.pitch, outcome.category
                    )
                elif outcome.status == "failed" and outcome.code != "no_gaze_detected":
                    errors.append(
                        (
                            outcome.clip_id,
                            outcome.message or outcome.code or "Gaze detection failed",
                        )
                    )

            self.result = run_gaze(
                self.tasks,
                self.options,
                cancel_event=self._cancel_event,
                on_outcome=deliver,
                progress=self.progress.emit,
            )
            if errors and not self.is_cancelled():
                self.error.emit(
                    next(
                        (
                            "Failed to load gaze detection model: "
                            + (o.message or "unavailable")
                            for o in self.result
                            if o.code == "model_load_failed"
                        ),
                        summarize_clip_errors(errors, operation_label="Gaze detection"),
                    )
                )
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.detection_completed.emit()
            self._log_complete()

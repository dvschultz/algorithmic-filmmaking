"""Compatibility QThread for shared, detached face analysis."""

from typing import TYPE_CHECKING
from PySide6.QtCore import Signal
from core.operations.faces import FaceOptions, FaceTask, FaceOutcome, run_faces
from ui.workers.base import CancellableWorker, summarize_clip_errors

if TYPE_CHECKING:
    from models.clip import Clip, Source


class FaceDetectionWorker(CancellableWorker):
    """Compute immutable face outcomes; model publication belongs to the owner."""

    progress = Signal(int, int)
    faces_ready = Signal(str, list)
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
        self.options = FaceOptions(sample_interval)
        self.tasks = tuple(
            FaceTask(
                c.id,
                c.source_id,
                sources_by_id[c.source_id].file_path
                if c.source_id in sources_by_id
                else None,
                c.start_frame,
                c.end_frame,
                sources_by_id[c.source_id].fps if c.source_id in sources_by_id else 0.0,
                skip=skip_existing and c.face_embeddings is not None,
            )
            for c in clips
        )
        self.result: tuple[FaceOutcome, ...] = ()

    def run(self) -> None:
        self._log_start()
        errors = []
        try:
            self.progress.emit(0, len(self.tasks))

            def deliver(outcome: FaceOutcome) -> None:
                if outcome.status == "succeeded":
                    self.faces_ready.emit(outcome.clip_id, outcome.face_dicts())
                elif outcome.status == "failed":
                    errors.append(
                        (
                            outcome.clip_id,
                            outcome.message or outcome.code or "Face detection failed",
                        )
                    )

            self.result = run_faces(
                self.tasks,
                self.options,
                cancel_event=self._cancel_event,
                on_outcome=deliver,
                progress=self.progress.emit,
            )
            if errors and not self.is_cancelled():
                self.error.emit(
                    summarize_clip_errors(errors, operation_label="Face detection")
                )
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.detection_completed.emit()
            self._log_complete()

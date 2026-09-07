"""Qt adapter for detached batch frame extraction."""

from pathlib import Path

from PySide6.QtCore import Signal

from core.operations.frame_extraction import (
    FrameExtractionTask,
    FrameExtractionOutcome,
    run_frame_extraction,
)
from models.clip import Clip, Source
from ui.workers.base import CancellableWorker


class FrameExtractionWorker(CancellableWorker):
    progress = Signal(int, int)
    frame_ready = Signal(str, str)
    extraction_completed = Signal(list)
    outcome_ready = Signal(object)

    def __init__(
        self,
        source: Source,
        clip: Clip | None,
        mode: str,
        interval: int,
        output_dir: Path,
        parent=None,
    ):
        super().__init__(parent)
        self.task = FrameExtractionTask.from_source(
            source, clip, mode, interval, output_dir
        )
        self.result: FrameExtractionOutcome | None = None

    def run(self) -> None:
        self._log_start()
        self.result = run_frame_extraction(
            self.task,
            cancel_event=self._cancel_event,
            progress=self.progress.emit,
        )
        self.outcome_ready.emit(self.result)
        frames = []
        if self.result.status == "succeeded" and not self.is_cancelled():
            frames = [frame.to_model(self.task) for frame in self.result.frames]
            for frame in frames:
                if frame.thumbnail_path is not None:
                    self.frame_ready.emit(frame.id, str(frame.thumbnail_path))
        elif self.result.status == "failed":
            self.error.emit(self.result.message or "Frame extraction failed")
        self.extraction_completed.emit(frames)
        self._log_complete()

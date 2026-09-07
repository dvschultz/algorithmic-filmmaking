"""Qt adapter for shared detached audio probing."""

from pathlib import Path
from PySide6.QtCore import Signal
from core.operations.audio_import import (
    AudioImportTask,
    AudioImportOutcome,
    run_audio_import,
)
from ui.workers.base import CancellableWorker


class AudioImportWorker(CancellableWorker):
    progress = Signal(int, int)
    audio_ready = Signal(object)
    outcome_ready = Signal(object)
    finished_signal = Signal()

    def __init__(self, file_path: Path, parent=None, *, session_id: str | None = None):
        super().__init__(parent)
        self.task = AudioImportTask.from_path(file_path)
        self.session_id = session_id
        self.result: AudioImportOutcome | None = None

    def run(self) -> None:
        self._log_start()
        try:
            self.result = run_audio_import(
                self.task, cancel_event=self._cancel_event, progress=self.progress.emit
            )
            self.outcome_ready.emit(self.result)
            if self.result.status == "succeeded" and not self.is_cancelled():
                self.audio_ready.emit(self.result.to_model(self.task))
                self._log_complete()
            elif self.result.status == "failed":
                self.error.emit(self.result.message or "Audio import failed")
        finally:
            self.finished_signal.emit()

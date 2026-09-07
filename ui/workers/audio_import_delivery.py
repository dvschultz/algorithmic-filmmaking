"""Publish audio imports only to the project that requested them."""

from typing import Any
from PySide6.QtCore import QObject, Slot
from core.operations.audio_import import AudioImportApplication, AudioImportOutcome


class AudioImportDelivery(QObject):
    def __init__(self, window: Any, worker: Any) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.application = AudioImportApplication(window.project, worker.task)
        worker.outcome_ready.connect(self.complete)
        worker.finished.connect(self.finished)

    @Slot(object)
    def complete(self, outcome: AudioImportOutcome) -> None:
        if (
            self.worker not in self.window._active_audio_imports
            or self.worker.is_cancelled()
            or not self.application.is_current(self.window.project)
            or self.application.consumed
        ):
            return
        if outcome.status == "failed":
            self.window._on_audio_import_error(outcome.message or "Audio import failed")
        elif outcome.status == "succeeded":
            try:
                added = self.application.apply(self.window.project, outcome)
            except Exception as exc:
                self.window._on_audio_import_error(str(exc))
                return
            if added:
                self.window._on_audio_imported(self.application.audio)
            elif self.application.audio is None:
                self.window._on_audio_import_error("Audio file changed during import")

    @Slot()
    def finished(self) -> None:
        self.window._active_audio_imports.discard(self.worker)
        self.worker.deleteLater()
        self.deleteLater()

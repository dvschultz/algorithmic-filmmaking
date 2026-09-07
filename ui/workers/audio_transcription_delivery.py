"""Queue standalone audio results to their original project owner."""

from typing import Any

from PySide6.QtCore import QObject, Slot

from core.operations.audio_transcription import (
    AudioTranscriptionApplication,
    AudioTranscriptionOutcome,
)


class AudioTranscriptionDelivery(QObject):
    def __init__(self, window: Any, worker: Any) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.application = AudioTranscriptionApplication(window.project, worker.task)
        worker.transcript_ready.connect(self.transcript)
        worker.error.connect(self.error)
        # Keep the QThread alive through native completion, not a domain signal.
        worker.finished.connect(self.finished)

    def current(self) -> bool:
        return (
            self.worker in self.window._active_audio_transcribes
            and not self.worker.is_cancelled()
            and self.application.is_current(self.window.project)
        )

    @Slot(str, list)
    def transcript(self, audio_source_id: str, segments: list) -> None:
        if not self.current() or self.application.consumed:
            return
        try:
            outcome = AudioTranscriptionOutcome(
                audio_source_id, "succeeded", tuple(segments)
            )
            cache = getattr(self.worker, "cache", None)
            receipt = cache.results[audio_source_id] if cache is not None else None
            if receipt is not None and not receipt.matches(outcome):
                raise ValueError(
                    "Queued audio transcript differs from its recorded result"
                )
            applied = self.application.apply(
                self.window.project,
                outcome,
            )
            if applied and receipt is not None:
                self.window.project.record_job_result(receipt.result_id, receipt.digest)
        except Exception as exc:
            self.error(f"Could not apply audio transcription: {exc}")
            return
        if applied:
            self.window._on_audio_transcript_ready(audio_source_id, segments)
        else:
            self.error("Audio changed during transcription. Run transcription again.")

    @Slot(str)
    def error(self, message: str) -> None:
        if self.current():
            self.window._on_audio_transcribe_error(message)

    @Slot()
    def finished(self) -> None:
        self.window._active_audio_transcribes.discard(self.worker)
        self.deleteLater()

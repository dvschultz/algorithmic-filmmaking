"""Qt adapter for shared standalone-audio transcription."""

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal

from core.operations.audio_transcription import (
    AudioTranscriptionTask,
    run_audio_transcription,
)
from core.operations.transcription import TranscriptionOptions
from models.audio_source import AudioSource
from ui.workers.base import CancellableWorker

if TYPE_CHECKING:
    from core.project import Project


class AudioTranscribeWorker(CancellableWorker):
    """Compute from detached inputs; callers publish on the project owner thread."""

    progress = Signal(int, int)
    transcript_ready = Signal(str, list)
    finished_signal = Signal()

    def __init__(
        self,
        audio_source: AudioSource,
        model_name: str = "small.en",
        language: str = "en",
        backend: str = "auto",
        segmentation_mode: str = "backend",
        segment_max_seconds: float = 12.0,
        parent=None,
        *,
        project: "Project | None" = None,
    ) -> None:
        super().__init__(parent)
        if project is not None:
            project.session.assert_owner()
        self.session_id = project.session.session_id if project is not None else None
        self.task = AudioTranscriptionTask.from_audio(audio_source)
        self.options = TranscriptionOptions(
            model=model_name,
            language=language,
            backend=backend,
            segmentation_mode=segmentation_mode,
            segment_max_seconds=segment_max_seconds,
        )

    def run(self) -> None:
        self._log_start()
        try:
            outcome = run_audio_transcription(
                self.task,
                self.options,
                cancel_event=self._cancel_event,
                progress=self.progress.emit,
            )
            if self.is_cancelled():
                self._log_cancelled()
            elif outcome.status == "succeeded":
                self.transcript_ready.emit(
                    outcome.audio_source_id, list(outcome.segments)
                )
                self._log_complete()
            elif outcome.status == "failed":
                self.error.emit(outcome.message or "Audio transcription failed")
        finally:
            self.finished_signal.emit()

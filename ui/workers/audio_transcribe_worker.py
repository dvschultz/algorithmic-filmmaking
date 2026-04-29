"""Worker that transcribes an AudioSource and emits the segments."""

import logging
from pathlib import Path

from PySide6.QtCore import Signal

from models.audio_source import AudioSource
from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)


class AudioTranscribeWorker(CancellableWorker):
    """Run Whisper transcription on a single AudioSource.

    Reuses core.transcription.transcribe_video — it accepts any file with
    an audio stream, including standalone .mp3/.wav files. The worker does
    not mutate the AudioSource directly; the caller wires `transcript_ready`
    to assign segments on the main thread.

    Signals:
        progress: (current, total) during processing
        transcript_ready: (audio_source_id, segments) on success
        finished_signal: emitted exactly once on completion or error
    """

    progress = Signal(int, int)
    transcript_ready = Signal(str, list)  # audio_source_id, list[TranscriptSegment]
    finished_signal = Signal()

    def __init__(
        self,
        audio_source: AudioSource,
        model_name: str = "small.en",
        language: str = "en",
        backend: str = "auto",
        parent=None,
    ):
        super().__init__(parent)
        self._audio_source = audio_source
        self._model_name = model_name
        self._language = language
        self._backend = backend

    def run(self) -> None:
        self._log_start()
        try:
            if not self._audio_source.file_path.exists():
                self.error.emit(
                    f"Audio file is missing on disk: {self._audio_source.file_path.name}"
                )
                return

            from core.transcription import transcribe_video

            def progress_cb(fraction: float, _message: str) -> None:
                if self.is_cancelled():
                    return
                # Coarse progress: convert 0..1 fraction to (n, 100)
                self.progress.emit(int(fraction * 100), 100)

            self.progress.emit(0, 100)

            try:
                segments = transcribe_video(
                    self._audio_source.file_path,
                    model_name=self._model_name,
                    language=self._language,
                    backend=self._backend,
                    progress_callback=progress_cb,
                )
            except Exception as exc:
                logger.exception("Audio transcription failed")
                self.error.emit(f"Transcription failed: {exc}")
                return

            if self.is_cancelled():
                self._log_cancelled()
                return

            self.progress.emit(100, 100)
            self.transcript_ready.emit(self._audio_source.id, segments)
            self._log_complete()
        finally:
            self.finished_signal.emit()

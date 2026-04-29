"""Worker that imports an audio file and emits an AudioSource."""

import logging
from pathlib import Path

from PySide6.QtCore import Signal

from core.ffmpeg import FFmpegProcessor
from models.audio_source import AudioSource
from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)


class AudioImportWorker(CancellableWorker):
    """Probe an audio file and emit an AudioSource for the project to add.

    The worker does not mutate the project — the caller is responsible
    for calling project.add_audio_source(audio) on the main thread when
    audio_ready fires. This mirrors how video import is structured.
    """

    progress = Signal(int, int)  # current, total
    audio_ready = Signal(object)  # AudioSource
    finished_signal = Signal()  # avoid clashing with QThread.finished

    def __init__(self, file_path: Path, parent=None):
        super().__init__(parent)
        self._file_path = Path(file_path)

    def run(self) -> None:
        self._log_start()
        try:
            if not self._file_path.exists():
                self.error.emit(f"File not found: {self._file_path}")
                return

            if self.is_cancelled():
                self._log_cancelled()
                return

            self.progress.emit(0, 1)

            try:
                processor = FFmpegProcessor()
            except RuntimeError as exc:
                self.error.emit(f"FFmpeg unavailable: {exc}")
                return

            if not processor.ffprobe_available:
                self.error.emit("FFprobe is not available — cannot probe audio file")
                return

            try:
                info = processor.get_audio_info(self._file_path)
            except ValueError as exc:
                # No audio stream
                self.error.emit(f"Not an audio file: {exc}")
                return
            except RuntimeError as exc:
                self.error.emit(f"Failed to probe audio: {exc}")
                return

            if self.is_cancelled():
                self._log_cancelled()
                return

            duration = info.get("duration", 0.0)
            if duration <= 0:
                self.error.emit(
                    f"Audio file has zero duration; refusing to import: {self._file_path.name}"
                )
                return

            audio = AudioSource(
                file_path=self._file_path,
                duration_seconds=duration,
                sample_rate=info.get("sample_rate", 0),
                channels=info.get("channels", 0),
            )

            self.progress.emit(1, 1)
            self.audio_ready.emit(audio)
            self._log_complete()
        except Exception as exc:  # noqa: BLE001 — defensive top-level
            logger.exception("AudioImportWorker failed unexpectedly")
            self.error.emit(f"Audio import failed: {exc}")
        finally:
            self.finished_signal.emit()

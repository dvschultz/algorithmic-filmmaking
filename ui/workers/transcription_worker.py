"""Background worker for clip transcription.

Runs Whisper transcription on multiple clips in a background thread,
using ThreadPoolExecutor for parallelism. Supports faster-whisper and
mlx-whisper backends.
"""

import logging
import time
from pathlib import Path

from PySide6.QtCore import Signal

from ui.workers.base import CancellableWorker, summarize_clip_errors
from core.operations.transcription import (
    TranscriptionOptions, TranscriptionOutcome, TranscriptionTask, run_transcription, snapshot_tasks,
)

logger = logging.getLogger(__name__)


def _summarize_errors(errors: list[tuple[str, str]]) -> str:
    """Return a compact user-facing summary for transcription failures."""
    return summarize_clip_errors(errors, operation_label="Transcription")


class TranscriptionWorker(CancellableWorker):
    """Background worker for transcribing clips using faster-whisper.

    Uses ThreadPoolExecutor for parallel processing. faster-whisper and cloud
    transcription can run concurrently, but local MLX transcription is forced
    to serial execution because model initialization/inference is not thread-safe.

    Signals:
        progress: Emitted with (current, total) during processing
        transcript_ready: Emitted with (clip_id, segments) when a clip finishes
        transcription_completed: Emitted when all clips are processed
        error: Emitted with error message string on failure (inherited)
    """

    progress = Signal(int, int)  # current, total
    status = Signal(str)
    transcript_ready = Signal(str, list)  # clip_id, segments
    transcription_completed = Signal()

    def __init__(
        self,
        clips: list,
        source,
        model_name: str = "small.en",
        language: str = "en",
        parallelism: int = 2,
        skip_existing: bool = True,
        backend: str = "auto",
        model_cache_dir: Path | None = None,
        min_free_disk_gb: float = 3.0,
        segmentation_mode: str = "backend",
        segment_max_seconds: float = 12.0,
        parent=None,
    ):
        super().__init__(parent)
        self._model_name = model_name
        self._language = language
        self._backend = self._resolve_backend(backend)
        self._requested_backend = backend
        self._model_cache_dir = model_cache_dir
        self._min_free_disk_gb = min_free_disk_gb
        self._segmentation_mode = segmentation_mode
        self._segment_max_seconds = segment_max_seconds
        requested_parallelism = min(max(1, parallelism), 4)
        self._parallelism = 1 if self._backend == "mlx-whisper" else requested_parallelism
        self._tasks = tuple(task for task in snapshot_tasks(
            clips, {source.id: source}, skip_existing=skip_existing,
        ) if not task.skip)
        self._options = TranscriptionOptions(
            model_name, language, self._backend, segmentation_mode,
            segment_max_seconds, self._parallelism,
        )

    @property
    def tasks(self) -> tuple[TranscriptionTask, ...]:
        """The exact immutable inputs submitted by this worker."""
        return self._tasks

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        """Resolve auto backend selection once at worker startup."""
        from core.transcription import _resolve_backend

        return _resolve_backend(backend)

    def run(self):
        """Execute transcription on all clips."""
        started_at = time.monotonic()
        self._log_start()

        total = len(self._tasks)
        if total == 0:
            logger.info("No clips to process for transcription")
            self.transcription_completed.emit()
            self._log_complete()
            return

        from core.binary_resolver import find_binary
        from core.transcription import FFmpegNotFoundError

        if find_binary("ffmpeg") is None:
            message = str(FFmpegNotFoundError())
            self._log_error(message)
            self.error.emit(message)
            self.transcription_completed.emit()
            self._log_complete()
            return

        logger.info(
            f"Starting transcription: {total} clips, "
            f"backend={self._backend}, parallelism={self._parallelism}"
        )

        if self._backend != "groq":
            try:
                from core.settings import load_settings
                from core.transcription_storage import validate_transcription_disk_space

                cache_dir = self._model_cache_dir or load_settings().model_cache_dir
                self.status.emit("Transcribe: checking disk space for Whisper model and temp audio...")
                validate_transcription_disk_space(
                    self._model_name,
                    self._backend,
                    cache_dir,
                    self._min_free_disk_gb,
                )
            except Exception as e:
                self.error.emit(str(e))
                self.transcription_completed.emit()
                self._log_complete()
                return

        # Pre-load Whisper model so user sees download status
        if self._backend != "groq":
            try:
                from core.transcription import (
                    WHISPER_MODELS,
                    get_model,
                    get_mlx_model,
                    is_mlx_whisper_available,
                )

                self.progress.emit(0, total)
                model_info = WHISPER_MODELS.get(self._model_name, {})
                model_size = model_info.get("size", "unknown size")
                self.status.emit(
                    f"Transcribe: loading {self._backend} model "
                    f"{self._model_name} ({model_size}); first run may download it..."
                )
                if self._backend in ("auto", "mlx-whisper") and is_mlx_whisper_available():
                    get_mlx_model(self._model_name)
                else:
                    get_model(self._model_name)
                self.status.emit(f"Transcribe: model ready; processing {total} clips...")
            except Exception as e:
                self.error.emit(f"Failed to load Whisper model: {e}")
                self.transcription_completed.emit()
                self._log_complete()
                return

            if self.is_cancelled():
                self._log_cancelled()
                self.transcription_completed.emit()
                return

        errors: list[tuple[str, str]] = []

        def deliver(outcome: TranscriptionOutcome) -> None:
            if outcome.status == "succeeded":
                self.transcript_ready.emit(outcome.clip_id, list(outcome.segments))
            elif outcome.status == "failed":
                errors.append((outcome.clip_id, outcome.message or outcome.code or "Transcription failed"))

        run_transcription(
            self._tasks, self._options, cancel_event=self._cancel_event,
            on_outcome=deliver, progress=self.progress.emit,
        )

        if errors:
            self.error.emit(_summarize_errors(errors))

        elapsed = time.monotonic() - started_at
        self.status.emit(f"Transcription completed in {elapsed:.1f}s")
        self.transcription_completed.emit()
        self._log_complete()

"""Background worker for clip transcription.

Runs Whisper transcription on multiple clips in a background thread,
using ThreadPoolExecutor for parallelism. Supports faster-whisper and
mlx-whisper backends.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal

from ui.workers.base import CancellableWorker, summarize_clip_errors

logger = logging.getLogger(__name__)


def _summarize_errors(errors: list[tuple[str, str]]) -> str:
    """Return a compact user-facing summary for transcription failures."""
    return summarize_clip_errors(errors, operation_label="Transcription")


@dataclass(frozen=True)
class TranscriptionTask:
    """Immutable task data for thread pool execution."""

    clip_id: str
    source_path: Path
    start_time: float
    end_time: float
    fps: float


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
        self._tasks = self._build_tasks(clips, source, skip_existing)

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        """Resolve auto backend selection once at worker startup."""
        from core.transcription import _resolve_backend

        return _resolve_backend(backend)

    def _build_tasks(
        self, clips: list, source, skip_existing: bool
    ) -> list[TranscriptionTask]:
        """Build immutable task list from clips."""
        tasks = []
        for clip in clips:
            if skip_existing and clip.transcript is not None:
                continue

            tasks.append(
                TranscriptionTask(
                    clip_id=clip.id,
                    source_path=source.file_path,
                    start_time=clip.start_time(source.fps),
                    end_time=clip.end_time(source.fps),
                    fps=source.fps,
                )
            )
        return tasks

    def _process_task(
        self, task: TranscriptionTask
    ) -> tuple[str, Optional[list], Optional[str], bool]:
        """Process a single task (runs in thread pool).

        Returns:
            Tuple of (clip_id, segments, error_message, is_critical)
        """
        if self.is_cancelled():
            return task.clip_id, None, "Cancelled", False

        try:
            from core.transcription import (
                transcribe_clip,
                FFmpegNotFoundError,
                FasterWhisperNotInstalledError,
                ModelDownloadError,
            )

            segments = transcribe_clip(
                task.source_path,
                task.start_time,
                task.end_time,
                self._model_name,
                self._language,
                backend=self._backend,
                segmentation_mode=self._segmentation_mode,
                segment_max_seconds=self._segment_max_seconds,
            )
            return task.clip_id, segments, None, False
        except (FFmpegNotFoundError, FasterWhisperNotInstalledError, ModelDownloadError) as e:
            return task.clip_id, None, str(e), True  # Critical error
        except Exception as e:
            return task.clip_id, None, str(e), False

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
                return

        completed = 0
        errors: list[tuple[str, str]] = []

        with ThreadPoolExecutor(max_workers=self._parallelism) as executor:
            future_to_task = {
                executor.submit(self._process_task, task): task
                for task in self._tasks
            }

            for future in as_completed(future_to_task):
                if self.is_cancelled():
                    self._log_cancelled()
                    for f in future_to_task:
                        f.cancel()
                    break

                task = future_to_task[future]
                completed += 1

                try:
                    clip_id, segments, error_msg, is_critical = future.result()

                    if error_msg and error_msg != "Cancelled":
                        self._log_error(error_msg, clip_id)
                        errors.append((clip_id, error_msg))
                        if is_critical:
                            # Cancel all remaining work
                            for f in future_to_task:
                                f.cancel()
                            break
                    elif segments is not None:
                        self.transcript_ready.emit(clip_id, segments)
                except Exception as e:
                    self._log_error(str(e), task.clip_id)
                    errors.append((task.clip_id, str(e)))

                self.progress.emit(completed, total)

        if errors:
            self.error.emit(_summarize_errors(errors))

        elapsed = time.monotonic() - started_at
        self.status.emit(f"Transcription completed in {elapsed:.1f}s")
        self.transcription_completed.emit()
        self._log_complete()

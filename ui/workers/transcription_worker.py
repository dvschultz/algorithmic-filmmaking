"""Background worker for clip transcription.

Runs Whisper transcription on multiple clips in a background thread,
using ThreadPoolExecutor for parallelism. Supports faster-whisper and
mlx-whisper backends.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal

from ui.workers.base import CancellableWorker

logger = logging.getLogger(__name__)


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

    Uses ThreadPoolExecutor for parallel processing. Default parallelism is 2
    because each transcription spawns an FFmpeg subprocess + Whisper inference.

    Signals:
        progress: Emitted with (current, total) during processing
        transcript_ready: Emitted with (clip_id, segments) when a clip finishes
        transcription_completed: Emitted when all clips are processed
        error: Emitted with error message string on failure (inherited)
    """

    progress = Signal(int, int)  # current, total
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
        parent=None,
    ):
        super().__init__(parent)
        self._model_name = model_name
        self._language = language
        self._backend = backend
        self._parallelism = min(max(1, parallelism), 4)
        self._tasks = self._build_tasks(clips, source, skip_existing)

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
            )
            return task.clip_id, segments, None, False
        except (FasterWhisperNotInstalledError, ModelDownloadError) as e:
            return task.clip_id, None, str(e), True  # Critical error
        except Exception as e:
            return task.clip_id, None, str(e), False

    def run(self):
        """Execute transcription on all clips."""
        self._log_start()

        total = len(self._tasks)
        if total == 0:
            logger.info("No clips to process for transcription")
            self.transcription_completed.emit()
            self._log_complete()
            return

        logger.info(
            f"Starting transcription: {total} clips, "
            f"parallelism={self._parallelism}"
        )

        completed = 0

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
                        self.error.emit(error_msg)
                        if is_critical:
                            # Cancel all remaining work
                            for f in future_to_task:
                                f.cancel()
                            break
                    elif segments is not None:
                        self.transcript_ready.emit(clip_id, segments)
                except Exception as e:
                    self._log_error(str(e), task.clip_id)

                self.progress.emit(completed, total)

        self.transcription_completed.emit()
        self._log_complete()

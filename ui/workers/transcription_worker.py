"""Qt lifecycle adapter for shared session transcription jobs."""

import logging
import time
from dataclasses import asdict
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from core.jobs import JobRuntime
from core.jobs.transcription import transcription_operation_spec
from core.jobs.gui_transcription import GuiTranscriptionCache
from core.jobs.media import media_stamp
from core.transcription_models import TranscriptSegment

from ui.workers.base import CancellableWorker, summarize_clip_errors
from core.operations.transcription import (
    TranscriptionOptions,
    TranscriptionOutcome,
    TranscriptionTask,
    run_transcription,
    snapshot_tasks,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.project import Project


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
    job_started = Signal(str, str)

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
        *,
        project: "Project | None" = None,
    ) -> None:
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
        self._parallelism = (
            1 if self._backend == "mlx-whisper" else requested_parallelism
        )
        self._tasks = tuple(
            task
            for task in snapshot_tasks(
                clips,
                {source.id: source},
                skip_existing=skip_existing,
            )
            if not task.skip
        )
        self._options = TranscriptionOptions(
            model_name,
            language,
            self._backend,
            segmentation_mode,
            segment_max_seconds,
            self._parallelism,
        )
        self.operation = transcription_operation_spec(
            self._tasks,
            self._options,
            arguments={
                **asdict(self._options),
                "model_cache_dir": str(model_cache_dir) if model_cache_dir else None,
                "min_free_disk_gb": min_free_disk_gb,
            },
            persistence="session_only",
            session_id=project.session.session_id if project is not None else None,
            input_revision=str(project.mutation_generation)
            if project is not None
            else None,
        )
        self.task_id: str | None = None
        self.job_status: str | None = None
        self.result: tuple[TranscriptionOutcome, ...] = ()
        self._runtime: JobRuntime | None = None
        self.cache = (
            GuiTranscriptionCache(
                project.path,
                project.metadata.id,
                {clip.id: clip.source_id for clip in clips},
                project.metadata.job_results,
                options=self._options,
                previous_transcripts={
                    clip.id: [segment.to_dict() for segment in clip.transcript]
                    if clip.transcript is not None
                    else None
                    for clip in clips
                },
                media_stamps={
                    task.source_path: media_stamp(task.source_path)
                    for task in self._tasks
                    if task.source_path is not None
                },
            )
            if project is not None and project.path is not None
            else None
        )

    def cancel(self) -> None:
        super().cancel()
        runtime = self._runtime
        if runtime is not None and self.task_id is not None:
            runtime.cancel(self.task_id)

    @property
    def tasks(self) -> tuple[TranscriptionTask, ...]:
        """The exact immutable inputs submitted by this worker."""
        return self._tasks

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        """Resolve auto backend selection once at worker startup."""
        from core.transcription import _resolve_backend

        return _resolve_backend(backend)

    def _prepare(self, events: Queue[tuple[str, tuple]]) -> bool:
        """Run preflight inside the job, reporting through the lifecycle adapter."""
        total = len(self._tasks)
        if self.is_cancelled() or total == 0:
            return False

        from core.binary_resolver import find_binary
        from core.transcription import FFmpegNotFoundError

        if find_binary("ffmpeg") is None:
            raise FFmpegNotFoundError()

        logger.info(
            f"Starting transcription: {total} clips, "
            f"backend={self._backend}, parallelism={self._parallelism}"
        )

        if self._backend != "groq":
            from core.settings import load_settings
            from core.transcription_storage import validate_transcription_disk_space

            cache_dir = self._model_cache_dir or load_settings().model_cache_dir
            events.put(
                (
                    "status",
                    (
                        "Transcribe: checking disk space for Whisper model and temp audio...",
                    ),
                )
            )
            validate_transcription_disk_space(
                self._model_name,
                self._backend,
                cache_dir,
                self._min_free_disk_gb,
            )
            if self.is_cancelled():
                return False

        # Pre-load Whisper model so user sees download status
        if self._backend != "groq":
            try:
                from core.transcription import (
                    WHISPER_MODELS,
                    get_model,
                    get_mlx_model,
                    is_mlx_whisper_available,
                )

                events.put(("progress", (0, total)))
                model_info = WHISPER_MODELS.get(self._model_name, {})
                model_size = model_info.get("size", "unknown size")
                events.put(
                    (
                        "status",
                        (
                            f"Transcribe: loading {self._backend} model "
                            f"{self._model_name} ({model_size}); first run may download it...",
                        ),
                    )
                )
                if (
                    self._backend in ("auto", "mlx-whisper")
                    and is_mlx_whisper_available()
                ):
                    get_mlx_model(self._model_name)
                else:
                    get_model(self._model_name)
                events.put(
                    (
                        "status",
                        (f"Transcribe: model ready; processing {total} clips...",),
                    )
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load Whisper model: {e}") from e
        return not self.is_cancelled()

    def run(self) -> None:
        """Observe one session job and emit Qt signals from this adapter."""
        started_at = time.monotonic()
        self._log_start()
        runtime = None
        events: Queue[tuple[str, tuple]] = Queue()

        def emit_event(event: tuple[str, tuple]) -> None:
            kind, args = event
            if kind == "status":
                self.status.emit(*args)
            elif kind == "progress":
                self.progress.emit(*args)
            elif kind == "transcript":
                self.transcript_ready.emit(*args)

        def compute(progress, cancel):
            try:

                def deliver(outcome):
                    if outcome.status == "succeeded":
                        events.put(
                            ("transcript", (outcome.clip_id, list(outcome.segments)))
                        )

                def report(current, total):
                    progress(
                        current / total if total else 1.0,
                        f"Transcribing ({current}/{total})",
                    )
                    events.put(("progress", (current, total)))

                if self.cache is not None:
                    outcomes = self.cache.run(
                        self._tasks,
                        cancel,
                        lambda: self._prepare(events),
                        deliver,
                        report,
                    )
                elif not self._prepare(events):
                    outcomes = tuple(
                        TranscriptionOutcome(
                            task.clip_id, "unprocessed", code="cancelled"
                        )
                        for task in self._tasks
                    )
                else:
                    outcomes = run_transcription(
                        self._tasks,
                        self._options,
                        cancel_event=cancel,
                        on_outcome=deliver,
                        progress=report,
                    )
                return {"outcomes": [asdict(outcome) for outcome in outcomes]}
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        try:
            runtime = JobRuntime.for_session(max_workers=1)
            self._runtime = runtime
            submission = runtime.submit(
                kind="transcribe",
                args=self.operation.arguments,
                operation=self.operation,
                run=compute,
                cancellation_event=self._cancel_event,
            )
            self.task_id = submission["task_id"]
            self.job_started.emit(self.task_id, submission["persistence"])
            while runtime.is_handle_live(self.task_id):
                try:
                    emit_event(events.get(timeout=0.05))
                except Empty:
                    pass
            runtime.shutdown()
            while not events.empty():
                emit_event(events.get_nowait())
            row = runtime.store.get(self.task_id)
            self.job_status = row.status
            payload = row.result or {}
            self.result = tuple(
                TranscriptionOutcome(
                    **{
                        **item,
                        "segments": tuple(
                            TranscriptSegment.from_dict(s) for s in item["segments"]
                        ),
                    }
                )
                for item in payload.get("outcomes", [])
            )
            if row.status == "cancelled" and not self.result:
                self.result = tuple(
                    TranscriptionOutcome(task.clip_id, "unprocessed", code="cancelled")
                    for task in self._tasks
                )
            if row.status == "failed":
                raise RuntimeError(row.error or "Transcription failed")
            errors = [
                (o.clip_id, o.message or o.code or "Transcription failed")
                for o in self.result
                if o.status == "failed"
            ]
            if errors:
                self.error.emit(_summarize_errors(errors))
            if row.status == "cancelled":
                self._log_cancelled()
            else:
                self.status.emit(
                    f"Transcription completed in {time.monotonic() - started_at:.1f}s"
                )
        except Exception as exc:
            self._log_error(str(exc))
            self.error.emit(str(exc))
        finally:
            try:
                if runtime is not None:
                    runtime.close_session()
            finally:
                self._runtime = None
                self.transcription_completed.emit()
                self._log_complete()

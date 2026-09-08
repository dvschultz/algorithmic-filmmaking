"""Qt adapter for shared standalone-audio transcription."""

from typing import TYPE_CHECKING
from dataclasses import asdict
from queue import Empty, Queue

from PySide6.QtCore import Signal

from core.operations.audio_transcription import (
    AudioTranscriptionTask,
    AudioTranscriptionOutcome,
    run_audio_transcription,
)
from core.jobs import JobRuntime
from core.jobs.spec import OperationSpec, encode_object
from core.jobs import gui_audio_transcription as audio_jobs
from core.operations.transcription import TranscriptionOptions
from core.operations.transcription_records import transcription_parameters
from models.audio_source import AudioSource
from ui.workers.base import CancellableWorker
from ui.workers.job_adapter import (
    gui_job_operation,
    gui_job_runtime,
    close_gui_job_runtime,
)

if TYPE_CHECKING:
    from core.project import Project


class AudioTranscribeWorker(CancellableWorker):
    """Compute from detached inputs; callers publish on the project owner thread."""

    progress = Signal(int, int)
    transcript_ready = Signal(str, list)
    outcome_ready = Signal(object)
    finished_signal = Signal()
    job_started = Signal(str, str)

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
        self.task = AudioTranscriptionTask.from_audio(audio_source, verified=True)
        self.options = audio_jobs.resolve_audio_options(
            TranscriptionOptions(
                model=model_name,
                language=language,
                backend=backend,
                segmentation_mode=segmentation_mode,
                segment_max_seconds=segment_max_seconds,
            )
        )
        runtime_identity = audio_jobs.audio_transcription_runtime(self.task, self.options)
        self.runtime_json = encode_object(runtime_identity)
        self.operation = gui_job_operation(
            OperationSpec.build(
                kind="audio_transcribe",
                version=2,
                arguments=transcription_parameters(self.options),
                inputs={
                    "audio_source_id": self.task.audio_source_id,
                    "path": str(self.task.path),
                    "media_stamp": self.task.media_stamp,
                    "analysis_json": self.task.analysis_json,
                    "runtime": self.runtime_json,
                },
                persistence="session_only",
                session_id=self.session_id,
                input_revision=str(project.mutation_generation)
                if project is not None
                else None,
            ),
            project.path if project is not None else None,
        )
        self.cache = (
            audio_jobs.GuiAudioTranscriptionCache(
                project,
                self.task,
                self.options,
                runtime_identity,
            )
            if project is not None and project.path is not None
            else None
        )
        self.task_id: str | None = None
        self.job_status: str | None = None
        self.result: AudioTranscriptionOutcome | None = None
        self._runtime: JobRuntime | None = None

    def cancel(self) -> None:
        super().cancel()
        if self._runtime is not None and self.task_id is not None:
            self._runtime.cancel(self.task_id)

    def run(self) -> None:
        self._log_start()
        runtime = None
        events: Queue[tuple[int, int]] = Queue()

        def compute(progress, cancel):
            def report(current, total):
                progress(current / total if total else 1.0, "Transcribing audio")
                events.put((current, total))

            if (
                encode_object(audio_jobs.audio_transcription_runtime(self.task, self.options))
                != self.runtime_json
            ):
                raise ValueError("Audio transcription runtime changed while queued")
            if self.cache is not None:
                outcome = self.cache.run(self.task, cancel, report)
            else:
                outcome = run_audio_transcription(
                    self.task,
                    self.options,
                    cancel_event=cancel,
                    progress=report,
                )
            return {
                "success": outcome.status != "failed",
                "error": outcome.message,
                "outcome": asdict(outcome),
            }

        try:
            runtime = gui_job_runtime(self.operation)
            self._runtime = runtime
            submission = runtime.submit(
                kind=self.operation.kind,
                args=self.operation.arguments,
                operation=self.operation,
                run=compute,
                cancellation_event=self._cancel_event,
                project_path=self.operation.arguments.get("project_path"),
            )
            self.task_id = submission["task_id"]
            self.job_started.emit(self.task_id, runtime.store.persistence)
            while runtime.is_handle_live(self.task_id):
                try:
                    current, total = events.get(timeout=0.05)
                    if not self.is_cancelled():
                        self.progress.emit(current, total)
                except Empty:
                    pass
            runtime.shutdown()
            while not events.empty():
                current, total = events.get_nowait()
                if not self.is_cancelled():
                    self.progress.emit(current, total)
            row = runtime.store.get(self.task_id)
            self.job_status = row.status
            if row.status == "cancelled" or self.is_cancelled():
                self.result = AudioTranscriptionOutcome(
                    self.task.audio_source_id, "unprocessed"
                )
                self._log_cancelled()
                return
            payload = (row.result or {}).get("outcome")
            if payload is None:
                raise RuntimeError(
                    row.error or "Audio transcription produced no result"
                )
            self.result = AudioTranscriptionOutcome.from_dict(payload)
            if self.result.can_apply:
                self.outcome_ready.emit(self.result)
            if self.result.has_result:
                self.transcript_ready.emit(
                    self.result.audio_source_id, list(self.result.segments)
                )
                self._log_complete()
            elif self.result.status == "failed":
                self.error.emit(self.result.message or "Audio transcription failed")
        except Exception as exc:
            self.result = AudioTranscriptionOutcome(
                self.task.audio_source_id, "failed", message=str(exc)
            )
            if not self.is_cancelled():
                self.error.emit(str(exc))
        finally:
            try:
                if runtime is not None:
                    close_gui_job_runtime(runtime)
            finally:
                self._runtime = None
                self.finished_signal.emit()

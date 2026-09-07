"""Owner-thread transcription signals scoped to their worker and request."""

from typing import Any

from PySide6.QtCore import QObject, Slot

from core.operations.transcription import TranscriptionApplication, TranscriptionOutcome
from ui.workers.analysis_pipeline_delivery import pipeline_can_continue
from ui.workers.gui_tool_reply import GuiToolReply


class TranscriptionDelivery(QObject):
    def __init__(
        self,
        window: Any,
        worker: Any,
        *,
        reply: GuiToolReply | None = None,
        pipeline: bool = False,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.application = TranscriptionApplication(window.project, worker.tasks)
        self.reply = reply
        self._delivered_targets: set[str] = set()
        self.pipeline = pipeline
        self.run = getattr(window, "_analysis_run", None) if pipeline else None
        worker.finished.connect(self.deleteLater)
        worker.progress.connect(self.progress)
        worker.status.connect(self.status)
        worker.transcript_ready.connect(self.transcript)
        worker.error.connect(self.error)
        worker.job_started.connect(self.job_started)

    def _current(self) -> bool:
        return (
            self.window.transcription_worker is self.worker
            and self.window.project is self.application.project
            and self.window.project.session.session_id == self.application.session_id
            and (self.reply is None or self.reply.is_current(self.window))
            and (
                not self.pipeline
                or (
                    getattr(self.window, "_analysis_run", None) is self.run
                    and pipeline_can_continue(self.window)
                )
            )
        )

    @Slot(int, int)
    def progress(self, current: int, total: int) -> None:
        if self._current():
            self.window._on_transcription_progress(current, total)

    @Slot(str, str)
    def job_started(self, task_id: str, persistence: str) -> None:
        if self._current() and persistence == "session_only":
            self.window.status_bar.showMessage(
                "Transcription results remain unsaved until you save the project."
            )

    @Slot(str)
    def status(self, message: str) -> None:
        if self._current():
            self.window.status_bar.showMessage(message)

    @Slot(str)
    def error(self, message: str) -> None:
        if self._current():
            self.window._on_transcription_error(message)

    @Slot(str, list)
    def transcript(self, clip_id: str, segments: list) -> None:
        if not self._current() or clip_id in self._delivered_targets:
            return
        self._delivered_targets.add(clip_id)
        try:
            cache = getattr(self.worker, "cache", None)
            project = self.window.project
            if cache is not None and (
                project.path is None or project.path.resolve() != cache.path
            ):
                raise ValueError("Project save location changed during transcription")
            receipt = cache.results[clip_id] if cache is not None else None
            outcome = TranscriptionOutcome(clip_id, "succeeded", tuple(segments))
            if receipt is not None and not receipt.matches(outcome):
                raise ValueError("Queued transcript differs from its recorded result")
            applied = self.application.apply(
                project,
                outcome,
            )
            if applied and receipt is not None:
                project.record_job_result(receipt.result_id, receipt.digest)
        except Exception as exc:
            self.error(f"Could not apply transcription: {exc}")
            return
        if applied and self._current():
            self.window._on_transcript_ready(clip_id, segments)
        elif not applied:
            self.error(
                f"Transcription result discarded for changed clip {clip_id}. Run transcription again."
            )

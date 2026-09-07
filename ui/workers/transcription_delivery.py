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
            applied = self.application.apply(
                self.window.project,
                TranscriptionOutcome(clip_id, "succeeded", tuple(segments)),
            )
        except Exception as exc:
            self.error(f"Could not apply transcription: {exc}")
            return
        if applied and self._current():
            self.window._on_transcript_ready(clip_id, segments)
        elif not applied:
            self.error(f"Transcription result discarded for changed clip {clip_id}. Run transcription again.")

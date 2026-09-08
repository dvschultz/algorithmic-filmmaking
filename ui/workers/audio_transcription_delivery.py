"""Queue standalone audio results to their original project owner."""

from typing import Any
from dataclasses import asdict

from PySide6.QtCore import Slot

from ui.workers.qt_lifetime import RetiringQObject

from core.operations.audio_transcription import (
    AudioTranscriptionApplication,
    AudioTranscriptionOutcome,
)


class AudioTranscriptionDelivery(RetiringQObject):
    def __init__(self, window: Any, worker: Any) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.application = AudioTranscriptionApplication(
            window.project, worker.task, getattr(worker, "options", None)
        )
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        self.applied = False
        self.failure: str | None = None
        if hasattr(worker, "outcome_ready"):
            worker.outcome_ready.connect(self.receive)
        else:
            worker.transcript_ready.connect(self.transcript)
        worker.error.connect(self.error)
        # Keep the QThread alive through native completion, not a domain signal.
        worker.finished.connect(self.finished)

    def current(self) -> bool:
        return (
            self.worker in self.window._active_audio_transcribes
            and not self.worker.is_cancelled()
            and self.application.is_current(self.window.project)
            and (self.reply is None or self.reply.is_current(self.window))
        )

    @Slot(str, list)
    def transcript(self, audio_source_id: str, segments: list) -> None:
        self.receive(AudioTranscriptionOutcome(audio_source_id, "succeeded", tuple(segments)))

    @Slot(object)
    def receive(self, outcome: AudioTranscriptionOutcome) -> None:
        if not isinstance(outcome, AudioTranscriptionOutcome) or not outcome.can_apply:
            return
        if not self.current() or self.application.consumed:
            return
        audio_source_id = outcome.audio_source_id
        try:
            cache = getattr(self.worker, "cache", None)
            project = self.window.project
            if cache is not None and (
                project.path is None or project.path.resolve() != cache.path
            ):
                raise ValueError("Project save location changed during transcription")
            receipt = cache.results.get(audio_source_id) if cache is not None else None
            if cache is not None and (
                (receipt is not None and not receipt.matches(outcome))
                or (receipt is None and cache.transient_outcomes.get(audio_source_id) != asdict(outcome))
            ):
                raise ValueError(
                    "Queued audio transcript differs from its recorded result"
                )
            applied = self.application.apply(
                self.window.project,
                outcome,
            )
            if applied and receipt is not None:
                self.window.project.record_job_result(receipt.result_id, receipt.digest)
        except Exception as exc:
            self.error(f"Could not apply audio transcription: {exc}")
            return
        if applied and outcome.has_result:
            self.applied = True
            self.window._on_audio_transcript_ready(audio_source_id, list(outcome.segments))
        elif not applied:
            self.error("Audio changed during transcription. Run transcription again.")

    @Slot(str)
    def error(self, message: str) -> None:
        if self.current():
            self.failure = message
            self.window._on_audio_transcribe_error(message)

    @Slot()
    def finished(self) -> None:
        if (
            self.reply is not None
            and self.worker in self.window._active_audio_transcribes
            and self.application.is_current(self.window.project)
        ):
            result: dict = {"success": self.applied}
            if self.applied:
                result["result"] = {
                    "audio_source_id": self.worker.task.audio_source_id,
                    "status": "succeeded",
                    "segment_count": len(self.worker.result.segments),
                }
            else:
                result["error"] = (
                    self.failure or "Audio transcription cancelled or discarded"
                )
            self.reply.send(self.window, result)
        self.window._active_audio_transcribes.discard(self.worker)
        self.retire()

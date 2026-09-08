"""Publish audio imports only to the project that requested them."""

from typing import Any
from PySide6.QtCore import Slot

from ui.workers.qt_lifetime import RetiringQObject
from core.operations.audio_import import (
    AudioImportApplication,
    AudioImportOutcome,
    AudioImportTask,
)
from core.jobs.audio_import import AudioImportRecord


class AudioImportDelivery(RetiringQObject):
    def __init__(self, window: Any, worker: Any) -> None:
        super().__init__(window)
        self.window = window
        self.worker = worker
        self.application = AudioImportApplication(window.project, worker.task)
        self.reply = getattr(window, "_dispatch_gui_reply", None)
        worker.gui_tool_reply = self.reply
        self.failure: str | None = None
        worker.outcome_ready.connect(self.complete)
        worker.finished.connect(self.finished)

    @Slot(object)
    def complete(self, outcome: AudioImportOutcome) -> None:
        if (
            self.worker not in self.window._active_audio_imports
            or self.worker.is_cancelled()
            or not self.application.is_current(self.window.project)
            or self.application.consumed
            or (self.reply is not None and not self.reply.is_current(self.window))
        ):
            return
        if outcome.status == "failed":
            self.failure = outcome.message or "Audio import failed"
            self.window._on_audio_import_error(self.failure)
        elif outcome.status == "succeeded":
            try:
                cache = getattr(self.worker, "cache", None)
                receipt = None
                recovered_task = None
                if cache is not None:
                    recorded = cache.recorded
                    if recorded is None:
                        raise ValueError("Audio import result has no durable receipt")
                    recovered_task = AudioImportTask.from_dict(recorded.task)
                    receipt = cache.results[str(self.worker.task.path)]
                    if not receipt.matches(
                        AudioImportRecord.build(recovered_task, outcome)
                    ):
                        raise ValueError(
                            "Queued audio import differs from its recorded result"
                        )
                added = self.application.apply(
                    self.window.project, outcome, recovered_task=recovered_task
                )
                if added and receipt is not None:
                    self.window.project.record_job_result(
                        receipt.result_id, receipt.digest
                    )
            except Exception as exc:
                self.failure = str(exc)
                self.window._on_audio_import_error(self.failure)
                return
            if added:
                self.window._on_audio_imported(self.application.audio)
            elif self.application.audio is None:
                self.failure = "Audio file changed during import"
                self.window._on_audio_import_error(self.failure)

    @Slot()
    def finished(self) -> None:
        if self.reply is not None and self.worker in self.window._active_audio_imports:
            audio = self.application.audio
            success = (
                self.failure is None
                and audio is not None
                and self.application.is_current(self.window.project)
            )
            result: dict = {"success": success}
            if success and audio is not None:
                result["result"] = {
                    "success": True,
                    "audio_source_id": audio.id,
                    "filename": audio.filename,
                    "duration": audio.duration_seconds,
                }
            else:
                result["error"] = self.failure or "Audio import cancelled or discarded"
            self.reply.send(self.window, result)
        self.window._active_audio_imports.discard(self.worker)
        self.worker.deleteLater()
        self.retire()

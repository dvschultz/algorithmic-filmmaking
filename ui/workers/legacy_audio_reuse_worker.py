"""Detached audio reuse decisions delivered through the normal audio owner guard."""

from PySide6.QtCore import Signal

from core.operations.audio_transcription import AudioTranscriptionTask, AudioTranscriptionOutcome
from core.operations.legacy_reuse import accept_legacy_audio_transcript, legacy_transcription_options
from core.project import Project
from core.settings import Settings
from models.analysis_record import AnalysisRecord
from ui.workers.base import CancellableWorker


class LegacyAudioReuseWorker(CancellableWorker):
    outcome_ready = Signal(object)
    is_legacy_reuse = True

    def __init__(self, project: Project, audio_source_id: str, *, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        project.session.assert_owner()
        audio = project.get_audio_source(audio_source_id)
        if audio is None:
            raise ValueError("Audio source not found")
        record = audio.analysis_records.get("transcribe")
        if record is not None and not isinstance(record, AnalysisRecord):
            raise ValueError("Unknown analysis record must be preserved; recompute analysis")
        self.session_id = project.session.session_id
        self.task = AudioTranscriptionTask.from_audio(audio, verified=True)
        self.options = legacy_transcription_options(settings)
        self.result: AudioTranscriptionOutcome | None = None

    def run(self) -> None:
        try:
            self.result = accept_legacy_audio_transcript(self.task, self.options, cancel_event=self._cancel_event)
            if self.result.has_result:
                self.outcome_ready.emit(self.result)
            elif self.result.status == "failed":
                self.error.emit(self.result.message or "Could not accept legacy transcript")
        except Exception as exc:
            self.error.emit(str(exc))

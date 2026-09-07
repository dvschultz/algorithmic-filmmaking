"""Record standalone-audio inference before owner delivery and explicit save."""

from dataclasses import asdict
import json
from threading import Event
from typing import Callable

from core.jobs.commits import StaleJobResult, canonical_json
from core.jobs.gui_results import GuiResultJournal, GuiResultRequest
from core.jobs.media import FingerprintCancelled
from core.jobs.audio_transcription import (
    audio_transcription_runtime,
    resolve_audio_options as resolve_audio_options,
)
from core.operations.audio_transcription import (
    AudioTranscriptionTask,
    AudioTranscriptionOutcome,
    run_audio_transcription,
)
from core.operations.transcription import TranscriptionOptions
from core.project import Project


class GuiAudioTranscriptionCache(GuiResultJournal):
    def __init__(
        self,
        project: Project,
        task: AudioTranscriptionTask,
        options: TranscriptionOptions,
        runtime: dict,
    ) -> None:
        project.session.assert_owner()
        if project.path is None:
            raise ValueError("Audio recovery requires a saved project")
        audio = project.get_audio_source(task.audio_source_id)
        if audio is None:
            raise ValueError("Audio source no longer exists")
        super().__init__(
            project.path,
            project.metadata.id,
            {task.audio_source_id: task.audio_source_id},
            project.metadata.job_results,
            kind="gui_audio_transcribe",
            arguments=asdict(options),
            media_stamps={task.path: task.media_stamp},
            target_id_field="audio_source_id",
        )
        self.options = options
        self.runtime_json = canonical_json(runtime)
        self.inputs_json = canonical_json(
            {
                "audio": audio.to_dict(),
                "runtime": runtime,
            }
        )

    def validate_media(self, request: GuiResultRequest) -> None:
        super().validate_media(request)
        if canonical_json(audio_transcription_runtime()) != self.runtime_json:
            raise StaleJobResult("Audio transcription runtime changed")

    def run(
        self,
        task: AudioTranscriptionTask,
        cancel: Event,
        progress: Callable[[int, int], None],
    ) -> AudioTranscriptionOutcome:
        try:
            if cancel.is_set():
                return AudioTranscriptionOutcome(task.audio_source_id, "unprocessed")
            self.start(cancel)
            request, payload = self.prepare(
                task.audio_source_id,
                json.loads(self.inputs_json),
                task.path,
            )
            self.validate_media(request)
            if payload is not None:
                outcome = AudioTranscriptionOutcome.from_dict(payload)
            else:
                outcome = run_audio_transcription(
                    task,
                    self.options,
                    cancel_event=cancel,
                    progress=progress,
                )
                if outcome.status == "succeeded" and not cancel.is_set():
                    # Validate the serialized form before making it replayable.
                    outcome = AudioTranscriptionOutcome.from_dict(
                        json.loads(canonical_json(asdict(outcome)))
                    )
                    self.record(request, outcome)
            if cancel.is_set():
                return AudioTranscriptionOutcome(task.audio_source_id, "unprocessed")
            return outcome
        except FingerprintCancelled:
            cancel.set()
            return AudioTranscriptionOutcome(task.audio_source_id, "unprocessed")
        finally:
            if hasattr(self, "store"):
                self.store.close()

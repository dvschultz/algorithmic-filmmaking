"""Journal audio probes before desktop publication, retaining explicit saves."""

from dataclasses import replace
import json
from threading import Event
from typing import Callable

from core.jobs.audio_import import AudioImportRecord, audio_import_runtime
from core.jobs.commits import StaleJobResult, canonical_json
from core.jobs.gui_results import GuiResultJournal, GuiResultRequest
from core.jobs.media import FingerprintCancelled
from core.operations.audio_import import (
    AudioImportTask,
    AudioImportOutcome,
    run_audio_import,
)
from core.project import Project


class GuiAudioImportCache(GuiResultJournal):
    def __init__(self, project: Project, task: AudioImportTask, runtime: dict) -> None:
        project.session.assert_owner()
        if project.path is None:
            raise ValueError("Audio import recovery requires a saved project")
        target = str(task.path)
        super().__init__(
            project.path,
            project.metadata.id,
            {target: target},
            project.metadata.job_results,
            kind="gui_audio_import",
            arguments={},
            media_stamps={task.path: task.media_stamp},
            target_id_field="path",
        )
        inputs = task.to_dict()
        inputs.pop("audio_source_id")
        self.inputs_json = canonical_json({"task": inputs, "runtime": runtime})
        self.runtime_json = canonical_json(runtime)
        self.recorded: AudioImportRecord | None = None

    def validate_media(self, request: GuiResultRequest) -> None:
        super().validate_media(request)
        if canonical_json(audio_import_runtime()) != self.runtime_json:
            raise StaleJobResult("Audio import runtime changed")

    def run(
        self, task: AudioImportTask, cancel: Event, progress: Callable[[int, int], None]
    ) -> AudioImportOutcome:
        try:
            if cancel.is_set():
                return AudioImportOutcome(task.audio_source_id, "unprocessed")
            self.start(cancel)
            request, payload = self.prepare(
                str(task.path), json.loads(self.inputs_json), task.path
            )
            self.validate_media(request)
            if payload is None:
                outcome = run_audio_import(task, cancel_event=cancel, progress=progress)
                if outcome.status != "succeeded" or cancel.is_set():
                    return (
                        AudioImportOutcome(task.audio_source_id, "unprocessed")
                        if cancel.is_set()
                        else outcome
                    )
                payload = self.record(request, AudioImportRecord.build(task, outcome))
            recorded = AudioImportRecord.from_dict(payload)
            recovered = AudioImportTask.from_dict(recorded.task)
            if replace(task, audio_source_id=recovered.audio_source_id) != recovered:
                raise StaleJobResult(
                    "Recovered audio import inputs differ from request"
                )
            if cancel.is_set():
                return AudioImportOutcome(task.audio_source_id, "unprocessed")
            self.recorded = recorded
            return AudioImportOutcome.from_dict(recorded.outcome)
        except FingerprintCancelled:
            cancel.set()
            return AudioImportOutcome(task.audio_source_id, "unprocessed")
        finally:
            if hasattr(self, "store"):
                self.store.close()

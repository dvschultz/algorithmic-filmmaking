"""Recover extracted artifacts before owner delivery and explicit project save."""

from dataclasses import replace
import json
from threading import Event
from typing import Callable

from core.jobs.commits import StaleJobResult, canonical_json
from core.jobs.gui_results import GuiResultJournal, GuiResultRequest
from core.jobs.media import FingerprintCancelled
from core.jobs.frame_extraction import (
    FrameExtractionRecord as FrameExtractionRecord,
    frame_extraction_runtime as frame_extraction_runtime,
    frame_extraction_task_inputs,
)
from core.operations.frame_extraction import (
    FrameExtractionTask,
    FrameExtractionOutcome,
    run_frame_extraction,
)
from core.project import Project


class GuiFrameExtractionCache(GuiResultJournal):
    def __init__(
        self, project: Project, task: FrameExtractionTask, runtime: dict
    ) -> None:
        project.session.assert_owner()
        if project.path is None:
            raise ValueError("Extraction recovery requires a saved project")
        super().__init__(
            project.path,
            project.metadata.id,
            {task.source_id: task.source_id},
            project.metadata.job_results,
            kind="gui_extract_frames",
            arguments={
                "mode": task.mode,
                "interval": task.interval,
                "clip_id": task.clip_id,
            },
            media_stamps={task.path: task.media_stamp},
            target_id_field="source_id",
        )
        inputs = frame_extraction_task_inputs(task)
        self.inputs_json = canonical_json({"task": inputs, "runtime": runtime})
        self.runtime_json = canonical_json(runtime)
        self.recorded: FrameExtractionRecord | None = None

    def validate_media(self, request: GuiResultRequest) -> None:
        super().validate_media(request)
        if canonical_json(frame_extraction_runtime()) != self.runtime_json:
            raise StaleJobResult("Frame extraction runtime changed")

    def run(
        self,
        task: FrameExtractionTask,
        cancel: Event,
        progress: Callable[[int, int], None],
    ) -> FrameExtractionOutcome:
        try:
            if cancel.is_set():
                return FrameExtractionOutcome(task.request_id, "unprocessed")
            self.start(cancel)
            request, payload = self.prepare(
                task.source_id, json.loads(self.inputs_json), task.path
            )
            self.validate_media(request)
            if payload is None:
                outcome = run_frame_extraction(
                    task, cancel_event=cancel, progress=progress
                )
                if outcome.status != "succeeded" or cancel.is_set():
                    return (
                        FrameExtractionOutcome(task.request_id, "unprocessed")
                        if cancel.is_set()
                        else outcome
                    )
                recorded = FrameExtractionRecord.build(task, outcome)
                payload = self.record(request, recorded)
            recorded = FrameExtractionRecord.from_dict(payload)
            recovered = FrameExtractionTask.from_dict(recorded.task)
            if (
                recovered.artifact_dir.parent != task.artifact_dir.parent
                or replace(
                    task,
                    request_id=recovered.request_id,
                    artifact_dir=recovered.artifact_dir,
                )
                != recovered
            ):
                raise StaleJobResult(
                    "Recovered extraction inputs differ from the request"
                )
            if cancel.is_set():
                return FrameExtractionOutcome(task.request_id, "unprocessed")
            self.recorded = recorded
            return FrameExtractionOutcome.from_dict(recorded.outcome)
        except FingerprintCancelled:
            cancel.set()
            return FrameExtractionOutcome(task.request_id, "unprocessed")
        finally:
            if hasattr(self, "store"):
                self.store.close()

"""Recover extracted artifacts before owner delivery and explicit project save."""

from dataclasses import dataclass, replace
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.commits import StaleJobResult, canonical_json
from core.jobs.gui_results import GuiResultJournal, GuiResultRequest
from core.jobs.media import FingerprintCancelled, media_stamp
from core.operations.frame_extraction import (
    FrameExtractionTask,
    FrameExtractionOutcome,
    run_frame_extraction,
    validate_frame_artifacts,
)
from core.project import Project


def frame_extraction_runtime() -> dict:
    from importlib.metadata import version
    from core.binary_resolver import find_binary

    binaries = {}
    for name in ("ffmpeg", "ffprobe"):
        binary = find_binary(name)
        binaries[name] = {
            "path": str(binary) if binary else None,
            "stamp": list(media_stamp(Path(binary)) or ()) if binary else None,
        }
    return {"algorithm": "frame-extraction/v1", "pillow": version("Pillow"), **binaries}


@dataclass(frozen=True)
class FrameExtractionRecord:
    source_id: str
    status: str
    task: dict
    outcome: dict

    @classmethod
    def build(
        cls, task: FrameExtractionTask, outcome: FrameExtractionOutcome
    ) -> "FrameExtractionRecord":
        return cls.from_dict(
            {
                "source_id": task.source_id,
                "status": outcome.status,
                "task": task.to_dict(),
                "outcome": outcome.to_dict(),
            }
        )

    @classmethod
    def from_dict(cls, data: dict) -> "FrameExtractionRecord":
        # Detach JSON containers from the producer, then validate typed values.
        values = json.loads(canonical_json(data))
        task = FrameExtractionTask.from_dict(values["task"])
        outcome = FrameExtractionOutcome.from_dict(values["outcome"])
        if values["source_id"] != task.source_id or values["status"] != "succeeded":
            raise ValueError("Extraction receipt does not match its source")
        validate_frame_artifacts(task, outcome)
        return cls(task.source_id, "succeeded", task.to_dict(), outcome.to_dict())


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
        inputs = task.to_dict()
        inputs.pop("request_id")
        inputs["artifact_dir"] = str(task.artifact_dir.parent)
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

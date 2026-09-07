"""Shared extraction artifact records and saved-project execution."""

from dataclasses import dataclass, asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.commits import ResultSpec, StaleJobResult, canonical_json, result_batch
from core.jobs.media import FingerprintCancelled, MediaFingerprints, media_stamp
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore
from core.operations.frame_extraction import (
    FrameExtractionTask,
    FrameExtractionOutcome,
    FrameExtractionApplication,
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


def _task_for(
    project: Project, source_id: str, mode: str, interval: int, clip_id: str | None
) -> FrameExtractionTask:
    source = project.sources_by_id.get(source_id)
    if source is None:
        raise ValueError(f"Unknown source: {source_id}")
    clip = project.clips_by_id.get(clip_id) if clip_id is not None else None
    if clip_id is not None and clip is None:
        raise ValueError(f"Unknown clip: {clip_id}")
    if project.path is None:
        raise ValueError("Frame extraction requires a saved project")
    task = FrameExtractionTask.from_source(
        source, clip, mode, interval, project.path.parent / "frames"
    )
    return FrameExtractionTask.from_dict(task.to_dict())


def frame_extraction_task_inputs(task: FrameExtractionTask) -> dict:
    inputs = task.to_dict()
    inputs.pop("request_id")
    inputs["artifact_dir"] = str(task.artifact_dir.parent)
    normalized: dict = json.loads(canonical_json(inputs))
    return normalized


def frame_extraction_job_spec(
    project: Project,
    source_id: str,
    *,
    mode: str = "interval",
    interval: int = 10,
    clip_id: str | None = None,
) -> OperationSpec:
    task = _task_for(project, source_id, mode, interval, clip_id)
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="extract_frames",
        version=1,
        arguments={
            "source_id": source_id,
            "mode": mode,
            "interval": interval,
            "clip_id": clip_id,
        },
        inputs={
            "task": frame_extraction_task_inputs(task),
            "runtime": frame_extraction_runtime(),
        },
        persistence="job_history",
        session_id=project.session.session_id,
        input_revision=revision.digest if revision else None,
    )


class _ExtractionStopped(Exception):
    pass


def run_frame_extraction_job(
    store: JobStore,
    path: Path,
    source_id: str,
    progress: Callable[[float, str], None],
    cancel: Event,
    *,
    mode: str = "interval",
    interval: int = 10,
    clip_id: str | None = None,
    operation: OperationSpec | None = None,
) -> dict:
    """Append one extraction batch; interrupted publication reuses its artifacts."""
    fingerprints = MediaFingerprints(cancel)
    try:
        with result_batch(store, path, max_items=1) as batch:
            project = batch.project
            live = frame_extraction_job_spec(
                project, source_id, mode=mode, interval=interval, clip_id=clip_id
            )
            if operation is not None and (
                live.inputs_json != operation.inputs_json
                or live.arguments_json != operation.arguments_json
                or live.input_revision != operation.input_revision
            ):
                raise StaleJobResult("Frame extraction inputs changed while queued")
            if cancel.is_set():
                raise _ExtractionStopped("cancelled")
            task = _task_for(project, source_id, mode, interval, clip_id)
            application = FrameExtractionApplication(project, task)

            def inputs(current: Project) -> dict:
                current_task = _task_for(current, source_id, mode, interval, clip_id)
                return {
                    "project_id": current.metadata.id,
                    "task": frame_extraction_task_inputs(current_task),
                    "media": fingerprints.get(current_task.path),
                    "runtime": frame_extraction_runtime(),
                }

            captured = inputs(project)

            def decode(
                payload: dict,
            ) -> tuple[FrameExtractionTask, FrameExtractionOutcome]:
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
                        "Recorded extraction inputs differ from the request"
                    )
                return recovered, FrameExtractionOutcome.from_dict(recorded.outcome)

            def is_applied(current: Project, payload: dict) -> bool:
                recovered, outcome = decode(payload)
                for extracted in outcome.frames:
                    frame = current.frames_by_id.get(extracted.id)
                    if frame is None:
                        return False
                    actual = frame.to_dict()
                    expected = extracted.to_model(recovered).to_dict()
                    fields = (
                        "id",
                        "file_path",
                        "source_id",
                        "clip_id",
                        "frame_number",
                        "thumbnail_path",
                        "width",
                        "height",
                    )
                    if any(actual.get(key) != expected.get(key) for key in fields):
                        return False
                return True

            known = []
            for result_id, digest in project.metadata.job_results.items():
                row = store.get_result(result_id)
                if (
                    row is None
                    or sha256(row["spec_json"].encode()).hexdigest() != result_id
                ):
                    raise StaleJobResult(
                        "Committed result identity is missing or corrupt"
                    )
                identity = json.loads(row["spec_json"])
                if (
                    identity["kind"] != "extract_frames"
                    or identity["target_id"] != source_id
                ):
                    continue
                if (
                    sha256(row["payload_json"].encode()).hexdigest() != digest
                    or row["payload_digest"] != digest
                ):
                    raise StaleJobResult("Committed extraction payload is corrupt")
                known.append((row, identity))
            pending = next(
                (
                    row
                    for row, identity in known
                    if not row["committed"]
                    and identity["project_path"] == str(path.resolve())
                    and identity["inputs"]["basis"] == captured
                    and identity["arguments"] == live.arguments
                    and is_applied(project, json.loads(row["payload_json"]))
                ),
                None,
            )
            spec = (
                ResultSpec(path.resolve(), pending["spec_json"])
                if pending
                else ResultSpec.build(
                    path,
                    kind="extract_frames",
                    version=1,
                    target_id=source_id,
                    arguments=live.arguments,
                    inputs={"basis": captured, "generation": len(known)},
                )
            )

            def compute() -> dict:
                outcome = run_frame_extraction(
                    task,
                    cancel_event=cancel,
                    progress=lambda n, total: progress(
                        n / total if total else 1.0, "Extracting frames"
                    ),
                )
                if outcome.status != "succeeded" or cancel.is_set():
                    raise _ExtractionStopped(
                        "cancelled"
                        if cancel.is_set()
                        else outcome.message or "Extraction failed"
                    )
                return asdict(FrameExtractionRecord.build(task, outcome))

            def apply(current: Project, payload: dict) -> None:
                if cancel.is_set():
                    raise _ExtractionStopped("cancelled")
                recovered, outcome = decode(payload)
                if not application.apply(current, outcome, recovered_task=recovered):
                    raise StaleJobResult("Extraction target changed")

            receipt = batch.commit(
                spec,
                compute=compute,
                validate_input=lambda current: not cancel.is_set()
                and inputs(current) == captured,
                apply=apply,
                is_applied=is_applied,
            )
            _, outcome = decode(receipt["payload"])
            progress(1.0, "Extracted frames saved")
            return {
                "success": True,
                "result": {
                    "source_id": source_id,
                    "status": "succeeded" if receipt["applied"] else "recovered",
                    "frame_ids": [frame.id for frame in outcome.frames],
                    "frame_count": len(outcome.frames),
                },
            }
    except (FingerprintCancelled, _ExtractionStopped) as exc:
        return {
            "success": False,
            "source_id": source_id,
            "error": "cancelled" if cancel.is_set() else str(exc),
        }

"""Detection job metadata and JSON results shared by runtime adapters."""

import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.spec import OperationSpec
from core.operations.detection import (
    DetectionCancelled,
    DetectionGuard,
    DetectionRequest,
    StaleDetectionResult,
    run_detection,
)

from core.jobs.commits import ResultSpec, StaleJobResult, result_batch
from core.jobs.store import JobStore
from core.project import Project

DETECTION_OPERATION_VERSION = 1


def _source_inputs(project: Project, source_id: str) -> dict:
    source = project.sources_by_id.get(source_id)
    if source is None:
        return {"source_id": source_id, "missing": True}
    request = DetectionRequest.build(source.file_path)
    return {
        "project_id": project.metadata.id,
        "source_id": source_id,
        "video_path": str(request.video_path),
        "media_stamp": request.media_stamp,
    }


def saved_detection_spec(
    project: Project, path: Path, source_ids: list[str], sensitivity: float
) -> OperationSpec:
    """Capture saved-project inputs at submission, including target edits."""
    targets = []
    for source_id in source_ids:
        target = _source_inputs(project, source_id)
        source = project.sources_by_id.get(source_id)
        if source is not None:
            target["target_digest"] = DetectionGuard.capture(
                project, source.file_path, source_id=source_id
            ).target_digest
        targets.append(target)
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="detect_scenes_bulk",
        version=DETECTION_OPERATION_VERSION,
        arguments={
            "project_path": str(path),
            "source_ids": source_ids,
            "sensitivity": sensitivity,
        },
        inputs={"targets": targets},
        persistence="job_history",
        session_id=project.session.session_id,
        input_revision=revision.digest if revision is not None else None,
    )


class _DetectionFailure(Exception):
    pass


def run_saved_detection(
    store: JobStore,
    path: Path,
    source_ids: list[str],
    sensitivity: float,
    progress: Callable[[float, str], None],
    cancel: Event,
) -> dict:
    """Publish one source and its receipt at a time, reusing recorded outputs."""
    from core.scene_detect import DetectionConfig
    from core.spine.detect import _generate_detected_clip_thumbnails
    from models.clip import Clip, Source

    output: dict = {"succeeded": [], "failed": [], "cancelled": []}
    with result_batch(store, path, max_items=1) as batch:
        project = batch.project
        for index, source_id in enumerate(source_ids):
            if cancel.is_set():
                output["cancelled"].extend(source_ids[index:])
                break
            source = project.sources_by_id.get(source_id)
            if source is None:
                output["failed"].append(
                    {
                        "source_id": source_id,
                        "code": "unknown_source_id",
                        "message": None,
                    }
                )
                continue
            if not source.file_path.exists():
                output["failed"].append(
                    {
                        "source_id": source_id,
                        "code": "source_file_missing",
                        "message": None,
                    }
                )
                continue
            inputs = _source_inputs(project, source_id)
            request = DetectionRequest.build(
                source.file_path, DetectionConfig(threshold=sensitivity)
            )
            guard = DetectionGuard.capture(
                project, source.file_path, source_id=source_id
            )
            spec = ResultSpec.build(
                path,
                kind="detect_scenes",
                version=DETECTION_OPERATION_VERSION,
                target_id=source_id,
                arguments={"sensitivity": sensitivity},
                inputs=inputs,
            )

            def compute():
                try:
                    _detected_source, clips = run_detection(
                        request,
                        cancel_event=cancel,
                        progress_callback=lambda fraction, message: progress(
                            (index + 0.8 * fraction) / max(len(source_ids), 1), message
                        ),
                    )
                    for clip in clips:
                        clip.source_id = source_id
                    published_source = Source.from_dict(source.to_dict())
                    published_source.analyzed = True
                    thumbnails = _generate_detected_clip_thumbnails(
                        published_source,
                        clips,
                        cancel_event=cancel,
                        progress_callback=lambda fraction, message: progress(
                            (index + fraction) / max(len(source_ids), 1), message
                        ),
                    )
                    if cancel.is_set():
                        raise DetectionCancelled()
                    request.validate_media()
                    return {
                        "before": guard.target_digest,
                        "source": published_source.to_dict(),
                        "clips": [c.to_dict() for c in clips],
                        "thumbnails": thumbnails,
                    }
                except (DetectionCancelled, StaleDetectionResult):
                    raise
                except Exception as exc:
                    if cancel.is_set():
                        raise DetectionCancelled() from exc
                    raise _DetectionFailure(str(exc)) from exc

            def validate(current):
                return _source_inputs(current, source_id) == inputs

            def apply(current, payload):
                live = DetectionGuard.capture(
                    current, source.file_path, source_id=source_id
                )
                if live.target_digest != payload["before"]:
                    raise StaleJobResult(
                        "Detection target changed before pending result publication"
                    )
                current.update_source(source_id, analyzed=True)
                current.replace_source_clips(
                    source_id, [Clip.from_dict(item) for item in payload["clips"]]
                )

            def is_applied(current, payload):
                live = current.sources_by_id.get(source_id)
                return (
                    live is not None
                    and live.to_dict() == payload["source"]
                    and [
                        c.to_dict() for c in current.clips_by_source.get(source_id, [])
                    ]
                    == payload["clips"]
                )

            try:
                receipt = batch.commit(
                    spec,
                    compute=compute,
                    validate_input=validate,
                    apply=apply,
                    is_applied=is_applied,
                )
            except DetectionCancelled:
                output["cancelled"].extend(source_ids[index:])
                break
            except _DetectionFailure as exc:
                output["failed"].append(
                    {
                        "source_id": source_id,
                        "code": "detection_failed",
                        "message": str(exc),
                    }
                )
                continue
            payload = receipt["payload"]
            output["succeeded"].append(
                {
                    "source_id": source_id,
                    "source_name": source.filename,
                    "clip_count": len(payload["clips"]),
                    "clip_ids": [c["id"] for c in payload["clips"]],
                    "thumbnails": payload["thumbnails"],
                }
            )
            progress(
                (index + 1) / max(len(source_ids), 1),
                f"Detected scenes in {source.filename}",
            )
    return {"success": True, "result": output}


def detection_job_spec(
    request: DetectionRequest, guard: DetectionGuard | None = None
) -> OperationSpec:
    """Describe live-project work without claiming restart-safe publication."""
    return OperationSpec.build(
        kind="detect_scenes",
        version=DETECTION_OPERATION_VERSION,
        arguments={
            "mode": request.mode,
            "config": json.loads(request.config_json),
            "karaoke_config": json.loads(request.karaoke_config_json),
        },
        inputs={
            "video_path": str(request.video_path),
            "media_stamp": request.media_stamp,
            "source_id": guard.source_id if guard else None,
            "target_digest": guard.target_digest if guard else None,
        },
        persistence="session_only",
        session_id=guard.session_id if guard else None,
    )


def run_detection_job(
    request: DetectionRequest,
    progress: Callable[[float, str], None],
    cancel: Event,
) -> dict:
    """Return a serializable result; let the runtime own terminal state."""
    try:
        source, clips = run_detection(
            request, progress_callback=progress, cancel_event=cancel
        )
    except Exception:
        # Native backends may report their own exception while stopping.
        # An accepted cancellation must retain the cancelled terminal state.
        if not cancel.is_set():
            raise
        return {"cancelled": True}
    return {
        "success": True,
        "source": source.to_dict(),
        "clips": [clip.to_dict() for clip in clips],
    }

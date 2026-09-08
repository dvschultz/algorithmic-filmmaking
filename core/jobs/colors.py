"""Color pilot using durable per-target result publication."""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
from threading import Event
from typing import Callable, Literal
import json

from core.jobs.commits import ResultBatch, ResultSpec, StaleJobResult, result_batch
from core.jobs.store import JobStore
from core.jobs.spec import OperationSpec
from core.analysis_records import AnalysisFingerprints, model_runtime
from core.operations.colors import (
    ColorApplication,
    ColorRequest,
    ColorTarget,
    color_request,
    color_identity,
    compute_colors,
    reusable_colors,
)
from core.operations.contracts import ColorOutcome, ColorResult
from core.project import Project
from models.analysis_record import AnalysisRecord

COLOR_OPERATION_VERSION = 2


def color_job_spec(
    request: ColorRequest,
    *,
    arguments: dict,
    persistence: Literal["job_history", "session_only"],
    session_id: str | None,
    input_revision: str | None,
) -> OperationSpec:
    """Describe the exact color snapshot and policy accepted at submission."""
    targets = []
    for target in request.targets:
        snapshot = asdict(target)
        snapshot["inputs"] = target.inputs.to_dict() if target.inputs is not None else None
        for key in ("video_path", "image_path"):
            snapshot[key] = str(snapshot[key]) if snapshot[key] is not None else None
        targets.append(snapshot)
    return OperationSpec.build(
        kind="analyze_colors",
        version=COLOR_OPERATION_VERSION,
        arguments=arguments,
        inputs={
            "targets": targets,
            "num_colors": request.num_colors,
            "skip_existing": request.skip_existing,
            "skip_empty": request.skip_empty,
        },
        persistence=persistence,
        cancellable=True,
        session_id=session_id,
        input_revision=input_revision,
    )


class _OutcomeError(Exception):
    def __init__(self, outcome: ColorOutcome):
        self.outcome = outcome


def _inputs(project: Project, target: ColorTarget) -> dict:
    clip = project.clips_by_id.get(target.target_id)
    return {
        "project_id": project.metadata.id,
        "source_id": clip.source_id if clip else None,
        "missing": target.missing,
        "video_path": str(target.video_path) if target.video_path else None,
        "image_path": str(target.image_path) if target.image_path else None,
        "start_frame": target.start_frame,
        "end_frame": target.end_frame,
        "file_stamp": list(target.file_stamp) if target.file_stamp else None,
        "analysis_input": target.inputs.to_dict() if target.inputs is not None else None,
    }


def run_colors(
    store: JobStore,
    path: Path,
    clip_ids: list[str] | None,
    num_colors: int,
    progress: Callable[[float, str], None],
    cancel: Event,
) -> dict:
    from core.spine.security import validate_project_path

    valid, error, resolved = validate_project_path(str(path))
    if not valid:
        raise ValueError(error)
    path = resolved
    with result_batch(store, path) as batch:
        return _run_color_batch(store, batch, clip_ids, num_colors, progress, cancel)


def _run_color_batch(
    store: JobStore,
    batch: ResultBatch,
    clip_ids: list[str] | None,
    num_colors: int,
    progress: Callable[[float, str], None],
    cancel: Event,
) -> dict:
    project, path = batch.project, batch.path
    ids = (
        list(dict.fromkeys(clip_ids))
        if clip_ids is not None
        else [c.id for c in project.clips]
    )
    output: dict = {
        "succeeded": [],
        "failed": [],
        "skipped": [],
        "unprocessed": [],
        "total_clips": len(ids),
    }
    known: dict[str, list[dict]] = {}
    fingerprints = AnalysisFingerprints(cancel)
    runtime = model_runtime("kmeans-rgb", ("numpy", "scikit-learn", "opencv-python"))
    for result_id in project.metadata.job_results:
        row = store.get_result(result_id)
        if row is None:
            continue
        identity = json.loads(row["spec_json"])
        if identity["kind"] == "analyze_colors":
            known.setdefault(identity["target_id"], []).append(row)
    for index, clip_id in enumerate(ids):
        if cancel.is_set():
            output["unprocessed"].extend(
                {"clip_id": cid, "reason": "cancelled"} for cid in ids[index:]
            )
            break
        request = color_request(project, [clip_id], num_colors, skip_existing=True)
        target = request.targets[0]
        inputs = _inputs(project, target)
        semantic = color_identity(target, num_colors, fingerprints, runtime) if target.inputs is not None and target.inputs.unchanged() else None
        reused = semantic is not None and reusable_colors(target, semantic)
        spec = ResultSpec.build(
            path,
            kind="analyze_colors",
            version=COLOR_OPERATION_VERSION,
            target_id=clip_id,
            arguments={"num_colors": num_colors},
            inputs={
                "basis": inputs, "analysis_key": semantic.key if semantic is not None else None,
                "previous_colors": target.existing_colors, "previous_record": target.record_json,
            },
        )
        if reused:
            assert semantic is not None
            for row in known.get(clip_id, []):
                saved_identity = json.loads(row["spec_json"])
                payload = json.loads(row["payload_json"])
                if (
                    saved_identity["project_path"] == str(path)
                    and saved_identity["inputs"].get("basis") == inputs
                    and saved_identity["inputs"].get("analysis_key") == semantic.key
                    and payload.get("record_json") == target.record_json
                ):
                    spec = ResultSpec(path, row["spec_json"])
                    break

        def compute():
            outcome = compute_colors(request, cancel_event=cancel, fingerprints=fingerprints, runtime=runtime).outcomes[0]
            if outcome.status not in ("succeeded", "skipped"):
                raise _OutcomeError(outcome)
            return {"colors": [list(color) for color in outcome.colors], "record_json": outcome.record_json}

        def validate(current, clip_id=clip_id, inputs=inputs):
            live = color_request(
                current, [clip_id], num_colors, skip_existing=False
            ).targets[0]
            return _inputs(current, live) == inputs

        def apply(current, payload):
            live_request = color_request(
                current, [clip_id], num_colors, skip_existing=False
            )
            outcome = ColorOutcome(
                clip_id, "succeeded", tuple(tuple(c) for c in payload["colors"]), record_json=payload.get("record_json"),
            )
            result = ColorApplication(current, live_request).apply(
                ColorResult(live_request.request_id, (outcome,))
            )
            if result.outcomes[0].status != "succeeded":
                raise StaleJobResult("Color inputs changed during application")

        def is_applied(current, payload, clip_id=clip_id):
            clip = current.clips_by_id.get(clip_id)
            return clip is not None and clip.dominant_colors == [
                tuple(c) for c in payload["colors"]
            ] and (
                payload.get("record_json") is None
                or (clip.analysis_records.get("colors") is not None
                    and clip.analysis_records["colors"].to_dict() == json.loads(payload["record_json"]))
            )

        try:
            receipt = batch.commit(
                spec,
                compute=compute,
                validate_input=validate,
                apply=apply,
                is_applied=is_applied,
            )
            if receipt["applied"] and not reused:
                output["succeeded"].append(
                    {
                        "clip_id": clip_id,
                        "color_count": len(receipt["payload"]["colors"]),
                    }
                )
            else:
                output["skipped"].append(
                    {"clip_id": clip_id, "reason": "already_committed"}
                )
        except _OutcomeError as exc:
            outcome = exc.outcome
            if outcome.status == "unprocessed":
                output["unprocessed"].append(
                    {"clip_id": clip_id, "reason": outcome.code}
                )
            else:
                if outcome.record_json is not None:
                    record = AnalysisRecord.from_dict(json.loads(outcome.record_json))

                    def apply_failure(current: Project, cid: str = clip_id, record: AnalysisRecord = record) -> None:
                        current.record_analysis("clip", cid, "colors", record)

                    def failure_applied(current: Project, cid: str = clip_id, record: AnalysisRecord = record) -> bool:
                        return current.clips_by_id[cid].analysis_records.get("colors") == record

                    batch.stage_analysis(
                        apply=apply_failure,
                        validate_input=validate,
                        is_applied=failure_applied,
                    )
                failure = {"clip_id": clip_id, "code": outcome.code}
                if outcome.message:
                    failure["message"] = outcome.message
                output["failed"].append(failure)
        progress(
            0.95 * (index + 1) / len(ids),
            f"Color computation ({index + 1}/{len(ids)}): {clip_id}",
        )
    batch.flush()
    progress(1.0, "Color analysis finished")
    return {"success": True, "result": output}

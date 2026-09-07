"""Color pilot using durable per-target result publication."""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
from threading import Event
from typing import Callable, Literal
import json

from core.jobs.commits import ResultSpec, StaleJobResult, commit_result
from core.jobs.store import JobStore
from core.jobs.spec import OperationSpec
from core.operations.colors import (
    ColorApplication,
    ColorRequest,
    ColorTarget,
    color_request,
    compute_colors,
)
from core.operations.contracts import ColorOutcome, ColorResult
from core.project import Project
from core.spine.project_io import load_with_mtime

COLOR_OPERATION_VERSION = 1


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
    project, _ = load_with_mtime(path)
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
    managed_targets = set()
    for result_id in project.metadata.job_results:
        row = store.get_result(result_id)
        if row is None:
            raise StaleJobResult("Committed result payload is missing")
        identity = json.loads(row["spec_json"])
        if identity["kind"] == "analyze_colors":
            managed_targets.add(identity["target_id"])
    for index, clip_id in enumerate(ids):
        if cancel.is_set():
            output["unprocessed"].extend(
                {"clip_id": cid, "reason": "cancelled"} for cid in ids[index:]
            )
            break
        project, _ = load_with_mtime(path)
        request = color_request(project, [clip_id], num_colors, skip_existing=False)
        target = request.targets[0]
        inputs = _inputs(project, target)
        spec = ResultSpec.build(
            path,
            kind="analyze_colors",
            version=COLOR_OPERATION_VERSION,
            target_id=clip_id,
            arguments={"num_colors": num_colors},
            inputs=inputs,
        )
        recorded = store.get_result(spec.result_id)
        # Existing results without our receipts retain the legacy skip policy.
        if target.existing_colors and clip_id not in managed_targets:
            output["skipped"].append(
                {"clip_id": clip_id, "reason": "already_populated"}
            )
            continue
        if (
            recorded
            and spec.result_id not in project.metadata.job_results
            and target.existing_colors
        ):
            raise StaleJobResult(
                "Color output changed before pending result publication"
            )

        def compute():
            outcome = compute_colors(request, cancel_event=cancel).outcomes[0]
            if outcome.status != "succeeded":
                raise _OutcomeError(outcome)
            return {"colors": [list(color) for color in outcome.colors]}

        def validate(current):
            live = color_request(
                current, [clip_id], num_colors, skip_existing=False
            ).targets[0]
            return _inputs(current, live) == inputs

        def apply(current, payload):
            live_request = color_request(
                current, [clip_id], num_colors, skip_existing=False
            )
            outcome = ColorOutcome(
                clip_id, "succeeded", tuple(tuple(c) for c in payload["colors"])
            )
            result = ColorApplication(current, live_request).apply(
                ColorResult(live_request.request_id, (outcome,))
            )
            if result.outcomes[0].status != "succeeded":
                raise StaleJobResult("Color inputs changed during application")

        def is_applied(current, payload):
            clip = current.clips_by_id.get(clip_id)
            return clip is not None and clip.dominant_colors == [
                tuple(c) for c in payload["colors"]
            ]

        try:
            receipt = commit_result(
                store,
                spec,
                compute=compute,
                validate_input=validate,
                apply=apply,
                is_applied=is_applied,
            )
            if receipt["applied"]:
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
                failure = {"clip_id": clip_id, "code": outcome.code}
                if outcome.message:
                    failure["message"] = outcome.message
                output["failed"].append(failure)
        progress(
            (index + 1) / len(ids),
            f"Color analysis ({index + 1}/{len(ids)}): {clip_id}",
        )
    progress(1.0, "Color analysis finished")
    return {"success": True, "result": output}

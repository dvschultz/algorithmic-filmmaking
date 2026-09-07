"""Durable gaze results for saved-project jobs."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.commits import ResultSpec, StaleJobResult, result_batch
from core.jobs.media import FingerprintCancelled, MediaFingerprints, media_stamp
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore
from core.operations.gaze import (
    GazeApplication,
    GazeOptions,
    GazeOutcome,
    GazeTask,
    run_gaze,
    gaze_model_session,
)
from core.project import Project
from core.project_revision import ProjectRevisionConflict


def _runtime() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    packages: dict[str, str | None] = {}
    for package in ("mediapipe", "opencv-python", "numpy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "model": "face_landmarker/float16/1",
        "algorithm": "iris-ratios/v1",
        "packages": packages,
    }


def _ids(project: Project, clip_ids: list[str] | None) -> list[str]:
    ids = (
        list(dict.fromkeys(clip_ids))
        if clip_ids is not None
        else [c.id for c in project.clips]
    )
    if any(cid not in project.clips_by_id for cid in ids):
        raise ValueError("Unknown gaze analysis clip ID")
    return ids


def _task(project: Project, cid: str) -> GazeTask:
    clip = project.clips_by_id[cid]
    source = project.sources_by_id.get(clip.source_id)
    return GazeTask(
        cid,
        clip.source_id,
        source.file_path if source else None,
        clip.start_frame,
        clip.end_frame,
        source.fps if source else 0.0,
    )


def _task_data(task: GazeTask) -> dict:
    data = asdict(task)
    data["source_path"] = str(task.source_path) if task.source_path else None
    return data


def _values(project: Project, cid: str) -> dict:
    clip = project.clips_by_id[cid]
    return {
        "gaze_yaw": clip.gaze_yaw,
        "gaze_pitch": clip.gaze_pitch,
        "gaze_category": clip.gaze_category,
    }


def _saved_gaze(payload: dict) -> dict:
    """Compare saved angles at the existing two-decimal project precision."""
    return {
        key: round(value, 2) if key != "gaze_category" and value is not None else value
        for key, value in payload.items()
    }


def _outcome(cid: str, payload: dict) -> GazeOutcome:
    if payload == {"gaze_yaw": None, "gaze_pitch": None, "gaze_category": None}:
        return GazeOutcome(cid, "succeeded")
    return GazeOutcome.from_result(cid, payload)


def gaze_job_spec(
    project: Project,
    clip_ids: list[str] | None,
    options: GazeOptions,
    *,
    arguments: dict,
) -> OperationSpec:
    targets = []
    for cid in _ids(project, clip_ids):
        task = _task(project, cid)
        targets.append(
            {
                **_task_data(task),
                "source_stamp": media_stamp(task.source_path)
                if task.source_path
                else None,
            }
        )
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="gaze",
        version=1,
        arguments=arguments,
        inputs={
            "targets": targets,
            "options": asdict(options),
            "runtime": _runtime(),
        },
        persistence="job_history",
        session_id=project.session.session_id,
        input_revision=revision.digest if revision else None,
    )


class _OutcomeError(Exception):
    def __init__(self, outcome: GazeOutcome):
        self.outcome = outcome


def run_gaze_job(
    store: JobStore,
    path: Path,
    clip_ids: list[str] | None,
    progress: Callable[[float, str], None],
    cancel: Event,
    *,
    options: GazeOptions | None = None,
    force: bool = False,
    operation: OperationSpec | None = None,
) -> dict:
    """Reuse completed computation after failed saves, without implicit provider calls."""
    options = (
        GazeOptions(**json.loads(operation.inputs_json)["options"])
        if operation
        else (options or GazeOptions())
    )
    if operation is not None:
        force = bool(operation.arguments.get("force", False))
    runtime = _runtime()
    fingerprint = MediaFingerprints(cancel).get
    with gaze_model_session() as model_session, result_batch(store, path) as batch:
        project = batch.project
        if operation is not None:
            revision = project.session.file_revision
            if operation.input_revision is not None and (
                revision is None or revision.digest != operation.input_revision
            ):
                raise ProjectRevisionConflict(path)
            live = gaze_job_spec(
                project, clip_ids, options, arguments=operation.arguments
            )
            if live.inputs_json != operation.inputs_json:
                raise StaleJobResult("Gaze analysis inputs changed while queued")
        ids = _ids(project, clip_ids)
        if force:
            batch.max_items = max(1, len(ids))
        known: dict[str, list[tuple[dict, dict, dict]]] = {}
        for rid, digest in project.metadata.job_results.items():
            row = store.get_result(rid)
            if row is None:
                raise StaleJobResult("Committed result payload is missing")
            if sha256(row["spec_json"].encode()).hexdigest() != rid:
                raise StaleJobResult("Committed result identity is corrupt")
            identity = json.loads(row["spec_json"])
            if identity["kind"] != "gaze":
                continue
            if (
                sha256(row["payload_json"].encode()).hexdigest() != digest
                or row["payload_digest"] != digest
            ):
                raise StaleJobResult("Committed gaze payload is corrupt")
            known.setdefault(identity["target_id"], []).append(
                (row, identity, json.loads(row["payload_json"]))
            )

        def inputs(current: Project, cid: str) -> dict:
            task = _task(current, cid)
            return {
                "project_id": current.metadata.id,
                "task": _task_data(task),
                "source": fingerprint(task.source_path),
                "runtime": _runtime(),
            }

        def is_output(current: Project, cid: str, payload: dict) -> bool:
            existing = _values(current, cid)
            return existing == payload or existing == _saved_gaze(payload)

        result: dict = {
            "succeeded": [],
            "failed": [],
            "skipped": [],
            "unprocessed": [],
            "total_clips": len(ids),
        }
        for index, cid in enumerate(ids):
            if cancel.is_set():
                result["unprocessed"].extend(
                    {"clip_id": rest, "code": "cancelled"} for rest in ids[index:]
                )
                break
            existing = any(
                value is not None for value in _values(project, cid).values()
            )
            if existing and not force and cid not in known:
                result["skipped"].append(
                    {"clip_id": cid, "reason": "already_populated"}
                )
                continue
            try:
                basis = inputs(project, cid)
                if basis["runtime"] != runtime:
                    raise StaleJobResult("Gaze analysis runtime changed")
                task = _task(project, cid)
                identity_inputs: dict = {"basis": basis}
                if force:
                    identity_inputs["generation"] = len(known.get(cid, []))
                    identity_inputs["previous_gaze"] = _values(project, cid)
                arguments = asdict(options)
                spec = ResultSpec.build(
                    path,
                    kind="gaze",
                    version=1,
                    target_id=cid,
                    arguments=arguments,
                    inputs=identity_inputs,
                )
                specs = [spec]
                if force:
                    for row, identity, payload in known.get(cid, []):
                        if (
                            not row["committed"]
                            and identity["project_path"] == str(path)
                            and identity["arguments"] == arguments
                            and identity["inputs"]["basis"] == basis
                            and is_output(project, cid, payload)
                        ):
                            specs = [ResultSpec(path, row["spec_json"])]
                            break
                if cid in known and not force:
                    matches = [
                        row
                        for row, identity, payload in known[cid]
                        if identity["project_path"] == str(path)
                        and identity["inputs"]["basis"] == basis
                        and identity["arguments"] == arguments
                        and is_output(project, cid, payload)
                    ]
                    if not matches and existing:
                        result["skipped"].append(
                            {"clip_id": cid, "reason": "already_populated"}
                        )
                        continue
                    if matches:
                        specs = [ResultSpec(path, row["spec_json"]) for row in matches]

                application = GazeApplication(project, (task,))

                def compute(task=task):
                    outcome = run_gaze(
                        (task,),
                        options,
                        cancel_event=cancel,
                        model_session=model_session,
                    )[0]
                    if (
                        outcome.status != "succeeded"
                        and outcome.code != "no_gaze_detected"
                    ):
                        raise _OutcomeError(outcome)
                    return {
                        "gaze_yaw": outcome.yaw,
                        "gaze_pitch": outcome.pitch,
                        "gaze_category": outcome.category,
                    }

                def apply(current, payload, cid=cid, application=application):
                    outcome = _outcome(cid, payload)
                    if not application.apply(current, outcome):
                        raise StaleJobResult(
                            "Gaze analysis target changed during application"
                        )

                def validate(current, cid=cid, basis=basis):
                    return inputs(current, cid) == basis

                def is_applied(current, payload, cid=cid):
                    return is_output(current, cid, payload)

                for candidate in specs:
                    receipt = batch.commit(
                        candidate,
                        compute=compute,
                        validate_input=validate,
                        apply=apply,
                        is_applied=is_applied,
                    )
                if receipt["payload"]["gaze_category"] is None:
                    # Keep the existing public no-gaze outcome while recording the
                    # completed observation, so retry does not repeat inference.
                    result["failed"].append(
                        {"clip_id": cid, "code": "no_gaze_detected"}
                    )
                elif receipt["applied"]:
                    result["succeeded"].append(
                        {
                            "clip_id": cid,
                            "gaze_category": receipt["payload"]["gaze_category"],
                        }
                    )
                else:
                    result["skipped"].append(
                        {"clip_id": cid, "reason": "already_committed"}
                    )
            except FingerprintCancelled:
                result["unprocessed"].extend(
                    {"clip_id": rest, "code": "cancelled"} for rest in ids[index:]
                )
                break
            except _OutcomeError as exc:
                outcome = exc.outcome
                result[outcome.status].append(
                    {"clip_id": cid, "code": outcome.code, "message": outcome.message}
                )
                if outcome.code == "model_load_failed":
                    result["unprocessed"].extend(
                        {"clip_id": rest, "code": "model_unavailable"}
                        for rest in ids[index + 1 :]
                    )
                    break
            progress(
                0.95 * (index + 1) / len(ids),
                f"Gaze analysis ({index + 1}/{len(ids)})",
            )
        batch.flush()
        progress(1.0, "Gaze analysis finished")
        return {"success": True, "result": result}

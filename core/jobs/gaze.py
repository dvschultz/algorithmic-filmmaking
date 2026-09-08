"""Durable gaze results for saved-project jobs."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot
from core.analysis_model_identity import gaze_runtime
from models.analysis_record import AnalysisRecord

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
    gaze_task,
    gaze_identity,
    gaze_values,
)
from core.project import Project
from core.project_revision import ProjectRevisionConflict


def _runtime() -> dict:
    return gaze_runtime()


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
    return gaze_task(clip, source, skip_existing=False)


def _task_data(task: GazeTask) -> dict:
    data = asdict(task)
    data.pop("analysis_json", None)
    data["analysis_version"] = 2 if task.analysis_json is not None else 1
    data["skip"] = False
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
        if key in ("gaze_yaw", "gaze_pitch", "gaze_category")
    }


def _outcome(cid: str, payload: dict) -> GazeOutcome:
    from dataclasses import replace

    values = {key: payload[key] for key in ("gaze_yaw", "gaze_pitch", "gaze_category")}
    outcome = (
        GazeOutcome(cid, "succeeded", code="no_gaze_detected")
        if all(value is None for value in values.values())
        else GazeOutcome.from_result(cid, values)
    )
    return replace(outcome, record_json=payload.get("record_json"))


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
        version=2,
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
    media_fingerprints = MediaFingerprints(cancel)
    fingerprint = media_fingerprints.get
    fingerprints = AnalysisFingerprints(cancel, media_fingerprints=media_fingerprints)
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
                continue  # Project records survive job-cache removal.
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
            record = current.clips_by_id[cid].analysis_records.get("gaze")
            return (
                isinstance(record, AnalysisRecord)
                and record.to_dict() == json.loads(payload.get("record_json") or "null")
                and gaze_values(current.clips_by_id[cid]) == _saved_gaze(payload)
            )

        def stage_record(cid: str, record: AnalysisRecord, basis: dict) -> None:
            batch.stage_analysis(
                apply=lambda current: current.record_analysis(
                    "clip", cid, "gaze", record
                ),
                validate_input=lambda current: inputs(current, cid) == basis,
                is_applied=lambda current: current.clips_by_id[
                    cid
                ].analysis_records.get("gaze")
                == record,
            )

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
            try:
                basis = inputs(project, cid)
                if basis["runtime"] != runtime:
                    raise StaleJobResult("Gaze analysis runtime changed")
                task = _task(project, cid)
                assert task.analysis_json is not None
                snapshot = AnalysisSnapshot.from_json(task.analysis_json)
                semantic = (
                    gaze_identity(snapshot, options, fingerprints, runtime)
                    if snapshot.inputs.unchanged()
                    else None
                )
                reused = (
                    snapshot.reusable_record(semantic)
                    if semantic is not None and not force
                    else None
                )
                prior_record = project.clips_by_id[cid].analysis_records.get("gaze")
                identity_inputs: dict = {
                    "basis": basis,
                    "previous_gaze": _values(project, cid),
                    "previous_record": prior_record.to_dict() if prior_record else None,
                }
                if force:
                    identity_inputs["generation"] = len(known.get(cid, []))
                    identity_inputs["previous_gaze"] = _values(project, cid)
                arguments = asdict(options)
                spec = ResultSpec.build(
                    path,
                    kind="gaze",
                    version=2,
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
                if not force:
                    matches = [
                        row
                        for row, identity, payload in known.get(cid, [])
                        if identity["project_path"] == str(path)
                        and identity["inputs"]["basis"] == basis
                        and identity["arguments"] == arguments
                        and is_output(project, cid, payload)
                    ]
                    if not matches and reused is not None:
                        if reused != prior_record:
                            stage_record(cid, reused, basis)
                        category = (
                            "failed"
                            if reused.value["gaze_category"] is None
                            else "skipped"
                        )
                        result[category].append(
                            {"clip_id": cid, "code": "no_gaze_detected"}
                            if category == "failed"
                            else {"clip_id": cid, "reason": "valid_analysis"}
                        )
                        continue
                    if matches:
                        specs = [ResultSpec(path, row["spec_json"]) for row in matches]

                application = GazeApplication(project, (task,), options)

                def compute(task=task):
                    outcome = run_gaze(
                        (task,),
                        options,
                        cancel_event=cancel,
                        model_session=model_session,
                        fingerprints=fingerprints,
                        runtime=runtime,
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
                        "record_json": outcome.record_json,
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
                if inputs(project, cid) != basis:
                    raise StaleJobResult("Gaze inputs changed during computation")
                if outcome.can_apply and outcome.record_json is not None:
                    stage_record(
                        cid,
                        AnalysisRecord.from_dict(json.loads(outcome.record_json)),
                        basis,
                    )
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

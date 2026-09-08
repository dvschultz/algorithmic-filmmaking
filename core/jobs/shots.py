"""Durable shot classification for saved-project jobs."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.analysis_records import (
    AnalysisFingerprints,
    AnalysisSnapshot,
    recorded_image_path,
)
from models.analysis_record import AnalysisRecord

from core.jobs.commits import ResultSpec, StaleJobResult, result_batch
from core.jobs.media import FingerprintCancelled, MediaFingerprints, media_stamp
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore
from core.operations.shots import (
    ShotTypeApplication,
    ShotTypeOptions,
    ShotTypeOutcome,
    ShotTypeTask,
    shot_task,
    shot_identity,
    run_shot_types,
)
from core.project import Project
from core.project_revision import ProjectFileRevision, ProjectRevisionConflict


def _runtime() -> dict:
    from core.analysis_model_identity import shot_runtime

    return shot_runtime()


def _ids(project: Project, clip_ids: list[str] | None) -> list[str]:
    ids = (
        list(dict.fromkeys(clip_ids))
        if clip_ids is not None
        else [c.id for c in project.clips]
    )
    if any(cid not in project.clips_by_id for cid in ids):
        raise ValueError("Unknown shots clip ID")
    return ids


def _task(
    project: Project,
    cid: str,
    thumbnails: dict[str, Path] | None = None,
    *,
    force: bool = False,
) -> ShotTypeTask:
    clip = project.clips_by_id[cid]
    source = project.sources_by_id.get(clip.source_id)
    image = (thumbnails or {}).get(cid)
    if image is None and not force:
        image = recorded_image_path(clip, source, "shots")
    return shot_task(clip, source, image_path=image, skip_existing=False)


def _task_data(task: ShotTypeTask) -> dict:
    data = asdict(task)
    data.pop("analysis_json")
    data["skip"] = False
    data["analysis_version"] = 2 if task.analysis_json is not None else 1
    data["thumbnail_path"] = str(task.thumbnail_path) if task.thumbnail_path else None
    data["source_path"] = str(task.source_path) if task.source_path else None
    # Result identities are compared with decoded JSON on recovery.
    data["image_stamp"] = list(task.image_stamp) if task.image_stamp else None
    data["source_stamp"] = list(task.source_stamp) if task.source_stamp else None
    return data


def _source_data(project: Project, cid: str) -> dict:
    source_id = project.clips_by_id[cid].source_id
    source = project.sources_by_id.get(source_id)
    return {
        "source_id": source_id,
        "start_frame": project.clips_by_id[cid].start_frame,
        "end_frame": project.clips_by_id[cid].end_frame,
        "fps": source.fps if source else None,
        "actual_source_path": str(source.file_path) if source else None,
    }


def shot_job_spec(
    project: Project,
    clip_ids: list[str] | None,
    options: ShotTypeOptions,
    *,
    arguments: dict,
) -> OperationSpec:
    targets = []
    for cid in _ids(project, clip_ids):
        task = _task(project, cid, force=bool(arguments.get("force", False)))
        source = _source_data(project, cid)
        source_path = source["actual_source_path"]
        targets.append(
            {
                **_task_data(task),
                **source,
                "image_stamp": media_stamp(task.thumbnail_path)
                if task.thumbnail_path
                else None,
                "source_stamp": media_stamp(Path(source_path)) if source_path else None,
            }
        )
    revision = project.session.file_revision
    if revision is None and project.path is not None:
        revision = ProjectFileRevision.capture(project.path)
    return OperationSpec.build(
        kind="shots",
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
    def __init__(self, outcome: ShotTypeOutcome):
        self.outcome = outcome


def run_shot_job(
    store: JobStore,
    path: Path,
    clip_ids: list[str] | None,
    progress: Callable[[float, str], None],
    cancel: Event,
    *,
    options: ShotTypeOptions | None = None,
    force: bool = False,
    thumbnail_paths: dict[str, Path] | None = None,
    operation: OperationSpec | None = None,
    atomic: bool = False,
) -> dict:
    """Reuse recorded computation after failed saves.

    Atomic callers retain computation but publish nothing on provider failure or
    cancellation. Missing thumbnails and unknown classifications remain skips
    for the legacy synchronous adapter's all-or-nothing save contract.
    """
    options = (
        ShotTypeOptions(**json.loads(operation.inputs_json)["options"])
        if operation
        else (options or ShotTypeOptions())
    )
    if operation is not None and thumbnail_paths:
        raise ValueError("Queued shots inputs cannot replace thumbnails")
    if operation is not None:
        force = bool(operation.arguments.get("force", False))
        atomic = bool(operation.arguments.get("atomic", atomic))
    thumbnails = dict(thumbnail_paths or {})
    runtime = _runtime()
    media_fingerprints = MediaFingerprints(cancel)
    fingerprint = media_fingerprints.get
    fingerprints = AnalysisFingerprints(cancel, media_fingerprints=media_fingerprints)
    with result_batch(store, path) as batch:
        project = batch.project
        if operation is not None:
            revision = project.session.file_revision
            if operation.input_revision is not None and (
                revision is None or revision.digest != operation.input_revision
            ):
                raise ProjectRevisionConflict(path)
            live = shot_job_spec(
                project, clip_ids, options, arguments=operation.arguments
            )
            if live.inputs_json != operation.inputs_json:
                raise StaleJobResult("Shot classification inputs changed while queued")
        ids = _ids(project, clip_ids)
        if any(cid not in ids for cid in thumbnails):
            raise ValueError("Thumbnail belongs to an unselected clip")
        if force:
            batch.max_items = max(1, len(ids))
        known: dict[str, list[tuple[dict, dict, dict]]] = {}
        for rid, digest in project.metadata.job_results.items():
            row = store.get_result(rid)
            if row is None:
                continue  # Verified project records survive job-cache removal.
            if sha256(row["spec_json"].encode()).hexdigest() != rid:
                raise StaleJobResult("Committed result identity is corrupt")
            identity = json.loads(row["spec_json"])
            if identity["kind"] != "shots":
                continue
            if (
                sha256(row["payload_json"].encode()).hexdigest() != digest
                or row["payload_digest"] != digest
            ):
                raise StaleJobResult("Committed shots payload is corrupt")
            known.setdefault(identity["target_id"], []).append(
                (row, identity, json.loads(row["payload_json"]))
            )
        if atomic:
            # Even reconciliation may stage several receipts for one target.
            # Keep every possible receipt below the automatic flush threshold.
            batch.max_items = len(ids) + sum(len(rows) for rows in known.values()) + 1

        def inputs(
            current: Project, cid: str, task: ShotTypeTask | None = None
        ) -> dict:
            task = task or _task(current, cid, thumbnails, force=force)
            display = current.clips_by_id[cid].thumbnail_path
            source = _source_data(current, cid)
            source_path = source["actual_source_path"]
            return {
                "project_id": current.metadata.id,
                "display_thumbnail_path": str(display) if display else None,
                **source,
                "task": _task_data(task),
                "image": fingerprint(task.thumbnail_path),
                "source": fingerprint(Path(source_path)) if source_path else None,
                "runtime": _runtime(),
            }

        def decode(cid: str, payload: dict) -> ShotTypeOutcome:
            if set(payload) not in (
                {"shot_type", "confidence"},
                {"shot_type", "confidence", "record_json"},
            ):
                raise StaleJobResult("Invalid recorded shot classification")
            outcome = ShotTypeOutcome(cid, "succeeded", **payload)
            if not outcome.valid_result():
                raise StaleJobResult("Invalid recorded shot classification")
            return outcome

        def is_output(current: Project, cid: str, payload: dict) -> bool:
            record = current.clips_by_id[cid].analysis_records.get("shots")
            return (
                isinstance(record, AnalysisRecord)
                and record.to_dict() == json.loads(payload.get("record_json") or "null")
                and current.clips_by_id[cid].shot_type == decode(cid, payload).shot_type
            )

        def stage_record(cid: str, record: AnalysisRecord, basis: dict) -> None:
            batch.stage_analysis(
                apply=lambda current: current.record_analysis(
                    "clip", cid, "shots", record
                ),
                validate_input=lambda current: inputs(current, cid) == basis,
                is_applied=lambda current: current.clips_by_id[
                    cid
                ].analysis_records.get("shots")
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
            clip = project.clips_by_id[cid]
            existing = clip.shot_type is not None
            try:
                basis = inputs(project, cid)
                if basis["runtime"] != runtime:
                    raise StaleJobResult("Shot classification runtime changed")
                task = _task(project, cid, thumbnails, force=force)
                assert task.analysis_json is not None
                snapshot = AnalysisSnapshot.from_json(task.analysis_json)
                semantic = (
                    shot_identity(snapshot, options, fingerprints, runtime)
                    if snapshot.inputs.unchanged()
                    else None
                )
                reused = (
                    snapshot.reusable_record(semantic)
                    if semantic is not None and not force
                    else None
                )
                prior_record = clip.analysis_records.get("shots")
                identity_inputs: dict = {
                    "basis": basis,
                    "previous_shot_type": clip.shot_type,
                    "previous_record": prior_record.to_dict() if prior_record else None,
                }
                if force:
                    identity_inputs["generation"] = len(known.get(cid, []))
                arguments = asdict(options)
                spec = ResultSpec.build(
                    path,
                    kind="shots",
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
                if existing and not force:
                    matches = [
                        row
                        for row, identity, payload in known.get(cid, [])
                        if identity["project_path"] == str(path)
                        and identity["inputs"]["basis"] == basis
                        and identity["arguments"] == arguments
                        and is_output(project, cid, payload)
                        and (reused is not None or not row["committed"])
                    ]
                    if matches:
                        specs = [ResultSpec(path, row["spec_json"]) for row in matches]
                    elif reused is not None:
                        if reused != prior_record:
                            stage_record(cid, reused, basis)
                        result["skipped"].append(
                            {"clip_id": cid, "reason": "valid_analysis"}
                        )
                        continue

                application = ShotTypeApplication(project, (task,), options)

                def compute(task=task):
                    outcome = run_shot_types(
                        (task,),
                        options,
                        cancel_event=cancel,
                        fingerprints=fingerprints,
                        runtime=runtime,
                    )[0]
                    if outcome.status != "succeeded":
                        raise _OutcomeError(outcome)
                    return {
                        "shot_type": outcome.shot_type,
                        "confidence": outcome.confidence,
                        "record_json": outcome.record_json,
                    }

                def apply(current, payload, cid=cid, application=application):
                    outcome = decode(cid, payload)
                    if not application.apply(current, outcome):
                        raise StaleJobResult(
                            "Shot classification target changed during application"
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
                if receipt["applied"]:
                    result["succeeded"].append(
                        {
                            "clip_id": cid,
                            "shot_type": receipt["payload"]["shot_type"],
                        }
                    )
                else:
                    result["skipped"].append(
                        {"clip_id": cid, "reason": "already_committed"}
                    )
            except FingerprintCancelled:
                if atomic:
                    raise
                result["unprocessed"].extend(
                    {"clip_id": rest, "code": "cancelled"} for rest in ids[index:]
                )
                break
            except _OutcomeError as exc:
                outcome = exc.outcome
                if inputs(project, cid) != basis:
                    raise StaleJobResult(
                        "stale_input: Shot classification inputs changed during computation"
                    )
                if atomic and outcome.code not in {
                    "thumbnail_missing",
                    "no_classification",
                }:
                    raise RuntimeError(
                        outcome.message or outcome.code or "Shot classification failed"
                    ) from exc
                if not atomic and outcome.can_apply and outcome.record_json is not None:
                    stage_record(
                        cid,
                        AnalysisRecord.from_dict(json.loads(outcome.record_json)),
                        basis,
                    )
                result[outcome.status].append(
                    {"clip_id": cid, "code": outcome.code, "message": outcome.message}
                )
            progress(
                0.95 * (index + 1) / len(ids),
                f"Classifying shots ({index + 1}/{len(ids)})",
            )
        if atomic and cancel.is_set():
            raise RuntimeError("Shot classification cancelled")
        batch.flush()
        progress(1.0, "Shot classification finished")
        return {"success": True, "result": result}

"""Project-first description receipts shared by CLI and MCP jobs."""

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
from core.operations.description import (
    DescriptionApplication,
    DescriptionOptions,
    DescriptionOutcome,
    DescriptionTask,
    description_task,
    description_identity,
    description_runtime,
    resolve_options,
    run_description,
)
from core.project import Project
from core.project_revision import ProjectRevisionConflict


def _runtime(options: DescriptionOptions) -> dict:
    if options.tier not in ("local", "cpu", "gpu"):
        return {"backend": "cloud"}
    from core.analysis.description import is_mlx_vlm_available
    from core.analysis_model_identity import local_description_runtime

    return local_description_runtime(options.model or "", mlx=is_mlx_vlm_available())


def _arguments(options: DescriptionOptions) -> dict:
    arguments = asdict(options)
    arguments.pop("parallelism")
    return arguments


def _ids(project: Project, clip_ids: list[str] | None) -> list[str]:
    ids = (
        list(dict.fromkeys(clip_ids))
        if clip_ids is not None
        else [c.id for c in project.clips]
    )
    if any(cid not in project.clips_by_id for cid in ids):
        raise ValueError("Unknown description clip ID")
    return ids


def _task(
    project: Project,
    cid: str,
    thumbnails: dict[str, Path] | None = None,
    *,
    force: bool = False,
) -> DescriptionTask:
    clip = project.clips_by_id[cid]
    source = project.sources_by_id.get(clip.source_id)
    image = (thumbnails or {}).get(cid)
    if image is None and not force:
        image = recorded_image_path(clip, source, "describe")
    return description_task(clip, source, image_path=image, skip_existing=False)


def _task_data(task: DescriptionTask) -> dict:
    data = asdict(task)
    data.pop("analysis_json")
    data["skip"] = False
    data["analysis_version"] = 2 if task.analysis_json is not None else 1
    for key in ("thumbnail_path", "source_path"):
        data[key] = str(data[key]) if data[key] is not None else None
    return data


def description_job_spec(
    project: Project,
    clip_ids: list[str] | None,
    options: DescriptionOptions,
    *,
    arguments: dict,
) -> OperationSpec:
    if options.model is None or options.input_mode is None:
        raise ValueError("Description job options must be resolved before submission")
    targets = []
    for cid in _ids(project, clip_ids):
        task = _task(project, cid, force=bool(arguments.get("force", False)))
        targets.append(
            {
                **_task_data(task),
                "runtime": description_runtime(task, options),
                "image_stamp": media_stamp(task.thumbnail_path)
                if task.thumbnail_path
                else None,
                "source_stamp": media_stamp(task.source_path)
                if task.source_path
                else None,
            }
        )
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="describe",
        version=2,
        arguments=arguments,
        inputs={
            "targets": targets,
            "options": asdict(options),
            "runtime": _runtime(options),
        },
        persistence="job_history",
        session_id=project.session.session_id,
        input_revision=revision.digest if revision else None,
    )


class _OutcomeError(Exception):
    def __init__(self, outcome: DescriptionOutcome):
        self.outcome = outcome


def run_description_job(
    store: JobStore,
    path: Path,
    clip_ids: list[str] | None,
    progress: Callable[[float, str], None],
    cancel: Event,
    *,
    options: DescriptionOptions | None = None,
    force: bool = False,
    operation: OperationSpec | None = None,
    thumbnail_paths: dict[str, Path] | None = None,
) -> dict:
    """Reuse completed computation after failed saves, without implicit provider calls."""
    if operation is not None and thumbnail_paths:
        raise ValueError("Queued description inputs cannot replace their thumbnails")
    if operation is not None:
        force = bool(operation.arguments.get("force", False))
    options = (
        DescriptionOptions(**json.loads(operation.inputs_json)["options"])
        if operation
        else (options or resolve_options())
    )
    if options.model is None or options.input_mode is None:
        raise ValueError("Description job options must be resolved")
    thumbnails = dict(thumbnail_paths or {})
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
            live = description_job_spec(
                project, clip_ids, options, arguments=operation.arguments
            )
            if live.inputs_json != operation.inputs_json:
                raise StaleJobResult("Description inputs changed while queued")
        ids = _ids(project, clip_ids)
        # CLI prepares high-resolution thumbnails before entering the shared runner.
        for cid in thumbnails:
            if cid not in ids:
                raise ValueError("Thumbnail belongs to an unselected clip")
        if force:
            batch.max_items = max(1, len(ids))
        known: dict[str, list[tuple[dict, dict, dict]]] = {}
        for rid, digest in project.metadata.job_results.items():
            row = store.get_result(rid)
            if row is None:
                continue
            if sha256(row["spec_json"].encode()).hexdigest() != rid:
                raise StaleJobResult("Committed result identity is corrupt")
            identity = json.loads(row["spec_json"])
            if identity["kind"] != "describe":
                continue
            if (
                sha256(row["payload_json"].encode()).hexdigest() != digest
                or row["payload_digest"] != digest
            ):
                raise StaleJobResult("Committed description payload is corrupt")
            known.setdefault(identity["target_id"], []).append(
                (row, identity, json.loads(row["payload_json"]))
            )

        def inputs(current: Project, cid: str) -> dict:
            task = _task(current, cid, thumbnails, force=force)
            display = current.clips_by_id[cid].thumbnail_path
            return {
                "project_id": current.metadata.id,
                "source_id": current.clips_by_id[cid].source_id,
                "display_thumbnail_path": str(display) if display else None,
                "task": _task_data(task),
                "image": fingerprint(task.thumbnail_path),
                "source": fingerprint(task.source_path),
                "runtime": description_runtime(task, options),
            }

        def output(current: Project, cid: str) -> dict:
            clip = current.clips_by_id[cid]
            return {
                "description": clip.description,
                "model": clip.description_model,
                "frames": clip.description_frames,
                "record_json": json.dumps(
                    clip.analysis_records["describe"].to_dict(), sort_keys=True
                )
                if "describe" in clip.analysis_records
                else None,
            }

        def stage_record(cid: str, record: AnalysisRecord, basis: dict) -> None:
            batch.stage_analysis(
                apply=lambda current: current.record_analysis(
                    "clip", cid, "describe", record
                ),
                validate_input=lambda current: inputs(current, cid) == basis,
                is_applied=lambda current: current.clips_by_id[
                    cid
                ].analysis_records.get("describe")
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
            existing = clip.description is not None
            try:
                basis = inputs(project, cid)
                task = _task(project, cid, thumbnails, force=force)
                assert task.analysis_json is not None
                snapshot = AnalysisSnapshot.from_json(task.analysis_json)
                semantic = (
                    description_identity(
                        snapshot, options, fingerprints, basis["runtime"]
                    )
                    if snapshot.inputs.unchanged()
                    else None
                )
                reused = (
                    snapshot.reusable_record(semantic)
                    if semantic is not None and not force
                    else None
                )
                prior_record = clip.analysis_records.get("describe")
                identity_inputs: dict = {
                    "basis": basis,
                    "previous_description": output(project, cid),
                }
                if force:
                    identity_inputs["refresh_generation"] = len(known.get(cid, []))
                arguments = _arguments(options)
                spec = ResultSpec.build(
                    path,
                    kind="describe",
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
                            and payload == output(project, cid)
                        ):
                            specs = [ResultSpec(path, row["spec_json"])]
                            break
                if existing and not force:
                    matches = [
                        row
                        for row, identity, payload in known.get(cid, [])
                        if reused is not None
                        and identity["project_path"] == str(path)
                        and identity["inputs"]["basis"] == basis
                        and identity["arguments"] == arguments
                        and payload == output(project, cid)
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

                application = DescriptionApplication(project, (task,), options)

                def compute(task=task):
                    outcome = run_description(
                        (task,), options, cancel_event=cancel, fingerprints=fingerprints
                    )[0]
                    if outcome.status != "succeeded":
                        raise _OutcomeError(outcome)
                    return {
                        "description": outcome.description,
                        "model": outcome.model,
                        "frames": 1,
                        "record_json": outcome.record_json,
                    }

                def apply(current, payload, cid=cid, application=application):
                    outcome = DescriptionOutcome(
                        cid,
                        "succeeded",
                        payload["description"],
                        payload["model"],
                        record_json=payload.get("record_json"),
                    )
                    if not application.apply(current, outcome):
                        raise StaleJobResult(
                            "Description target changed during application"
                        )

                def validate(current, cid=cid, basis=basis):
                    return inputs(current, cid) == basis

                def is_applied(current, payload, cid=cid):
                    return output(current, cid) == payload

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
                        {"clip_id": cid, "model": receipt["payload"]["model"]}
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
                    raise StaleJobResult(
                        "Description inputs changed during computation"
                    )
                if outcome.can_apply and outcome.record_json is not None:
                    stage_record(
                        cid,
                        AnalysisRecord.from_dict(json.loads(outcome.record_json)),
                        basis,
                    )
                result[outcome.status].append(
                    {"clip_id": cid, "code": outcome.code, "message": outcome.message}
                )
            progress(
                0.95 * (index + 1) / len(ids), f"Describing ({index + 1}/{len(ids)})"
            )
        batch.flush()
        progress(1.0, "Descriptions finished")
        return {"success": True, "result": result}

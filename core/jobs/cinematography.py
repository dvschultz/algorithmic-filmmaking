"""Durable cinematography results for saved-project jobs."""

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
from core.operations.cinematography import (
    CinematographyApplication,
    CinematographyOptions,
    CinematographyOutcome,
    CinematographyTask,
    resolve_options,
    run_cinematography,
)
from core.project import Project
from core.project_revision import ProjectRevisionConflict


def _runtime(options: CinematographyOptions) -> dict:
    if options.tier != "local":
        return {"backend": "cloud"}
    from core.analysis.description import is_mlx_vlm_available

    return {"backend": "mlx" if is_mlx_vlm_available() else "unavailable"}


def _ids(project: Project, clip_ids: list[str] | None) -> list[str]:
    ids = (
        list(dict.fromkeys(clip_ids))
        if clip_ids is not None
        else [c.id for c in project.clips]
    )
    if any(cid not in project.clips_by_id for cid in ids):
        raise ValueError("Unknown cinematography clip ID")
    return ids


def _task(project: Project, cid: str) -> CinematographyTask:
    clip = project.clips_by_id[cid]
    source = project.sources_by_id.get(clip.source_id)
    return CinematographyTask(
        cid,
        clip.thumbnail_path,
        source.file_path if source and source.file_path.exists() else None,
        clip.start_frame,
        clip.end_frame,
        source.fps if source else None,
    )


def _task_data(task: CinematographyTask) -> dict:
    data = asdict(task)
    for key in ("thumbnail_path", "source_path"):
        data[key] = str(data[key]) if data[key] is not None else None
    return data


def _source_data(project: Project, cid: str) -> dict:
    source_id = project.clips_by_id[cid].source_id
    source = project.sources_by_id.get(source_id)
    return {
        "source_id": source_id,
        "actual_source_path": str(source.file_path) if source else None,
    }


def cinematography_job_spec(
    project: Project,
    clip_ids: list[str] | None,
    options: CinematographyOptions,
    *,
    arguments: dict,
) -> OperationSpec:
    targets = []
    for cid in _ids(project, clip_ids):
        task = _task(project, cid)
        targets.append(
            {
                **_task_data(task),
                **_source_data(project, cid),
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
        kind="cinematography",
        version=1,
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
    def __init__(self, outcome: CinematographyOutcome):
        self.outcome = outcome


def run_cinematography_job(
    store: JobStore,
    path: Path,
    clip_ids: list[str] | None,
    progress: Callable[[float, str], None],
    cancel: Event,
    *,
    options: CinematographyOptions | None = None,
    operation: OperationSpec | None = None,
) -> dict:
    """Reuse completed computation after failed saves, without implicit provider calls."""
    options = (
        CinematographyOptions(**json.loads(operation.inputs_json)["options"])
        if operation
        else (options or resolve_options())
    )
    runtime = _runtime(options)
    fingerprint = MediaFingerprints(cancel).get
    with result_batch(store, path) as batch:
        project = batch.project
        if operation is not None:
            revision = project.session.file_revision
            if operation.input_revision is not None and (
                revision is None or revision.digest != operation.input_revision
            ):
                raise ProjectRevisionConflict(path)
            live = cinematography_job_spec(
                project, clip_ids, options, arguments=operation.arguments
            )
            if live.inputs_json != operation.inputs_json:
                raise StaleJobResult("Cinematography inputs changed while queued")
        ids = _ids(project, clip_ids)
        known: dict[str, list[tuple[dict, dict, dict]]] = {}
        for rid, digest in project.metadata.job_results.items():
            row = store.get_result(rid)
            if row is None:
                raise StaleJobResult("Committed result payload is missing")
            if sha256(row["spec_json"].encode()).hexdigest() != rid:
                raise StaleJobResult("Committed result identity is corrupt")
            identity = json.loads(row["spec_json"])
            if identity["kind"] != "cinematography":
                continue
            if (
                sha256(row["payload_json"].encode()).hexdigest() != digest
                or row["payload_digest"] != digest
            ):
                raise StaleJobResult("Committed cinematography payload is corrupt")
            known.setdefault(identity["target_id"], []).append(
                (row, identity, json.loads(row["payload_json"]))
            )

        def inputs(current: Project, cid: str) -> dict:
            task = _task(current, cid)
            return {
                "project_id": current.metadata.id,
                **_source_data(current, cid),
                "task": _task_data(task),
                "image": fingerprint(task.thumbnail_path),
                "source": fingerprint(task.source_path),
                "runtime": _runtime(options),
            }

        def output(current: Project, cid: str) -> dict:
            clip = current.clips_by_id[cid]
            return {
                "analysis": clip.cinematography.to_dict()
                if clip.cinematography
                else None,
                "shot_type": clip.shot_type,
            }

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
            existing = clip.cinematography is not None
            if existing and cid not in known:
                result["skipped"].append(
                    {"clip_id": cid, "reason": "already_populated"}
                )
                continue
            try:
                basis = inputs(project, cid)
                if basis["runtime"] != runtime:
                    raise StaleJobResult("Cinematography runtime changed")
                task = _task(project, cid)
                identity_inputs: dict = {"basis": basis}
                arguments = asdict(options)
                spec = ResultSpec.build(
                    path,
                    kind="cinematography",
                    version=1,
                    target_id=cid,
                    arguments=arguments,
                    inputs=identity_inputs,
                )
                specs = [spec]
                if existing:
                    matches = [
                        row
                        for row, identity, payload in known[cid]
                        if identity["project_path"] == str(path)
                        and identity["inputs"]["basis"] == basis
                        and identity["arguments"] == arguments
                        and payload == output(project, cid)
                    ]
                    if not matches:
                        result["skipped"].append(
                            {"clip_id": cid, "reason": "already_populated"}
                        )
                        continue
                    specs = [ResultSpec(path, row["spec_json"]) for row in matches]

                application = CinematographyApplication(project, (task,))

                def compute(task=task):
                    outcome = run_cinematography((task,), options, cancel_event=cancel)[
                        0
                    ]
                    if outcome.status != "succeeded":
                        raise _OutcomeError(outcome)
                    analysis = outcome.analysis
                    assert analysis is not None
                    return {
                        "analysis": analysis.to_dict(),
                        "shot_type": analysis.get_simple_shot_type(),
                    }

                def apply(current, payload, cid=cid, application=application):
                    outcome = CinematographyOutcome(
                        cid, "succeeded", json.dumps(payload["analysis"])
                    )
                    if not application.apply(current, outcome):
                        raise StaleJobResult(
                            "Cinematography target changed during application"
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
                        {
                            "clip_id": cid,
                            "shot_size": receipt["payload"]["analysis"]["shot_size"],
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
            progress(
                0.95 * (index + 1) / len(ids),
                f"Cinematography ({index + 1}/{len(ids)})",
            )
        batch.flush()
        progress(1.0, "Cinematography finished")
        return {"success": True, "result": result}

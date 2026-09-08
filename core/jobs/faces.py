"""Durable faces results for saved-project jobs."""

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
from core.operations.faces import (
    FaceApplication,
    FaceOptions,
    FaceOutcome,
    FaceTask,
    Face,
    run_faces,
    face_model_session,
)
from core.project import Project
from core.project_revision import ProjectRevisionConflict


def _runtime() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    packages: dict[str, str | None] = {}
    for package in ("insightface", "onnxruntime", "onnxruntime-gpu"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "model": "buffalo_l",
        "embedding": "ArcFace-512",
        "packages": packages,
    }


def _ids(project: Project, clip_ids: list[str] | None) -> list[str]:
    ids = (
        list(dict.fromkeys(clip_ids))
        if clip_ids is not None
        else [c.id for c in project.clips]
    )
    if any(cid not in project.clips_by_id for cid in ids):
        raise ValueError("Unknown face analysis clip ID")
    return ids


def _task(project: Project, cid: str) -> FaceTask:
    clip = project.clips_by_id[cid]
    source = project.sources_by_id.get(clip.source_id)
    return FaceTask(
        cid,
        clip.source_id,
        source.file_path if source else None,
        clip.start_frame,
        clip.end_frame,
        source.fps if source else 0.0,
    )


def _task_data(task: FaceTask) -> dict:
    data = asdict(task)
    if task.analysis_json is None:
        # Preserve existing version-1 receipt identities for raw callers.
        data.pop("analysis_json")
    data["source_path"] = str(task.source_path) if task.source_path else None
    return data


def _saved_faces(faces: list[dict]) -> list[dict]:
    """Match the existing project serializer without rounding recorded computation."""
    return [
        {**face, "embedding": [round(value, 5) for value in face["embedding"]]}
        for face in faces
    ]


def face_job_spec(
    project: Project,
    clip_ids: list[str] | None,
    options: FaceOptions,
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
        kind="faces",
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
    def __init__(self, outcome: FaceOutcome):
        self.outcome = outcome


def run_face_job(
    store: JobStore,
    path: Path,
    clip_ids: list[str] | None,
    progress: Callable[[float, str], None],
    cancel: Event,
    *,
    options: FaceOptions | None = None,
    force: bool = False,
    operation: OperationSpec | None = None,
) -> dict:
    """Reuse completed computation after failed saves, without implicit provider calls."""
    options = (
        FaceOptions(**json.loads(operation.inputs_json)["options"])
        if operation
        else (options or FaceOptions())
    )
    if operation is not None:
        force = bool(operation.arguments.get("force", False))
    runtime = _runtime()
    fingerprint = MediaFingerprints(cancel).get
    with face_model_session() as model_session, result_batch(store, path) as batch:
        project = batch.project
        if operation is not None:
            revision = project.session.file_revision
            if operation.input_revision is not None and (
                revision is None or revision.digest != operation.input_revision
            ):
                raise ProjectRevisionConflict(path)
            live = face_job_spec(
                project, clip_ids, options, arguments=operation.arguments
            )
            if live.inputs_json != operation.inputs_json:
                raise StaleJobResult("Face analysis inputs changed while queued")
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
            if identity["kind"] != "faces":
                continue
            if (
                sha256(row["payload_json"].encode()).hexdigest() != digest
                or row["payload_digest"] != digest
            ):
                raise StaleJobResult("Committed faces payload is corrupt")
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
            existing = current.clips_by_id[cid].face_embeddings
            return existing == payload["faces"] or existing == _saved_faces(
                payload["faces"]
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
            existing = clip.face_embeddings is not None
            if existing and not force and cid not in known:
                result["skipped"].append(
                    {"clip_id": cid, "reason": "already_populated"}
                )
                continue
            try:
                basis = inputs(project, cid)
                if basis["runtime"] != runtime:
                    raise StaleJobResult("Face analysis runtime changed")
                task = _task(project, cid)
                identity_inputs: dict = {"basis": basis}
                if force:
                    identity_inputs["generation"] = len(known.get(cid, []))
                    identity_inputs["previous_faces"] = clip.face_embeddings
                arguments = asdict(options)
                spec = ResultSpec.build(
                    path,
                    kind="faces",
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
                if existing and not force:
                    matches = [
                        row
                        for row, identity, payload in known[cid]
                        if identity["project_path"] == str(path)
                        and identity["inputs"]["basis"] == basis
                        and identity["arguments"] == arguments
                        and is_output(project, cid, payload)
                    ]
                    if not matches:
                        result["skipped"].append(
                            {"clip_id": cid, "reason": "already_populated"}
                        )
                        continue
                    specs = [ResultSpec(path, row["spec_json"]) for row in matches]

                application = FaceApplication(project, (task,))

                def compute(task=task):
                    outcome = run_faces(
                        (task,),
                        options,
                        cancel_event=cancel,
                        model_session=model_session,
                    )[0]
                    if outcome.status != "succeeded":
                        raise _OutcomeError(outcome)
                    return {"faces": outcome.face_dicts()}

                def apply(current, payload, cid=cid, application=application):
                    outcome = FaceOutcome(
                        cid,
                        "succeeded",
                        tuple(Face.from_dict(face) for face in payload["faces"]),
                    )
                    if not application.apply(current, outcome):
                        raise StaleJobResult(
                            "Face analysis target changed during application"
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
                            "face_count": len(receipt["payload"]["faces"]),
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
                f"Face analysis ({index + 1}/{len(ids)})",
            )
        batch.flush()
        progress(1.0, "Face analysis finished")
        return {"success": True, "result": result}

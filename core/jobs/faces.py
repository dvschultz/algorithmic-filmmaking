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
    face_task,
)
from core.operations import faces as face_operations
from core.operations.face_records import (
    face_target_runtime,
    face_target_matches,
    face_value,
    face_result_value,
    face_runtime,
    saved_execution,
    reusable_face_record,
    FACE_SAMPLING,
    face_parameters,
)
from core.analysis_records import AnalysisFingerprints, AnalysisInput, AnalysisSnapshot
from models.analysis_record import AnalysisRecord
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
    return face_task(clip, source, skip_existing=False)


def _job_task_data(task: FaceTask) -> dict:
    data = _task_data(task)
    data.pop("analysis_json", None)
    data["analysis_version"] = 2
    return data


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
                **_job_task_data(task),
                "source_stamp": media_stamp(task.source_path)
                if task.source_path
                else None,
            }
        )
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="faces",
        version=2,
        arguments=arguments,
        inputs={
            "targets": targets,
            "options": asdict(options),
            "runtime": face_target_runtime(),
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
    """Verify saved records and recover authenticated computation before saving."""
    options = (
        FaceOptions(**json.loads(operation.inputs_json)["options"])
        if operation
        else (options or FaceOptions())
    )
    if operation is not None:
        force = bool(operation.arguments.get("force", False))
    target_runtime = face_target_runtime()
    media = MediaFingerprints(cancel)
    fingerprints = AnalysisFingerprints(cancel, media_fingerprints=media)
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
                continue
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
            target = face_target_runtime()
            return {
                "project_id": current.metadata.id,
                "task": _job_task_data(task),
                "source": media.get(task.source_path),
                "runtime": {
                    key: value for key, value in target.items() if key != "files"
                },
            }

        def current_output(current: Project, cid: str) -> dict:
            from dataclasses import replace
            from core.artifacts import ArtifactStore, ArtifactUnavailable

            clip = current.clips_by_id[cid]
            record = clip.analysis_records.get("face_embeddings")
            if isinstance(record, AnalysisRecord) and record.artifact is not None:
                try:
                    value = json.loads(ArtifactStore().read_bytes(record.artifact))
                    if value == face_value(clip):
                        record = replace(record, artifact=None, value_json=json.dumps(value, sort_keys=True))
                except (ArtifactUnavailable, OSError, ValueError):
                    pass  # A missing payload cannot prove prior publication.
            return {
                "faces": face_value(clip)["face_embeddings"],
                "record": record.to_dict() if record else None,
            }

        def payload_output(payload: dict) -> dict:
            return {
                "faces": face_result_value(payload["faces"])["face_embeddings"],
                "record": json.loads(payload["record_json"])
                if payload.get("record_json")
                else None,
            }

        def payload_is_current(payload: dict) -> bool:
            if not payload.get("record_json"):
                return False
            try:
                record = AnalysisRecord.from_dict(json.loads(payload["record_json"]))
                if (
                    record.identity is None
                    or record.state != "succeeded"
                    or record.value != face_result_value(payload["faces"])
                ):
                    return False
                bound = AnalysisInput.from_dict(json.loads(record.input_json or "null"))
                environment = face_operations.face_environment()
                runtime = face_runtime(saved_execution(record), environment)
                candidate = fingerprints.identity(
                    bound,
                    operation="face_embeddings",
                    operation_version=2,
                    model=runtime,
                    parameters=face_parameters(options.sample_interval),
                    sampling=FACE_SAMPLING,
                )
                return candidate == record.identity
            except (ValueError, TypeError, KeyError, AttributeError, OSError):
                return False

        def stage_outcome(
            cid: str, outcome: FaceOutcome, application: FaceApplication, basis: dict
        ) -> None:
            record = AnalysisRecord.from_dict(json.loads(outcome.record_json or "null"))
            published = False

            def publish(current: Project) -> None:
                nonlocal published
                if not application.apply(current, outcome):
                    raise StaleJobResult("Face analysis target changed")
                published = True

            def validate(current: Project) -> bool:
                if cancel.is_set() and not published:
                    raise FingerprintCancelled()
                return inputs(current, cid) == basis

            def applied(current: Project) -> bool:
                return (
                    current.clips_by_id[cid].analysis_records.get("face_embeddings")
                    == record
                )

            batch.stage_analysis(
                apply=publish, validate_input=validate, is_applied=applied
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
            current_clip = project.clips_by_id[cid]
            current_record = current_clip.analysis_records.get("face_embeddings")
            missing_payload = (
                current_clip.face_embeddings is None
                and isinstance(current_record, AnalysisRecord)
                and current_record.artifact is not None
                and current_record.state == "missing"
            )
            if (
                not force
                and not missing_payload
                and cid in known
                and not any(
                    current_output(project, cid)["faces"]
                    == payload_output(payload)["faces"]
                    for _, _, payload in known[cid]
                )
            ):
                raise StaleJobResult(
                    "Previously committed faces changed; use force to replace edited faces"
                )
            try:
                if not face_target_matches(target_runtime, face_target_runtime()):
                    raise StaleJobResult(
                        "Face model weights or runtime changed during the job"
                    )
                basis = inputs(project, cid)
                task = _task(project, cid)
                application = FaceApplication(project, (task,), options)
                snapshot = AnalysisSnapshot.from_json(task.analysis_json or "")
                reused = None
                if not force and snapshot.inputs.unchanged():
                    try:
                        reused = reusable_face_record(
                            snapshot,
                            options.sample_interval,
                            fingerprints,
                            face_operations.face_environment(),
                        )
                    except ImportError:
                        pass  # The shared compute path retains a missing-runtime failure.
                identity_inputs = {
                    "basis": basis,
                    "previous_output": current_output(project, cid),
                    "weights": face_target_runtime()["files"],
                }
                if force:
                    identity_inputs["generation"] = len(known.get(cid, []))
                arguments = asdict(options)
                spec = ResultSpec.build(
                    path,
                    kind="faces",
                    version=2,
                    target_id=cid,
                    arguments=arguments,
                    inputs=identity_inputs,
                )
                specs = [spec]
                matches = [
                    row
                    for row, identity, payload in known.get(cid, [])
                    if (not force or not row["committed"])
                    and identity["project_path"] == str(path)
                    and identity["arguments"] == arguments
                    and identity["inputs"]["basis"] == basis
                    and current_output(project, cid) == payload_output(payload)
                    and payload_is_current(payload)
                ]
                if matches and (force or reused is not None):
                    specs = [ResultSpec(path, row["spec_json"]) for row in matches]
                elif reused is not None:
                    outcome = FaceOutcome(
                        cid,
                        "skipped",
                        tuple(
                            Face.from_dict(f) for f in reused.value["face_embeddings"]
                        ),
                        code="valid_analysis",
                        record_json=json.dumps(reused.to_dict(), sort_keys=True),
                    )
                    stage_outcome(cid, outcome, application, basis)
                    result["skipped"].append(
                        {"clip_id": cid, "reason": "valid_analysis"}
                    )
                    continue

                # Initial download can fill an absent model pack before a save fails.
                if (
                    specs == [spec]
                    and store.get_result(spec.result_id) is None
                    and identity_inputs["weights"]
                ):
                    initial = ResultSpec.build(
                        path,
                        kind="faces",
                        version=2,
                        target_id=cid,
                        arguments=arguments,
                        inputs={**identity_inputs, "weights": []},
                    )
                    row = store.get_result(initial.result_id)
                    if row is not None:
                        if (
                            row["spec_json"] != initial.identity_json
                            or sha256(row["payload_json"].encode()).hexdigest()
                            != row["payload_digest"]
                        ):
                            raise StaleJobResult("Initial face computation is corrupt")
                        if payload_is_current(json.loads(row["payload_json"])):
                            specs = [initial]

                def compute(task=task):
                    outcome = run_faces(
                        (task,),
                        options,
                        cancel_event=cancel,
                        model_session=model_session,
                        fingerprints=fingerprints,
                    )[0]
                    if outcome.status != "succeeded":
                        raise _OutcomeError(outcome)
                    return {
                        "faces": outcome.face_dicts(),
                        "record_json": outcome.record_json,
                    }

                def apply(current, payload, cid=cid, application=application):
                    if not payload_is_current(payload):
                        raise StaleJobResult(
                            "Face record inputs changed before publication"
                        )
                    outcome = FaceOutcome(
                        cid,
                        "succeeded",
                        tuple(Face.from_dict(f) for f in payload["faces"]),
                        record_json=payload["record_json"],
                    )
                    if not application.apply(current, outcome):
                        raise StaleJobResult(
                            "Face analysis target changed during application"
                        )

                def validate(
                    current,
                    cid=cid,
                    basis=basis,
                    specs=tuple(specs),
                    expected_runtime=target_runtime,
                ):
                    if cancel.is_set() and not any(
                        s.result_id in current.metadata.job_results for s in specs
                    ):
                        raise FingerprintCancelled()
                    return inputs(current, cid) == basis and face_target_matches(
                        expected_runtime, face_target_runtime()
                    )

                def is_applied(current, payload, cid=cid):
                    return current_output(current, cid) == payload_output(
                        payload
                    ) and payload_is_current(payload)

                for candidate in specs:
                    receipt = batch.commit(
                        candidate,
                        compute=compute,
                        validate_input=validate,
                        apply=apply,
                        is_applied=is_applied,
                    )
                target_runtime = face_target_runtime()
                if receipt["applied"]:
                    result["succeeded"].append(
                        {"clip_id": cid, "face_count": len(receipt["payload"]["faces"])}
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
                        "Face analysis inputs changed during computation"
                    )
                if outcome.can_apply:
                    stage_outcome(cid, outcome, application, basis)
                result[outcome.status].append(
                    {"clip_id": cid, "code": outcome.code, "message": outcome.message}
                )
            progress(
                0.95 * (index + 1) / len(ids), f"Face analysis ({index + 1}/{len(ids)})"
            )
        batch.flush()
        progress(1.0, "Face analysis finished")
        return {"success": True, "result": result}

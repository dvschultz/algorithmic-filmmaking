"""Durable first/last-frame embedding pairs for saved projects."""

from hashlib import sha256
from dataclasses import replace
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.commits import ResultSpec, StaleJobResult, result_batch, canonical_json
from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot
from core.analysis_model_identity import boundary_embedding_runtime
from models.analysis_record import AnalysisRecord
from core.jobs.media import FingerprintCancelled, MediaFingerprints, media_stamp
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore
from core.operations.boundary_embeddings import (
    BoundaryEmbeddingOutcome,
    BoundaryEmbeddingApplication,
    boundary_embedding_task,
    boundary_embedding_identity,
    run_boundary_embeddings,
)
from core.operations.embeddings import EmbeddingOutcome, embedding_model_session
from core.project import Project
from core.project_revision import ProjectRevisionConflict


def _runtime() -> dict:
    return boundary_embedding_runtime()


def _ids(project: Project, clip_ids: list[str] | None) -> list[str]:
    ids = (
        list(dict.fromkeys(clip_ids))
        if clip_ids is not None
        else [c.id for c in project.clips]
    )
    if any(cid not in project.clips_by_id for cid in ids):
        raise ValueError("Unknown boundary embedding clip ID")
    return ids


def _target(project: Project, cid: str) -> dict:
    clip = project.clips_by_id[cid]
    source = project.sources_by_id.get(clip.source_id)
    return {
        "analysis_version": 2,
        "clip_id": cid,
        "source_id": clip.source_id,
        "source_path": str(source.file_path) if source else None,
        "start_frame": clip.start_frame,
        "end_frame": clip.end_frame,
        "fps": source.fps if source else 0.0,
    }


def _values(project: Project, cid: str) -> dict:
    clip = project.clips_by_id[cid]
    record = clip.analysis_records.get("boundary_embeddings")
    if isinstance(record, AnalysisRecord) and record.artifact is not None:
        if record.state == "succeeded":
            record = replace(
                record,
                artifact=None,
                value_json=canonical_json(
                    {
                        "first_frame_embedding": clip.first_frame_embedding,
                        "last_frame_embedding": clip.last_frame_embedding,
                        "embedding_model": clip.embedding_model,
                    }
                ),
            )
        elif record.state == "failed":
            record = replace(record, artifact=None, value_json=None)
    return {
        "first": clip.first_frame_embedding,
        "last": clip.last_frame_embedding,
        "model": clip.embedding_model,
        "record_json": json.dumps(record.to_dict(), sort_keys=True)
        if isinstance(record, AnalysisRecord)
        else None,
    }


def _validated(cid: str, payload: dict) -> dict:
    first = EmbeddingOutcome.from_vector(cid, payload["first"])
    last = EmbeddingOutcome.from_vector(cid, payload["last"])
    if payload["model"] != first.model:
        raise ValueError("Boundary embedding model identity does not match")
    return {
        "first": list(first.vector),
        "last": list(last.vector),
        "model": first.model,
        "record_json": payload.get("record_json"),
    }


def boundary_embedding_job_spec(
    project: Project, clip_ids: list[str] | None, *, arguments: dict
) -> OperationSpec:
    targets = []
    for cid in _ids(project, clip_ids):
        target = _target(project, cid)
        targets.append(
            {
                **target,
                "source_stamp": media_stamp(Path(target["source_path"]))
                if target["source_path"]
                else None,
            }
        )
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="boundary_embeddings",
        version=2,
        arguments=arguments,
        inputs={"targets": targets, "runtime": _runtime()},
        persistence="job_history",
        session_id=project.session.session_id,
        input_revision=revision.digest if revision else None,
    )


class _OutcomeError(Exception):
    def __init__(
        self,
        status: str,
        code: str | None,
        message: str | None = None,
        record_json: str | None = None,
    ):
        self.status, self.code, self.message = status, code, message
        self.record_json = record_json


def run_boundary_embedding_job(
    store: JobStore,
    path: Path,
    clip_ids: list[str] | None,
    progress: Callable[[float, str], None],
    cancel: Event,
    *,
    force: bool = False,
    operation: OperationSpec | None = None,
) -> dict:
    """Journal validated pairs before applying them to a private project writer."""
    path = path.expanduser().resolve()
    if operation is not None:
        force = bool(operation.arguments.get("force", False))
    runtime = _runtime()
    media_fingerprints = MediaFingerprints(cancel)
    fingerprint = media_fingerprints.get
    fingerprints = AnalysisFingerprints(cancel, media_fingerprints=media_fingerprints)
    with embedding_model_session() as model_session, result_batch(store, path) as batch:
        project = batch.project
        if operation is not None:
            revision = project.session.file_revision
            if operation.input_revision is not None and (
                revision is None or revision.digest != operation.input_revision
            ):
                raise ProjectRevisionConflict(path)
            live = boundary_embedding_job_spec(
                project, clip_ids, arguments=operation.arguments
            )
            if live.inputs_json != operation.inputs_json:
                raise StaleJobResult("Boundary embedding inputs changed while queued")
        ids = _ids(project, clip_ids)
        if force:
            batch.max_items = max(1, len(ids))
        known: dict[str, list[tuple[dict, dict, dict]]] = {}
        for rid, digest in project.metadata.job_results.items():
            row = store.get_result(rid)
            if row is None:
                continue  # Project records survive removal of old job payloads.
            if sha256(row["spec_json"].encode()).hexdigest() != rid:
                raise StaleJobResult("Committed result identity is corrupt")
            identity = json.loads(row["spec_json"])
            if identity["kind"] != "boundary_embeddings":
                continue
            if (
                sha256(row["payload_json"].encode()).hexdigest() != digest
                or row["payload_digest"] != digest
            ):
                raise StaleJobResult("Committed boundary embedding payload is corrupt")
            payload = _validated(identity["target_id"], json.loads(row["payload_json"]))
            known.setdefault(identity["target_id"], []).append((row, identity, payload))

        def inputs(current: Project, cid: str) -> dict:
            target = _target(current, cid)
            return {
                "project_id": current.metadata.id,
                "target": target,
                "source": fingerprint(
                    Path(target["source_path"]) if target["source_path"] else None
                ),
                "runtime": _runtime(),
            }

        result: dict = {
            "succeeded": [],
            "failed": [],
            "skipped": [],
            "unprocessed": [],
            "total_clips": len(ids),
        }

        def stage_record(cid: str, record: AnalysisRecord, basis: dict) -> None:
            batch.stage_analysis(
                apply=lambda current: current.record_analysis(
                    "clip", cid, "boundary_embeddings", record
                ),
                validate_input=lambda current: inputs(current, cid) == basis,
                is_applied=lambda current: current.clips_by_id[
                    cid
                ].analysis_records.get("boundary_embeddings")
                == record,
            )

        for index, cid in enumerate(ids):
            if cancel.is_set() or model_session.failed:
                result["unprocessed"].extend(
                    {
                        "clip_id": rest,
                        "code": "cancelled" if cancel.is_set() else "embedding_failed",
                    }
                    for rest in ids[index:]
                )
                break
            previous = _values(project, cid)
            try:
                basis = inputs(project, cid)
                if basis["runtime"] != runtime:
                    raise StaleJobResult("Boundary embedding runtime changed")
                clip = project.clips_by_id[cid]
                task = boundary_embedding_task(
                    clip, project.sources_by_id.get(clip.source_id), skip_existing=False
                )
                assert task.analysis_json is not None
                snapshot = AnalysisSnapshot.from_json(task.analysis_json)
                semantic = (
                    boundary_embedding_identity(snapshot, fingerprints, runtime)
                    if snapshot.inputs.unchanged()
                    else None
                )
                reused = (
                    snapshot.reusable_record(semantic)
                    if semantic is not None and not force
                    else None
                )
                identity_inputs: dict = {"basis": basis, "previous": previous}
                if force:
                    identity_inputs.update(
                        generation=len(known.get(cid, [])), previous=previous
                    )
                spec = ResultSpec.build(
                    path,
                    kind="boundary_embeddings",
                    version=2,
                    target_id=cid,
                    arguments={},
                    inputs=identity_inputs,
                )
                matches = [
                    row
                    for row, identity, payload in known.get(cid, [])
                    if identity["project_path"] == str(path)
                    and identity["arguments"] == {}
                    and identity["inputs"]["basis"] == basis
                    and _values(project, cid) == payload
                    and (not force or not row["committed"])
                ]
                if matches:
                    specs = [ResultSpec(path, row["spec_json"]) for row in matches]
                elif reused is not None:
                    if previous["record_json"] != json.dumps(
                        reused.to_dict(), sort_keys=True
                    ):
                        stage_record(cid, reused, basis)
                    result["skipped"].append(
                        {"clip_id": cid, "reason": "valid_analysis"}
                    )
                    continue
                else:
                    specs = [spec]

                application = BoundaryEmbeddingApplication(project, (task,))

                def compute(cid=cid, task=task):
                    if task.source_path is None:
                        raise _OutcomeError("failed", "source_missing")
                    outcome = run_boundary_embeddings(
                        (task,),
                        cancel_event=cancel,
                        model_session=model_session,
                        fingerprints=fingerprints,
                        runtime=runtime,
                    )[0]
                    if outcome.status != "succeeded":
                        raise _OutcomeError(
                            outcome.status,
                            outcome.code,
                            outcome.message,
                            outcome.record_json,
                        )
                    return {
                        "first": list(outcome.first),
                        "last": list(outcome.last),
                        "model": outcome.model,
                        "record_json": outcome.record_json,
                    }

                def apply(current, payload, cid=cid, application=application):
                    checked = _validated(cid, payload)
                    outcome = BoundaryEmbeddingOutcome(
                        cid,
                        "succeeded",
                        tuple(checked["first"]),
                        tuple(checked["last"]),
                        checked["model"],
                        record_json=checked["record_json"],
                    )
                    if not application.apply(current, outcome):
                        raise StaleJobResult(
                            "Boundary output changed before application"
                        )

                def validate(current, cid=cid, basis=basis):
                    return inputs(current, cid) == basis

                def is_applied(current, payload, cid=cid):
                    return _values(current, cid) == _validated(cid, payload)

                for candidate in specs:
                    receipt = batch.commit(
                        candidate,
                        compute=compute,
                        validate_input=validate,
                        apply=apply,
                        is_applied=is_applied,
                    )
                bucket = "succeeded" if receipt["applied"] else "skipped"
                result[bucket].append(
                    {
                        "clip_id": cid,
                        **(
                            {}
                            if receipt["applied"]
                            else {"reason": "already_committed"}
                        ),
                    }
                )
            except FingerprintCancelled:
                result["unprocessed"].extend(
                    {"clip_id": rest, "code": "cancelled"} for rest in ids[index:]
                )
                break
            except _OutcomeError as exc:
                if inputs(project, cid) != basis:
                    raise StaleJobResult("Boundary inputs changed during computation")
                if exc.record_json is not None:
                    stage_record(
                        cid,
                        AnalysisRecord.from_dict(json.loads(exc.record_json)),
                        basis,
                    )
                result[exc.status].append(
                    {"clip_id": cid, "code": exc.code, "message": exc.message}
                )
            progress(
                0.95 * (index + 1) / len(ids),
                f"Boundary embeddings ({index + 1}/{len(ids)})",
            )
        batch.flush()
        progress(1.0, "Boundary embeddings finished")
        return {"success": True, "result": result}

"""Durable, batched thumbnail embeddings for saved projects."""

from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.commits import ResultSpec, StaleJobResult, canonical_json, result_batch
from core.jobs.media import FingerprintCancelled, MediaFingerprints, media_stamp
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore
from core.operations.embeddings import (
    EmbeddingApplication,
    EmbeddingOptions,
    EmbeddingOutcome,
    EmbeddingTask,
    embedding_model_session,
    run_embeddings,
)
from core.project import Project
from core.project_revision import ProjectRevisionConflict


def _runtime() -> dict:
    packages: dict[str, str | None] = {}
    for package in ("torch", "transformers", "Pillow"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "model": "facebook/dinov2-base",
        "tag": "dinov2-vit-b-14",
        "dimensions": 768,
        "algorithm": "cls-l2/v1",
        "packages": packages,
    }


def _ids(project: Project, ids: list[str] | None) -> list[str]:
    selected = (
        list(dict.fromkeys(ids)) if ids is not None else [c.id for c in project.clips]
    )
    if any(cid not in project.clips_by_id for cid in selected):
        raise ValueError("Unknown embedding clip ID")
    return selected


def _task(project: Project, cid: str) -> EmbeddingTask:
    return EmbeddingTask(cid, project.clips_by_id[cid].thumbnail_path)


def _target(project: Project, cid: str) -> dict:
    clip = project.clips_by_id[cid]
    source = project.sources_by_id.get(clip.source_id)
    return {
        "clip_id": cid,
        "source_id": clip.source_id,
        "thumbnail_path": str(clip.thumbnail_path) if clip.thumbnail_path else None,
        "start_frame": clip.start_frame,
        "end_frame": clip.end_frame,
        "source_path": str(source.file_path) if source else None,
        "fps": source.fps if source else None,
    }


def embedding_job_spec(
    project: Project,
    clip_ids: list[str] | None,
    options: EmbeddingOptions,
    *,
    arguments: dict,
) -> OperationSpec:
    targets = []
    for cid in _ids(project, clip_ids):
        target = _target(project, cid)
        targets.append(
            {
                **target,
                "image_stamp": media_stamp(Path(target["thumbnail_path"]))
                if target["thumbnail_path"]
                else None,
                "source_stamp": media_stamp(Path(target["source_path"]))
                if target["source_path"]
                else None,
            }
        )
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="embeddings",
        version=1,
        arguments=arguments,
        inputs={"targets": targets, "options": asdict(options), "runtime": _runtime()},
        persistence="job_history",
        session_id=project.session.session_id,
        input_revision=revision.digest if revision else None,
    )


def _outcome(cid: str, payload: dict) -> EmbeddingOutcome:
    outcome = EmbeddingOutcome.from_vector(cid, payload["vector"])
    if outcome.model != payload["model"]:
        raise StaleJobResult("Embedding model identity is corrupt")
    return outcome


def _values(project: Project, cid: str) -> dict:
    clip = project.clips_by_id[cid]
    return {"vector": clip.embedding, "model": clip.embedding_model}


def _cached(store: JobStore, spec: ResultSpec) -> dict | None:
    row = store.get_result(spec.result_id)
    if row is None:
        return None
    if (
        row["spec_json"] != spec.identity_json
        or sha256(row["payload_json"].encode()).hexdigest() != row["payload_digest"]
    ):
        raise StaleJobResult("Cached embedding result is corrupt")
    payload: dict = json.loads(row["payload_json"])
    _outcome(json.loads(spec.identity_json)["target_id"], payload)
    return payload


def run_embedding_job(
    store: JobStore,
    path: Path,
    clip_ids: list[str] | None,
    progress: Callable[[float, str], None],
    cancel: Event,
    *,
    options: EmbeddingOptions | None = None,
    force: bool = False,
    operation: OperationSpec | None = None,
) -> dict:
    """Record each computed batch before saving any of its project changes."""
    options = (
        EmbeddingOptions(**json.loads(operation.inputs_json)["options"])
        if operation
        else (options or EmbeddingOptions())
    )
    if type(options.chunk_size) is not int or options.chunk_size < 1:
        raise ValueError("Embedding chunk size must be a positive integer")
    if operation:
        force = bool(operation.arguments.get("force", False))
    runtime = _runtime()
    fingerprint = MediaFingerprints(cancel).get
    with embedding_model_session() as model_session, result_batch(store, path) as batch:
        project = batch.project
        if operation:
            revision = project.session.file_revision
            if operation.input_revision is not None and (
                revision is None or revision.digest != operation.input_revision
            ):
                raise ProjectRevisionConflict(path)
            if (
                embedding_job_spec(
                    project, clip_ids, options, arguments=operation.arguments
                ).inputs_json
                != operation.inputs_json
            ):
                raise StaleJobResult("Embedding inputs changed while queued")
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
            if identity["kind"] != "embeddings":
                continue
            if (
                row["payload_digest"] != digest
                or sha256(row["payload_json"].encode()).hexdigest() != digest
            ):
                raise StaleJobResult("Committed embedding payload is corrupt")
            payload = json.loads(row["payload_json"])
            _outcome(identity["target_id"], payload)
            known.setdefault(identity["target_id"], []).append((row, identity, payload))

        def inputs(current: Project, cid: str) -> dict:
            target = _target(current, cid)
            return {
                "project_id": current.metadata.id,
                **target,
                "image": fingerprint(Path(target["thumbnail_path"]))
                if target["thumbnail_path"]
                else None,
                "source": fingerprint(Path(target["source_path"]))
                if target["source_path"]
                else None,
                "runtime": _runtime(),
            }

        result: dict = {
            "succeeded": [],
            "failed": [],
            "skipped": [],
            "unprocessed": [],
            "total_clips": len(ids),
        }
        settled = set()
        for start in range(0, len(ids), options.chunk_size):
            if cancel.is_set():
                break
            plans = {}
            pending = []
            try:
                for cid in ids[start : start + options.chunk_size]:
                    existing = project.clips_by_id[cid].embedding is not None
                    if existing and not force and cid not in known:
                        result["skipped"].append(
                            {"clip_id": cid, "reason": "already_populated"}
                        )
                        settled.add(cid)
                        continue
                    basis = inputs(project, cid)
                    if basis["runtime"] != runtime:
                        raise StaleJobResult("Embedding runtime changed")
                    identity_inputs: dict = {"basis": basis}
                    if force:
                        identity_inputs.update(
                            generation=len(known.get(cid, [])),
                            previous=_values(project, cid),
                        )
                    specs = [
                        ResultSpec.build(
                            path,
                            kind="embeddings",
                            version=1,
                            target_id=cid,
                            arguments=asdict(options),
                            inputs=identity_inputs,
                        )
                    ]
                    matches = [
                        row
                        for row, identity, payload in known.get(cid, [])
                        if identity["project_path"] == str(path)
                        and identity["arguments"] == asdict(options)
                        and identity["inputs"]["basis"] == basis
                        and _values(project, cid) == payload
                        and (not force or not row["committed"])
                    ]
                    if matches:
                        specs = [ResultSpec(path, row["spec_json"]) for row in matches]
                    elif existing and not force:
                        result["skipped"].append(
                            {"clip_id": cid, "reason": "already_populated"}
                        )
                        settled.add(cid)
                        continue
                    task = _task(project, cid)
                    plans[cid] = (basis, specs, EmbeddingApplication(project, (task,)))
                    if any(_cached(store, spec) is None for spec in specs):
                        pending.append(task)
                outcomes = run_embeddings(
                    tuple(pending),
                    options,
                    cancel_event=cancel,
                    model_session=model_session,
                )
                # Record the whole computed batch before project publication can fail.
                for outcome in outcomes:
                    cid = outcome.clip_id
                    if outcome.status != "succeeded":
                        result[outcome.status].append(
                            {
                                "clip_id": cid,
                                "code": outcome.code,
                                "message": outcome.message,
                            }
                        )
                        settled.add(cid)
                        continue
                    basis, specs, _ = plans[cid]
                    if inputs(project, cid) != basis:
                        raise StaleJobResult(
                            "Embedding inputs changed during computation"
                        )
                    payload_json = canonical_json(
                        {"vector": list(outcome.vector), "model": outcome.model}
                    )
                    for spec in specs:
                        store.record_result(
                            spec.result_id,
                            spec.identity_json,
                            payload_json,
                            sha256(payload_json.encode()).hexdigest(),
                        )
                for cid, (basis, specs, application) in plans.items():
                    if cid in settled or cancel.is_set():
                        continue

                    def apply(current, payload, cid=cid, application=application):
                        if not application.apply(current, _outcome(cid, payload)):
                            raise StaleJobResult(
                                "Embedding target changed during application"
                            )

                    def no_compute():
                        raise StaleJobResult("Recorded embedding payload disappeared")

                    def validate(
                        current: Project, cid: str = cid, basis: dict = basis
                    ) -> bool:
                        return inputs(current, cid) == basis

                    def is_applied(
                        current: Project, payload: dict, cid: str = cid
                    ) -> bool:
                        return _values(current, cid) == payload

                    for spec in specs:
                        receipt = batch.commit(
                            spec,
                            compute=no_compute,
                            validate_input=validate,
                            apply=apply,
                            is_applied=is_applied,
                        )
                    key = "succeeded" if receipt["applied"] else "skipped"
                    item = (
                        {
                            "clip_id": cid,
                            "embedding_dim": len(receipt["payload"]["vector"]),
                        }
                        if receipt["applied"]
                        else {"clip_id": cid, "reason": "already_committed"}
                    )
                    result[key].append(item)
                    settled.add(cid)
                    progress(
                        0.95 * len(settled) / len(ids),
                        f"Embeddings ({len(settled)}/{len(ids)})",
                    )
                if model_session.failed:
                    break
            except FingerprintCancelled:
                break
        code = (
            "embedding_failed"
            if model_session.failed and not cancel.is_set()
            else "cancelled"
        )
        result["unprocessed"].extend(
            {"clip_id": cid, "code": code} for cid in ids if cid not in settled
        )
        batch.flush()
        progress(1.0, "Embeddings finished")
        return {"success": True, "result": result}

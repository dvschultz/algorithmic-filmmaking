"""Durable append-only custom-query results for saved-project jobs."""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.commits import ResultSpec, StaleJobResult, result_batch
from core.jobs.description import _runtime
from core.jobs.media import FingerprintCancelled, MediaFingerprints, media_stamp
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore
from core.operations.custom_query import (
    CustomQueryApplication,
    CustomQueryOptions,
    CustomQueryOutcome,
    CustomQueryTask,
    resolve_options,
    run_custom_query,
)
from core.operations.description import DescriptionOptions
from core.project import Project
from core.project_revision import ProjectRevisionConflict


def _ids(project: Project, clip_ids: list[str] | None) -> list[str]:
    ids = (
        list(dict.fromkeys(clip_ids))
        if clip_ids is not None
        else [c.id for c in project.clips]
    )
    if any(cid not in project.clips_by_id for cid in ids):
        raise ValueError("Unknown custom-query clip ID")
    return ids


def _target(project: Project, cid: str) -> dict:
    clip = project.clips_by_id[cid]
    source = project.sources_by_id.get(clip.source_id)
    return {
        "clip_id": cid,
        "source_id": clip.source_id,
        "image_path": str(clip.thumbnail_path) if clip.thumbnail_path else None,
        "source_path": str(source.file_path) if source else None,
        "start_frame": clip.start_frame,
        "end_frame": clip.end_frame,
        "fps": source.fps if source else None,
    }


def _provenance(options: CustomQueryOptions) -> dict:
    from core.analysis_model_identity import CUSTOM_QUERY_RESPONSE_SCHEMA

    tier = {"cpu": "local", "gpu": "cloud"}.get(options.tier, options.tier)
    return {
        **_runtime(DescriptionOptions(tier, model=options.model, input_mode="frame")),
        "response_schema": CUSTOM_QUERY_RESPONSE_SCHEMA,
    }


def custom_query_job_spec(
    project: Project,
    clip_ids: list[str] | None,
    options: CustomQueryOptions,
    *,
    arguments: dict,
) -> OperationSpec:
    query = (arguments.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    targets = []
    for cid in _ids(project, clip_ids):
        data = _target(project, cid)
        targets.append(
            {
                **data,
                **{
                    key + "_stamp": media_stamp(Path(data[key])) if data[key] else None
                    for key in ("image_path", "source_path")
                },
            }
        )
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="custom_query",
        version=1,
        arguments={**arguments, "query": query},
        inputs={
            "targets": targets,
            "options": asdict(options),
            "runtime": _provenance(options),
        },
        persistence="job_history",
        session_id=project.session.session_id,
        input_revision=revision.digest if revision else None,
    )


class _OutcomeError(Exception):
    def __init__(self, outcome: CustomQueryOutcome):
        self.outcome = outcome


def run_custom_query_job(
    store: JobStore,
    path: Path,
    clip_ids: list[str] | None,
    progress: Callable[[float, str], None],
    cancel: Event,
    *,
    query: str | None = None,
    options: CustomQueryOptions | None = None,
    operation: OperationSpec | None = None,
) -> dict:
    if operation is not None:
        query = operation.arguments["query"]
        options = CustomQueryOptions(**json.loads(operation.inputs_json)["options"])
    query = (query or "").strip()
    if not query:
        raise ValueError("query is required")
    options = options or resolve_options()
    fingerprint = MediaFingerprints(cancel).get
    runtime = _provenance(options)
    arguments = {**asdict(options), "query": query}
    with result_batch(store, path) as batch:
        project = batch.project
        if operation is not None:
            revision = project.session.file_revision
            if operation.input_revision is not None and (
                revision is None or revision.digest != operation.input_revision
            ):
                raise ProjectRevisionConflict(path)
            live = custom_query_job_spec(
                project, clip_ids, options, arguments=operation.arguments
            )
            if live.inputs_json != operation.inputs_json:
                raise StaleJobResult("Custom-query inputs changed while queued")
        ids = _ids(project, clip_ids)
        # One query request is one append generation. Publish its successful
        # results together so a save failure cannot leave a partially advanced
        # generation that a retry would mistake for a new request.
        batch.max_items = max(1, len(ids))
        known: dict[str, list[tuple[dict, dict, dict]]] = {}
        for rid, digest in project.metadata.job_results.items():
            row = store.get_result(rid)
            if row is None:
                raise StaleJobResult("Committed result payload is missing")
            if sha256(row["spec_json"].encode()).hexdigest() != rid:
                raise StaleJobResult("Committed result identity is corrupt")
            identity = json.loads(row["spec_json"])
            if identity["kind"] != "custom_query":
                continue
            if (
                sha256(row["payload_json"].encode()).hexdigest() != digest
                or row["payload_digest"] != digest
            ):
                raise StaleJobResult("Committed custom-query payload is corrupt")
            known.setdefault(identity["target_id"], []).append(
                (row, identity, json.loads(row["payload_json"]))
            )

        def inputs(current: Project, cid: str) -> dict:
            target = _target(current, cid)
            return {
                "project_id": current.metadata.id,
                "target": target,
                "image": fingerprint(
                    Path(target["image_path"]) if target["image_path"] else None
                ),
                "source": fingerprint(
                    Path(target["source_path"]) if target["source_path"] else None
                ),
                "runtime": _provenance(options),
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
            try:
                basis = inputs(project, cid)
                if basis["runtime"] != runtime:
                    raise StaleJobResult("Custom-query runtime changed")
                clip = project.clips_by_id[cid]
                previous = deepcopy(clip.custom_queries or [])
                spec = ResultSpec.build(
                    path,
                    kind="custom_query",
                    version=1,
                    target_id=cid,
                    arguments=arguments,
                    inputs={
                        "basis": basis,
                        "previous_queries": previous,
                        "generation": len(known.get(cid, [])),
                    },
                )
                # A project save may succeed before its cache checkpoint fails.
                # Reconcile that exact append before treating a retry as a refresh.
                for row, identity, payload in known.get(cid, []):
                    if (
                        not row["committed"]
                        and identity["project_path"] == str(path)
                        and identity["arguments"] == arguments
                        and identity["inputs"]["basis"] == basis
                        and previous
                        == [*identity["inputs"]["previous_queries"], payload]
                    ):
                        spec = ResultSpec(path, row["spec_json"])
                        previous = identity["inputs"]["previous_queries"]
                        break
                task = CustomQueryTask(cid, clip.thumbnail_path, query)
                application = CustomQueryApplication(project, (task,))

                def compute(task=task):
                    outcome = run_custom_query((task,), options, cancel_event=cancel)[0]
                    if outcome.status != "succeeded":
                        raise _OutcomeError(outcome)
                    return {
                        "query": query,
                        "match": outcome.match,
                        "confidence": round(outcome.confidence or 0.0, 4),
                        "model": outcome.model,
                    }

                def apply(current, payload, cid=cid, application=application):
                    if not application.apply(
                        current,
                        CustomQueryOutcome(
                            cid,
                            payload["query"],
                            "succeeded",
                            payload["match"],
                            payload["confidence"],
                            payload["model"],
                        ),
                    ):
                        raise StaleJobResult(
                            "Custom-query target changed during application"
                        )

                def validate(current, cid=cid, basis=basis):
                    return inputs(current, cid) == basis

                def is_applied(current, payload, cid=cid, previous=previous):
                    return current.clips_by_id[cid].custom_queries == [
                        *previous,
                        payload,
                    ]

                receipt = batch.commit(
                    spec,
                    compute=compute,
                    validate_input=validate,
                    apply=apply,
                    is_applied=is_applied,
                )
                if receipt["applied"]:
                    result["succeeded"].append({"clip_id": cid, **receipt["payload"]})
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
                0.95 * (index + 1) / len(ids), f"Custom query ({index + 1}/{len(ids)})"
            )
        batch.flush()
        progress(1.0, "Custom queries finished")
        return {"success": True, "result": result}

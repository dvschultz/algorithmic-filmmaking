"""Durable verified brightness and volume analysis for saved projects."""

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot
from core.jobs.commits import ResultSpec, StaleJobResult, result_batch
from core.jobs.media import FingerprintCancelled
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore
from core.operations.scalars import (
    FIELDS,
    ScalarApplication,
    ScalarOperation,
    ScalarOutcome,
    ScalarTask,
    run_scalars,
    scalar_parameters,
    scalar_runtime,
    scalar_sampling,
    scalar_task,
)
from core.project import Project
from core.project_revision import ProjectRevisionConflict
from models.analysis_record import AnalysisIdentity, AnalysisRecord


def _ids(project: Project, clip_ids: list[str] | None) -> list[str]:
    ids = (
        list(dict.fromkeys(clip_ids))
        if clip_ids is not None
        else [clip.id for clip in project.clips]
    )
    if any(cid not in project.clips_by_id for cid in ids):
        raise ValueError("Unknown scalar clip ID")
    return ids


def _task(
    project: Project, cid: str, kind: ScalarOperation, samples: int
) -> ScalarTask:
    clip = project.clips_by_id[cid]
    return scalar_task(
        clip, project.sources_by_id.get(clip.source_id), kind, num_samples=samples
    )


def scalar_job_spec(
    project: Project,
    clip_ids: list[str] | None,
    kind: ScalarOperation,
    *,
    num_samples: int = 5,
    arguments: dict,
) -> OperationSpec:
    runtime = scalar_runtime(kind)
    if type(num_samples) is not int or num_samples < 1:
        raise ValueError("Sample count must be a positive integer")
    tasks = [_task(project, cid, kind, num_samples) for cid in _ids(project, clip_ids)]
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="scalars",
        version=1,
        arguments=arguments,
        inputs={
            "tasks": [asdict(task) for task in tasks],
            "runtime": runtime,
            "scalar": kind,
            "num_samples": num_samples,
        },
        persistence="job_history",
        session_id=project.session.session_id,
        input_revision=revision.digest if revision else None,
    )


class _OutcomeError(Exception):
    def __init__(self, outcome: ScalarOutcome):
        self.outcome = outcome


def run_scalar_job(
    store: JobStore,
    path: Path,
    clip_ids: list[str] | None,
    progress: Callable[[float, str], None],
    cancel: Event,
    *,
    kind: ScalarOperation,
    num_samples: int = 5,
    force: bool = False,
    operation: OperationSpec | None = None,
) -> dict:
    """Recover computed records after failed publication without rerunning providers."""
    if operation is not None:
        options = json.loads(operation.inputs_json)
        if (
            operation.kind != "scalars"
            or operation.version != 1
            or options["scalar"] != kind
        ):
            raise ValueError("Queued scalar operation does not match")
        num_samples = options["num_samples"]
        force = bool(operation.arguments.get("force", False))
    runtime = scalar_runtime(kind)
    if type(num_samples) is not int or num_samples < 1:
        raise ValueError("Sample count must be a positive integer")
    fingerprints = AnalysisFingerprints(cancel)
    with result_batch(store, path) as batch:
        project = batch.project
        ids = _ids(project, clip_ids)
        if operation is not None:
            revision = project.session.file_revision
            if operation.input_revision is not None and (
                revision is None or revision.digest != operation.input_revision
            ):
                raise ProjectRevisionConflict(path)
            live = scalar_job_spec(
                project,
                clip_ids,
                kind,
                num_samples=num_samples,
                arguments=operation.arguments,
            )
            if live.inputs_json != operation.inputs_json:
                raise StaleJobResult("Scalar inputs changed while queued")
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
            if identity["kind"] != f"scalar_{kind}":
                continue
            if (
                sha256(row["payload_json"].encode()).hexdigest() != digest
                or row["payload_digest"] != digest
            ):
                raise StaleJobResult("Committed scalar result is corrupt")
            known.setdefault(identity["target_id"], []).append(
                (row, identity, json.loads(row["payload_json"]))
            )

        def basis(current: Project, cid: str) -> dict:
            snapshot = AnalysisSnapshot.from_json(
                _task(current, cid, kind, num_samples).snapshot_json
            )
            identity = fingerprints.identity(
                snapshot.inputs,
                operation=kind,
                model=scalar_runtime(kind),
                parameters={"num_samples": num_samples} if kind == "brightness" else {},
                sampling=scalar_sampling(kind),
            )
            return {
                "project_id": current.metadata.id,
                "inputs": snapshot.inputs.to_dict(),
                "identity": identity.to_dict(),
            }

        def is_output(current: Project, cid: str, payload: dict) -> bool:
            clip = current.clips_by_id[cid]
            record = clip.analysis_records.get(kind)
            return bool(
                isinstance(record, AnalysisRecord)
                and record.to_dict() == json.loads(payload.get("record_json") or "null")
                and record.identity is not None
                and record.identity.to_dict() == basis(current, cid)["identity"]
                and record.value == {FIELDS[kind]: getattr(clip, FIELDS[kind])}
            )

        result: dict = {
            key: [] for key in ("succeeded", "skipped", "failed", "unprocessed")
        }
        result["total_clips"] = len(ids)
        for index, cid in enumerate(ids):
            if cancel.is_set():
                result["unprocessed"].extend(
                    {"clip_id": rest, "message": "Cancelled"} for rest in ids[index:]
                )
                break
            try:
                captured = basis(project, cid)
                if captured["identity"]["model"] != runtime:
                    raise StaleJobResult("Scalar runtime changed")
                task = _task(project, cid, kind, num_samples)
                snapshot = AnalysisSnapshot.from_json(task.snapshot_json)
                semantic = AnalysisIdentity.from_dict(captured["identity"])
                reused = snapshot.reusable_record(semantic) if not force else None
                prior = project.clips_by_id[cid].analysis_records.get(kind)
                previous = {
                    "basis": captured,
                    "value": json.loads(snapshot.value_json),
                    "record": prior.to_dict()
                    if isinstance(prior, AnalysisRecord)
                    else None,
                }
                if force:
                    previous["generation"] = len(known.get(cid, []))
                spec = ResultSpec.build(
                    path,
                    kind=f"scalar_{kind}",
                    version=1,
                    target_id=cid,
                    arguments=scalar_parameters(task),
                    inputs=previous,
                )
                candidates = [spec]
                for row, identity, payload in known.get(cid, []):
                    if (
                        identity["project_path"] == str(batch.path)
                        and identity["inputs"]["basis"] == captured
                        and identity["arguments"] == scalar_parameters(task)
                        and is_output(project, cid, payload)
                        and (not force or not row["committed"])
                    ):
                        candidates = [ResultSpec(batch.path, row["spec_json"])]
                        break
                application = ScalarApplication(project, task)

                publication = {"applied": False}

                def validate(
                    current, cid=cid, captured=captured, publication=publication
                ):
                    if cancel.is_set() and not publication["applied"]:
                        raise FingerprintCancelled()
                    return basis(current, cid) == captured

                def apply(
                    current,
                    payload,
                    application=application,
                    publication=publication,
                    semantic=semantic,
                ):
                    recovered = AnalysisRecord.from_dict(
                        json.loads(payload.get("record_json") or "null")
                    )
                    if recovered.identity != semantic:
                        raise StaleJobResult(
                            "Recovered scalar identity does not match current inputs"
                        )
                    if not application.apply(current, ScalarOutcome(**payload)):
                        raise StaleJobResult("Scalar target changed during application")
                    publication["applied"] = True

                def applied(current, payload, cid=cid):
                    return is_output(current, cid, payload)

                if reused is not None and candidates == [spec]:
                    outcome = ScalarOutcome(
                        cid,
                        kind,
                        "skipped",
                        json.dumps(reused.to_dict(), sort_keys=True),
                    )
                    if reused != prior:

                        def apply_reuse(
                            current: Project,
                            payload: dict = asdict(outcome),
                            publish: Callable = apply,
                        ) -> None:
                            publish(current, payload)

                        def reused_applied(
                            current: Project,
                            payload: dict = asdict(outcome),
                            check: Callable = applied,
                        ) -> bool:
                            return bool(check(current, payload))

                        batch.stage_analysis(
                            apply=apply_reuse,
                            validate_input=validate,
                            is_applied=reused_applied,
                        )
                    result["skipped"].append(
                        {"clip_id": cid, "reason": "valid_analysis"}
                    )
                    continue

                def compute(task=task):
                    outcome = run_scalars(
                        (replace(task, skip_existing=False),),
                        cancel_event=cancel,
                        fingerprints=fingerprints,
                    )[0]
                    if outcome.status != "succeeded":
                        raise _OutcomeError(outcome)
                    return asdict(outcome)

                for candidate in candidates:
                    receipt = batch.commit(
                        candidate,
                        compute=compute,
                        validate_input=validate,
                        apply=apply,
                        is_applied=applied,
                    )
                result["succeeded" if receipt["applied"] else "skipped"].append(
                    {"clip_id": cid}
                )
            except FingerprintCancelled:
                result["unprocessed"].extend(
                    {"clip_id": rest, "message": "Cancelled"} for rest in ids[index:]
                )
                break
            except _OutcomeError as exc:
                outcome = exc.outcome
                if outcome.status == "unprocessed":
                    result["unprocessed"].extend(
                        {"clip_id": rest, "message": "Cancelled"}
                        for rest in ids[index:]
                    )
                    break
                if outcome.record_json is not None:
                    record = AnalysisRecord.from_dict(json.loads(outcome.record_json))

                    def apply_failure(
                        current: Project,
                        cid: str = cid,
                        record: AnalysisRecord = record,
                        publication: dict = publication,
                    ) -> None:
                        current.record_analysis("clip", cid, kind, record)
                        publication["applied"] = True

                    def failure_applied(
                        current: Project,
                        cid: str = cid,
                        record: AnalysisRecord = record,
                    ) -> bool:
                        return bool(
                            current.clips_by_id[cid].analysis_records.get(kind)
                            == record
                        )

                    batch.stage_analysis(
                        apply=apply_failure,
                        validate_input=validate,
                        is_applied=failure_applied,
                    )
                result["failed"].append({"clip_id": cid, "message": outcome.message})
            progress(
                0.95 * (index + 1) / len(ids),
                f"{kind.capitalize()} ({index + 1}/{len(ids)})",
            )
        batch.flush()
        progress(1.0, f"{kind.capitalize()} finished")
        return {"success": True, "result": result}

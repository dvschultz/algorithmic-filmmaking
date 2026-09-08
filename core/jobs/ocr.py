"""Durable OCR results for saved-project jobs."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot
from core.analysis_model_identity import ocr_runtime
from models.analysis_record import AnalysisRecord

from core.jobs.commits import ResultSpec, StaleJobResult, result_batch
from core.jobs.media import FingerprintCancelled, MediaFingerprints, media_stamp
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore
from core.operations.ocr import (
    OcrApplication,
    OcrOptions,
    OcrOutcome,
    OcrTask,
    ocr_identity,
    resolve_ocr_options,
    run_ocr,
)
from core.project import Project
from core.project_revision import ProjectRevisionConflict


def _runtime() -> dict:
    return ocr_runtime()


def resolve_options(options: OcrOptions) -> OcrOptions:
    return resolve_ocr_options(options)


def _ids(project: Project, clip_ids: list[str] | None) -> list[str]:
    ids = (
        list(dict.fromkeys(clip_ids))
        if clip_ids is not None
        else [c.id for c in project.clips]
    )
    if any(cid not in project.clips_by_id for cid in ids):
        raise ValueError("Unknown OCR clip ID")
    return ids


def _task(project: Project, cid: str) -> OcrTask:
    clip = project.clips_by_id[cid]
    source = project.sources_by_id.get(clip.source_id)
    return OcrTask.from_clip(clip, source)


def _task_data(task: OcrTask) -> dict:
    data = asdict(task)
    data.pop("analysis_json")
    data["skip"] = False
    data["analysis_version"] = 2 if task.analysis_json is not None else 1
    data["path"] = str(task.path) if task.path else None
    return data


def _values(project: Project, cid: str) -> list[dict] | None:
    texts = project.clips_by_id[cid].extracted_texts
    return [text.to_dict() for text in texts] if texts is not None else None


def _outcome(cid: str, payload: dict) -> OcrOutcome:
    try:
        outcome = OcrOutcome.from_dict(payload)
        if (
            outcome.clip_id != cid
            or outcome.target_type != "clip"
            or outcome.status != "succeeded"
        ):
            raise ValueError("Unexpected OCR outcome identity")
        return outcome
    except (TypeError, ValueError, KeyError) as exc:
        raise StaleJobResult("Recorded OCR payload is invalid") from exc


def ocr_job_spec(
    project: Project,
    clip_ids: list[str] | None,
    options: OcrOptions,
    *,
    arguments: dict,
) -> OperationSpec:
    options = resolve_options(options)
    targets = []
    for cid in _ids(project, clip_ids):
        task = _task(project, cid)
        targets.append(
            {
                **_task_data(task),
                "source_stamp": media_stamp(task.path) if task.path else None,
            }
        )
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="ocr",
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
    def __init__(self, outcome: OcrOutcome):
        self.outcome = outcome


def run_ocr_job(
    store: JobStore,
    path: Path,
    clip_ids: list[str] | None,
    progress: Callable[[float, str], None],
    cancel: Event,
    *,
    options: OcrOptions | None = None,
    force: bool = False,
    operation: OperationSpec | None = None,
) -> dict:
    """Reuse completed computation after failed saves, without implicit provider calls."""
    options = (
        OcrOptions(**json.loads(operation.inputs_json)["options"])
        if operation
        else resolve_options(options or OcrOptions())
    )
    if operation is not None:
        force = bool(operation.arguments.get("force", False))
    path = path.resolve()
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
            live = ocr_job_spec(
                project, clip_ids, options, arguments=operation.arguments
            )
            if live.inputs_json != operation.inputs_json:
                raise StaleJobResult("OCR inputs changed while queued")
        ids = _ids(project, clip_ids)
        if force:
            batch.max_items = max(1, len(ids))
        known: dict[str, list[tuple[dict, dict, dict]]] = {}
        for rid, digest in project.metadata.job_results.items():
            row = store.get_result(rid)
            if row is None:
                continue  # Verified project records can outlive the job cache.
            if sha256(row["spec_json"].encode()).hexdigest() != rid:
                raise StaleJobResult("Committed result identity is corrupt")
            identity = json.loads(row["spec_json"])
            if identity["kind"] != "ocr":
                continue
            if (
                sha256(row["payload_json"].encode()).hexdigest() != digest
                or row["payload_digest"] != digest
            ):
                raise StaleJobResult("Committed OCR payload is corrupt")
            _outcome(identity["target_id"], json.loads(row["payload_json"]))
            known.setdefault(identity["target_id"], []).append(
                (row, identity, json.loads(row["payload_json"]))
            )

        def inputs(current: Project, cid: str) -> dict:
            task = _task(current, cid)
            return {
                "project_id": current.metadata.id,
                "task": _task_data(task),
                "source": fingerprint(task.path),
                "runtime": _runtime(),
            }

        def is_output(current: Project, cid: str, payload: dict) -> bool:
            record = current.clips_by_id[cid].analysis_records.get("extract_text")
            if payload.get("record_json") is not None and (
                record is None or record.to_dict() != json.loads(payload["record_json"])
            ):
                return False
            existing = _values(current, cid)
            return existing == [
                text.to_model().to_dict() for text in _outcome(cid, payload).texts
            ]

        def stage_record(cid: str, record: AnalysisRecord, basis: dict) -> None:
            batch.stage_analysis(
                apply=lambda current: current.record_analysis(
                    "clip", cid, "extract_text", record
                ),
                validate_input=lambda current: inputs(current, cid) == basis,
                is_applied=lambda current: current.clips_by_id[
                    cid
                ].analysis_records.get("extract_text")
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
            existing = _values(project, cid) is not None
            try:
                basis = inputs(project, cid)
                if basis["runtime"] != runtime:
                    raise StaleJobResult("OCR runtime changed")
                task = _task(project, cid)
                assert task.analysis_json is not None
                snapshot = AnalysisSnapshot.from_json(task.analysis_json)
                semantic = (
                    ocr_identity(snapshot, options, fingerprints, runtime)
                    if snapshot.inputs.unchanged()
                    else None
                )
                reused = (
                    snapshot.reusable_record(semantic)
                    if semantic is not None and not force
                    else None
                )
                prior_record = project.clips_by_id[cid].analysis_records.get(
                    "extract_text"
                )
                identity_inputs: dict = {
                    "basis": basis,
                    "previous_record": prior_record.to_dict() if prior_record else None,
                    "previous_texts": _values(project, cid),
                }
                if force:
                    identity_inputs["generation"] = len(known.get(cid, []))
                arguments = asdict(options)
                spec = ResultSpec.build(
                    path,
                    kind="ocr",
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

                application = OcrApplication(project, (task,), options)

                def compute(task=task):
                    outcome = run_ocr(
                        (task,),
                        options,
                        cancel_event=cancel,
                        fingerprints=fingerprints,
                        runtime=runtime,
                    )[0]
                    if outcome.status != "succeeded":
                        raise _OutcomeError(outcome)
                    return asdict(outcome)

                def apply(current, payload, cid=cid, application=application):
                    outcome = _outcome(cid, payload)
                    if not application.apply(current, outcome):
                        raise StaleJobResult("OCR target changed during application")

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
                            "text_count": len(_outcome(cid, receipt["payload"]).texts),
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
                    raise StaleJobResult("OCR inputs changed during computation")
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
                f"OCR ({index + 1}/{len(ids)})",
            )
        batch.flush()
        progress(1.0, "OCR finished")
        return {"success": True, "result": result}

"""Durable word alignment using project-first result receipts."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from typing import Callable, Literal

from core.jobs.commits import ResultSpec, StaleJobResult, result_batch
from core.jobs.media import FingerprintCancelled, MediaFingerprints, media_stamp
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore
from core.operations.alignment import (
    AlignmentApplication,
    AlignmentOutcome,
    AlignmentTask,
    aligned_segments,
    run_alignment,
    snapshot_alignment_tasks,
)
from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot
from core.operations.alignment_records import (
    alignment_runtime,
    alignment_identity,
    execution_is_current,
    prior_execution,
)
from models.analysis_record import AnalysisRecord
from core.project import Project
from core.project_revision import ProjectRevisionConflict
from core.transcription_models import WordTimestamp


def _ids(project: Project, clip_ids: list[str] | None) -> list[str]:
    ids = (
        list(dict.fromkeys(clip_ids))
        if clip_ids is not None
        else [c.id for c in project.clips]
    )
    if any(cid not in project.clips_by_id for cid in ids):
        raise ValueError("Unknown alignment clip ID")
    return ids


def _task_data(task: AlignmentTask) -> dict:
    value = asdict(task)
    value["target"]["source_path"] = (
        str(task.target.source_path) if task.target.source_path else None
    )
    return value


def alignment_job_spec(
    project: Project, clip_ids: list[str] | None, *, force: bool, arguments: dict
) -> OperationSpec:
    tasks = snapshot_alignment_tasks(
        [project.clips_by_id[cid] for cid in _ids(project, clip_ids)],
        project.sources_by_id,
        skip_existing=False,
        verified=True,
    )
    revision = project.session.file_revision
    return alignment_operation_spec(
        tasks,
        force=force,
        arguments=arguments,
        persistence="job_history",
        session_id=project.session.session_id,
        input_revision=revision.digest if revision else None,
    )


def alignment_operation_spec(
    tasks: tuple[AlignmentTask, ...],
    *,
    force: bool,
    arguments: dict,
    persistence: Literal["job_history", "session_only"],
    session_id: str | None,
    input_revision: str | None,
) -> OperationSpec:
    """Describe detached alignment inputs for either runtime surface."""
    from core.operations.alignment_records import alignment_runtime

    runtime = (
        alignment_runtime()
        if any(task.analysis_json is not None for task in tasks)
        else None
    )
    targets = [
        {
            **_task_data(task),
            **({"runtime": runtime} if task.analysis_json is not None else {}),
            "media_stamp": media_stamp(task.target.source_path)
            if task.target.source_path
            else None,
        }
        for task in tasks
    ]
    return OperationSpec.build(
        kind="align_words",
        version=2 if runtime is not None else 1,
        arguments=arguments,
        inputs={"targets": targets, "force": force},
        persistence=persistence,
        session_id=session_id,
        input_revision=input_revision,
    )


class _OutcomeError(Exception):
    def __init__(self, outcome: AlignmentOutcome):
        self.outcome = outcome


def run_alignment_job(
    store: JobStore,
    path: Path,
    clip_ids: list[str] | None,
    progress: Callable[[float, str], None],
    cancel: Event,
    *,
    force: bool = False,
    operation: OperationSpec | None = None,
) -> dict:
    media_fingerprints = MediaFingerprints(cancel)
    fingerprint = media_fingerprints.get
    fingerprints = AnalysisFingerprints(cancel, media_fingerprints=media_fingerprints)
    availability: tuple[bool, list[str]] | None = None
    with result_batch(store, path) as batch:
        project = batch.project
        if operation is not None:
            revision = project.session.file_revision
            if operation.input_revision is not None and (
                revision is None or revision.digest != operation.input_revision
            ):
                raise ProjectRevisionConflict(path)
            live = alignment_job_spec(
                project, clip_ids, force=force, arguments=operation.arguments
            )
            if live.inputs_json != operation.inputs_json:
                raise StaleJobResult(
                    "Alignment inputs changed while the job was queued"
                )
        ids = _ids(project, clip_ids)
        if force:
            batch.max_items = max(1, len(ids))
        known: dict[str, list[tuple[dict, dict, dict]]] = {}
        for result_id, receipt_digest in project.metadata.job_results.items():
            row = store.get_result(result_id)
            if row is None:
                continue
            if sha256(row["spec_json"].encode()).hexdigest() != result_id:
                raise StaleJobResult("Committed alignment identity is corrupt")
            identity = json.loads(row["spec_json"])
            if identity["kind"] != "align_words":
                continue
            digest = sha256(row["payload_json"].encode()).hexdigest()
            if digest != row["payload_digest"] or digest != receipt_digest:
                raise StaleJobResult("Committed alignment payload is corrupt")
            known.setdefault(identity["target_id"], []).append(
                (row, identity, json.loads(row["payload_json"]))
            )

        def inputs(current: Project, cid: str) -> dict:
            clip = current.clips_by_id[cid]
            task = snapshot_alignment_tasks(
                [clip], current.sources_by_id, skip_existing=False
            )[0]
            data = _task_data(task)
            data.pop("analysis_json", None)
            segments = json.loads(task.transcript_json)
            for segment in segments:
                segment.pop("words", None)
            data["transcript_json"] = json.dumps(
                segments, sort_keys=True, allow_nan=False
            )
            return {
                "project_id": current.metadata.id,
                "source_id": clip.source_id,
                "task": data,
                "media": fingerprint(task.target.source_path),
                "runtime": {
                    key: value
                    for key, value in alignment_runtime().items()
                    if key != "revision"
                },
            }

        def transcript(current: Project, cid: str):
            value = current.clips_by_id[cid].transcript
            return (
                [segment.to_dict() for segment in value] if value is not None else None
            )

        def current_output(current: Project, cid: str) -> dict:
            record = current.clips_by_id[cid].analysis_records.get("align_words")
            return {
                "segments": transcript(current, cid),
                "record": record.to_dict() if record is not None else None,
            }

        def payload_output(payload: dict) -> dict:
            return {
                "segments": payload["segments"],
                "record": json.loads(payload["record_json"])
                if payload.get("record_json")
                else None,
            }

        def payload_is_current(payload: dict) -> bool:
            if not payload.get("record_json"):
                return False
            record = AnalysisRecord.from_dict(json.loads(payload["record_json"]))
            if record.identity is None:
                return False
            runtime = record.identity.to_dict()["model"]
            return execution_is_current(runtime) and runtime == alignment_runtime(
                execution=runtime["execution"]
            )

        def stage_reuse(cid: str, record: AnalysisRecord, basis: dict) -> None:
            batch.stage_analysis(
                apply=lambda current: current.record_analysis(
                    "clip", cid, "align_words", record
                ),
                validate_input=lambda current: inputs(current, cid) == basis,
                is_applied=lambda current: current.clips_by_id[
                    cid
                ].analysis_records.get("align_words")
                == record,
            )

        output: dict = {
            "succeeded": [],
            "failed": [],
            "skipped": [],
            "unprocessed": [],
            "total_clips": len(ids),
        }
        for index, cid in enumerate(ids):
            if cancel.is_set():
                output["unprocessed"].extend(
                    {"clip_id": item, "code": "cancelled"} for item in ids[index:]
                )
                break
            clip = project.clips_by_id[cid]
            if not clip.transcript:
                output["skipped"].append({"clip_id": cid, "reason": "no_transcript"})
                continue
            if (
                not force
                and cid in known
                and not any(
                    payload["segments"] == transcript(project, cid)
                    for _, _, payload in known[cid]
                )
            ):
                raise StaleJobResult(
                    "Previously committed alignment changed; use force to replace edited words"
                )
            try:
                basis = inputs(project, cid)
                task = snapshot_alignment_tasks(
                    [clip], project.sources_by_id, skip_existing=False, verified=True
                )[0]
                snapshot = (
                    AnalysisSnapshot.from_json(task.analysis_json)
                    if task.analysis_json is not None
                    else None
                )
                runtime = (
                    alignment_runtime(execution=prior_execution(snapshot))
                    if snapshot is not None
                    else alignment_runtime()
                )
                reused = None
                if (
                    not force
                    and snapshot is not None
                    and snapshot.inputs.unchanged()
                    and execution_is_current(runtime)
                ):
                    semantic = alignment_identity(
                        snapshot, task.transcript_json, fingerprints, runtime
                    )
                    reused = snapshot.reusable_record(semantic)
                    if (
                        inputs(project, cid) != basis
                        or not snapshot.inputs.unchanged()
                        or alignment_runtime(execution=runtime["execution"]) != runtime
                    ):
                        raise StaleJobResult(
                            "Alignment inputs changed during verification"
                        )
                if cancel.is_set():
                    raise FingerprintCancelled()
                identity_inputs: dict = {
                    "basis": basis,
                    "previous_output": current_output(project, cid),
                    "model_revision": runtime["revision"],
                }
                if force:
                    identity_inputs["refresh_generation"] = len(known.get(cid, []))
                spec = ResultSpec.build(
                    path,
                    kind="align_words",
                    version=2,
                    target_id=cid,
                    arguments={},
                    inputs=identity_inputs,
                )
                specs = [spec]
                matches = [
                    row
                    for row, identity, payload in known.get(cid, [])
                    if identity["project_path"] == str(path.resolve())
                    and identity["inputs"]["basis"] == basis
                    and payload_output(payload) == current_output(project, cid)
                    and payload_is_current(payload)
                    and (
                        (not force and reused is not None)
                        or (force and not row["committed"])
                    )
                ]
                if matches:
                    specs = [ResultSpec(path, row["spec_json"]) for row in matches]
                elif reused is not None:
                    if reused != clip.analysis_records.get("align_words"):
                        stage_reuse(cid, reused, basis)
                    output["skipped"].append(
                        {"clip_id": cid, "reason": "valid_analysis"}
                    )
                    continue
                elif (
                    runtime["revision"] is not None
                    and store.get_result(spec.result_id) is None
                ):
                    # The first inference can populate the model cache. Its
                    # pre-inference receipt has a None revision; authenticate
                    # its actual result before recovering that initial receipt.
                    initial = ResultSpec.build(
                        path,
                        kind="align_words",
                        version=2,
                        target_id=cid,
                        arguments={},
                        inputs={**identity_inputs, "model_revision": None},
                    )
                    row = store.get_result(initial.result_id)
                    if row is not None:
                        if (
                            row["spec_json"] != initial.identity_json
                            or sha256(row["payload_json"].encode()).hexdigest()
                            != row["payload_digest"]
                        ):
                            raise StaleJobResult("Initial alignment receipt is corrupt")
                        if payload_is_current(json.loads(row["payload_json"])):
                            specs = [initial]
                application = AlignmentApplication(project, (task,))

                def compute(task=task):
                    nonlocal availability
                    if availability is None:
                        from core.feature_registry import check_feature_ready

                        availability = check_feature_ready("word_alignment")
                    if cancel.is_set():
                        raise FingerprintCancelled()
                    if not availability[0]:
                        raise _OutcomeError(
                            AlignmentOutcome(
                                task.clip_id,
                                "failed",
                                code="dependency_missing",
                                message="Word alignment dependencies unavailable. Install them from Settings > Dependencies.",
                            )
                        )
                    outcome = run_alignment(
                        (task,), cancel_event=cancel, fingerprints=fingerprints
                    )[0]
                    if outcome.status != "succeeded":
                        raise _OutcomeError(outcome)
                    return {
                        "segments": [
                            segment.to_dict()
                            for segment in aligned_segments(task, outcome.words)
                        ],
                        "words": [word.to_dict() for word in outcome.words],
                        "record_json": outcome.record_json,
                    }

                def apply(current, payload, cid=cid, application=application):
                    if not payload_is_current(payload):
                        raise StaleJobResult(
                            "Alignment result runtime is no longer current"
                        )
                    result = AlignmentOutcome(
                        cid,
                        "succeeded",
                        tuple(
                            WordTimestamp.from_dict(value) for value in payload["words"]
                        ),
                        record_json=payload["record_json"],
                    )
                    if not application.apply(current, result):
                        raise StaleJobResult(
                            "Alignment target changed during application"
                        )

                def validate(
                    current: Project,
                    cid: str = cid,
                    basis: dict = basis,
                    request_ids: tuple[str, ...] = tuple(
                        item.result_id for item in specs
                    ),
                ) -> bool:
                    # Cancellation stops admission before publication; already
                    # staged receipts remain eligible for the final prefix save.
                    if cancel.is_set() and not any(
                        rid in current.metadata.job_results for rid in request_ids
                    ):
                        raise FingerprintCancelled()
                    return inputs(current, cid) == basis

                def is_applied(current: Project, payload: dict, cid: str = cid) -> bool:
                    return current_output(current, cid) == payload_output(
                        payload
                    ) and payload_is_current(payload)

                # Reconcile every matching saved refresh, including identical
                # outputs whose checkpoints failed on more than one run.
                for candidate in specs:
                    receipt = batch.commit(
                        candidate,
                        compute=compute,
                        validate_input=validate,
                        apply=apply,
                        is_applied=is_applied,
                    )
                if receipt["applied"]:
                    output["succeeded"].append(
                        {
                            "clip_id": cid,
                            "word_count": sum(
                                len(segment.get("words") or [])
                                for segment in receipt["payload"]["segments"]
                            ),
                        }
                    )
                else:
                    output["skipped"].append(
                        {"clip_id": cid, "reason": "already_committed"}
                    )
            except FingerprintCancelled:
                output["unprocessed"].extend(
                    {"clip_id": item, "code": "cancelled"} for item in ids[index:]
                )
                break
            except _OutcomeError as exc:
                outcome = exc.outcome
                if inputs(project, cid) != basis:
                    raise StaleJobResult("Alignment inputs changed during computation")
                if (
                    outcome.can_apply
                    and outcome.record_json is not None
                    and not cancel.is_set()
                ):
                    record = AnalysisRecord.from_dict(json.loads(outcome.record_json))

                    def publish_failure(
                        current, application=application, outcome=outcome
                    ):
                        if not application.apply(current, outcome):
                            raise StaleJobResult(
                                "Alignment target changed during failure publication"
                            )

                    def failure_input(
                        current: Project, cid: str = cid, basis: dict = basis
                    ) -> bool:
                        return inputs(current, cid) == basis

                    def failure_applied(
                        current: Project,
                        cid: str = cid,
                        record: AnalysisRecord = record,
                    ) -> bool:
                        return (
                            current.clips_by_id[cid].analysis_records.get("align_words")
                            == record
                        )

                    batch.stage_analysis(
                        apply=publish_failure,
                        validate_input=failure_input,
                        is_applied=failure_applied,
                    )
                output[outcome.status].append(
                    {"clip_id": cid, "code": outcome.code, "message": outcome.message}
                )
            progress(
                0.95 * (index + 1) / len(ids),
                f"Aligning words ({index + 1}/{len(ids)})",
            )
        batch.flush()
        progress(1.0, "Alignment finished")
        return {"success": True, "result": output}

"""Restart-safe transcription receipts for saved project jobs."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from typing import Callable, Literal

from core.jobs.commits import ResultSpec, StaleJobResult, result_batch
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore
from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot
from models.analysis_record import AnalysisRecord
from core.operations.transcription_records import (
    transcription_task,
    transcription_runtime,
    transcription_identity,
    transcription_parameters,
)
from core.jobs.media import (
    MediaFingerprints,
    FingerprintCancelled as _Cancelled,
    media_stamp as _stamp,
)
from core.operations.transcription import (
    TranscriptionApplication,
    TranscriptionOptions,
    TranscriptionOutcome,
    TranscriptionTask,
    run_transcription,
    resolve_transcription_options,
)
from core.project import Project
from core.transcription_models import TranscriptSegment


def _task(project: Project, cid: str) -> TranscriptionTask:
    clip = project.clips_by_id[cid]
    return transcription_task(
        clip, project.sources_by_id.get(clip.source_id), skip_existing=False
    )


def _task_data(task: TranscriptionTask) -> dict:
    value = asdict(task)
    value.pop("analysis_json")
    value["skip"] = False
    value["source_path"] = str(task.source_path) if task.source_path else None
    return value


def transcription_job_spec(
    project: Project,
    clip_ids: list[str] | None,
    options: TranscriptionOptions,
    *,
    arguments: dict,
) -> OperationSpec:
    ids = (
        list(dict.fromkeys(clip_ids))
        if clip_ids is not None
        else [c.id for c in project.clips]
    )
    if any(cid not in project.clips_by_id for cid in ids):
        raise ValueError("Unknown transcription clip ID")
    tasks = tuple(_task(project, cid) for cid in ids)
    revision = project.session.file_revision
    return transcription_operation_spec(
        tasks,
        options,
        arguments=arguments,
        persistence="job_history",
        session_id=project.session.session_id,
        input_revision=revision.digest if revision else None,
    )


def transcription_operation_spec(
    tasks: tuple[TranscriptionTask, ...],
    options: TranscriptionOptions,
    *,
    arguments: dict,
    persistence: Literal["job_history", "session_only"],
    session_id: str | None,
    input_revision: str | None,
) -> OperationSpec:
    """Describe detached transcription inputs for either runtime surface."""
    options = resolve_transcription_options(options)
    targets = []
    for task in tasks:
        target = asdict(task)
        target["source_path"] = str(task.source_path) if task.source_path else None
        target["media_stamp"] = _stamp(task.source_path) if task.source_path else None
        target["runtime"] = transcription_runtime(options)
        targets.append(target)
    return OperationSpec.build(
        kind="transcribe",
        version=2 if any(task.analysis_json is not None for task in tasks) else 1,
        arguments=arguments,
        inputs={
            "targets": targets,
            "options": asdict(options),
        },
        persistence=persistence,
        cancellable=True,
        session_id=session_id,
        input_revision=input_revision,
    )


class _OutcomeError(Exception):
    def __init__(self, outcome: TranscriptionOutcome):
        self.outcome = outcome


def run_transcription_job(
    store: JobStore,
    path: Path,
    clip_ids: list[str] | None,
    options: TranscriptionOptions,
    progress: Callable[[float, str], None],
    cancel: Event,
    *,
    operation: OperationSpec | None = None,
    force: bool = False,
    skip_existing: bool = False,
) -> dict:
    """Verify saved transcripts and recover receipts; force requests a new generation.

    ``skip_existing`` remains accepted for callers, but field presence alone does
    not establish reusable analysis. Explicitly edited managed text requires force.
    """
    options = resolve_transcription_options(options)
    media_fingerprints = MediaFingerprints(cancel)
    fingerprint = media_fingerprints.get
    fingerprints = AnalysisFingerprints(cancel, media_fingerprints=media_fingerprints)

    with result_batch(store, path) as batch:
        project = batch.project
        if operation is not None:
            from core.project_revision import ProjectRevisionConflict

            revision = project.session.file_revision
            if operation.input_revision is not None and (
                revision is None or revision.digest != operation.input_revision
            ):
                raise ProjectRevisionConflict(path)
            live = transcription_job_spec(
                project, clip_ids, options, arguments=operation.arguments
            )
            if live.inputs_json != operation.inputs_json:
                raise StaleJobResult(
                    "Transcription inputs changed while the job was queued"
                )
        ids = (
            list(dict.fromkeys(clip_ids))
            if clip_ids is not None
            else [c.id for c in project.clips]
        )
        if any(cid not in project.clips_by_id for cid in ids):
            raise ValueError("Unknown transcription clip ID")
        if force:
            batch.max_items = max(1, len(ids))
        managed: dict[str, list[tuple[dict, dict, dict]]] = {}
        for result_id in project.metadata.job_results:
            row = store.get_result(result_id)
            if row is None:
                continue
            if sha256(row["spec_json"].encode()).hexdigest() != result_id:
                raise StaleJobResult("Committed transcription identity is corrupt")
            identity = json.loads(row["spec_json"])
            if identity["kind"] == "transcribe":
                digest = sha256(row["payload_json"].encode()).hexdigest()
                if (
                    digest != row["payload_digest"]
                    or project.metadata.job_results[result_id] != digest
                ):
                    raise StaleJobResult("Committed transcription payload is corrupt")
                managed.setdefault(identity["target_id"], []).append(
                    (row, identity, json.loads(row["payload_json"]))
                )
        output: dict = {
            "succeeded": [],
            "failed": [],
            "skipped": [],
            "unprocessed": [],
            "total_clips": len(ids),
        }

        def inputs(current: Project, clip_id: str) -> dict:
            clip = current.clips_by_id.get(clip_id)
            if clip is None:
                return {"missing": True}
            task = _task(current, clip_id)
            return {
                "project_id": current.metadata.id,
                "source_id": clip.source_id,
                "task": _task_data(task),
                "media": fingerprint(task.source_path),
                "runtime": transcription_runtime(options),
            }

        def current_output(current: Project, cid: str) -> dict:
            clip = current.clips_by_id[cid]
            record = clip.analysis_records.get("transcribe")
            return {
                "segments": [s.to_dict() for s in clip.transcript]
                if clip.transcript is not None
                else None,
                "record_json": json.dumps(record.to_dict(), sort_keys=True)
                if record is not None
                else None,
            }

        def stage_record(cid: str, record: AnalysisRecord, basis: dict) -> None:
            batch.stage_analysis(
                apply=lambda current: current.record_analysis(
                    "clip", cid, "transcribe", record
                ),
                validate_input=lambda current: inputs(current, cid) == basis,
                is_applied=lambda current: current.clips_by_id[
                    cid
                ].analysis_records.get("transcribe")
                == record,
            )

        for index, clip_id in enumerate(ids):
            if cancel.is_set():
                output["unprocessed"].extend(
                    {"clip_id": cid, "code": "cancelled"} for cid in ids[index:]
                )
                break
            clip = project.clips_by_id[clip_id]
            if clip_id in managed and not force:
                current_segments = (
                    None
                    if clip.transcript is None
                    else [s.to_dict() for s in clip.transcript]
                )
                if not any(
                    payload["segments"] == current_segments
                    for _, _, payload in managed[clip_id]
                ):
                    raise StaleJobResult(
                        "Previously committed output changed; refusing stale replay"
                    )
            try:
                captured = inputs(project, clip_id)
                task = _task(project, clip_id)
                snapshot = (
                    AnalysisSnapshot.from_json(task.analysis_json)
                    if task.analysis_json
                    else None
                )
                runtime = captured["runtime"]
                if snapshot is not None and snapshot.inputs.unchanged():
                    from core.transcription import _has_audio_stream

                    if (
                        task.source_path is not None
                        and _has_audio_stream(task.source_path) is False
                    ):
                        runtime = transcription_runtime(
                            options,
                            execution={
                                "backend": "audio-probe",
                                "model": None,
                                "input_mode": "no-audio",
                            },
                        )
                semantic = (
                    transcription_identity(snapshot, options, fingerprints, runtime)
                    if snapshot is not None and snapshot.inputs.unchanged()
                    else None
                )
                reused = (
                    snapshot.reusable_record(semantic)
                    if snapshot is not None and semantic is not None and not force
                    else None
                )
                identity_inputs: dict = {
                    "basis": captured,
                    "previous_transcript": current_output(project, clip_id),
                }
                if force:
                    # A failed save leaves both the previous output and receipt
                    # count unchanged, so retries reuse its pending computation.
                    # A completed refresh advances the count even for silence.
                    identity_inputs["refresh_generation"] = len(
                        managed.get(clip_id, [])
                    )
                arguments = transcription_parameters(options)
                spec = ResultSpec.build(
                    path,
                    kind="transcribe",
                    version=2,
                    target_id=clip_id,
                    arguments=arguments,
                    inputs=identity_inputs,
                )
                candidates = [spec]
                matches = [
                    row
                    for row, identity, payload in managed.get(clip_id, [])
                    if identity["project_path"] == str(path.resolve())
                    and identity["arguments"] == arguments
                    and identity["inputs"].get("basis") == captured
                    and payload == current_output(project, clip_id)
                    and (
                        (force and not row["committed"])
                        or (not force and reused is not None)
                    )
                ]
                if matches:
                    candidates = [ResultSpec(path, row["spec_json"]) for row in matches]
                elif reused is not None:
                    if reused != clip.analysis_records.get("transcribe"):
                        stage_record(clip_id, reused, captured)
                    output["skipped"].append(
                        {"clip_id": clip_id, "reason": "valid_analysis"}
                    )
                    continue
                application = TranscriptionApplication(project, (task,), options)

                def compute(task=task):
                    outcome = run_transcription(
                        (task,), options, cancel_event=cancel, fingerprints=fingerprints
                    )[0]
                    if outcome.status != "succeeded":
                        raise _OutcomeError(outcome)
                    return {
                        "segments": [segment.to_dict() for segment in outcome.segments],
                        "record_json": outcome.record_json,
                    }

                def validate(current, clip_id=clip_id, captured=captured):
                    return inputs(current, clip_id) == captured

                def apply(current, payload, clip_id=clip_id, application=application):
                    outcome = TranscriptionOutcome(
                        clip_id,
                        "succeeded",
                        tuple(
                            TranscriptSegment.from_dict(item)
                            for item in payload["segments"]
                        ),
                        record_json=payload["record_json"],
                    )
                    if not application.apply(current, outcome):
                        raise StaleJobResult(
                            "Transcription target changed during application"
                        )

                def is_applied(current, payload, clip_id=clip_id):
                    return current_output(current, clip_id) == payload

                for candidate in candidates:
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
                            "clip_id": clip_id,
                            "segment_count": len(receipt["payload"]["segments"]),
                        }
                    )
                else:
                    output["skipped"].append(
                        {"clip_id": clip_id, "reason": "already_committed"}
                    )
            except _Cancelled:
                output["unprocessed"].extend(
                    {"clip_id": cid, "code": "cancelled"} for cid in ids[index:]
                )
                break
            except _OutcomeError as exc:
                outcome = exc.outcome
                if inputs(project, clip_id) != captured:
                    raise StaleJobResult(
                        "Transcription inputs changed during computation"
                    )
                if outcome.can_apply and outcome.record_json is not None:
                    record = AnalysisRecord.from_dict(json.loads(outcome.record_json))

                    def publish_failure(
                        current, application=application, outcome=outcome
                    ):
                        if not application.apply(current, outcome):
                            raise StaleJobResult(
                                "Transcription target changed during failure publication"
                            )

                    def failure_input(
                        current: Project, cid: str = clip_id, basis: dict = captured
                    ) -> bool:
                        return inputs(current, cid) == basis

                    def failure_applied(
                        current: Project,
                        cid: str = clip_id,
                        expected: AnalysisRecord = record,
                    ) -> bool:
                        return (
                            current.clips_by_id[cid].analysis_records.get("transcribe")
                            == expected
                        )

                    batch.stage_analysis(
                        apply=publish_failure,
                        validate_input=failure_input,
                        is_applied=failure_applied,
                    )
                bucket = "unprocessed" if outcome.status == "unprocessed" else "failed"
                output[bucket].append(
                    {
                        "clip_id": clip_id,
                        "code": outcome.code,
                        "message": outcome.message,
                    }
                )
                if outcome.critical:
                    output["unprocessed"].extend(
                        {"clip_id": cid, "code": "batch_aborted"}
                        for cid in ids[index + 1 :]
                    )
                    break
            progress(
                0.95 * (index + 1) / len(ids), f"Transcribing ({index + 1}/{len(ids)})"
            )
        batch.flush()
        progress(1.0, "Transcription finished")
        return {"success": True, "result": output}

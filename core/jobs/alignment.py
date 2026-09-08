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
    AlignmentOutcome,
    AlignmentTask,
    aligned_segments,
    needs_alignment,
    run_alignment,
    snapshot_alignment_tasks,
)
from core.operations.transcription import TranscriptionApplication, TranscriptionOutcome
from core.project import Project
from core.project_revision import ProjectRevisionConflict
from core.transcription_models import TranscriptSegment


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

    runtime = alignment_runtime() if any(task.analysis_json is not None for task in tasks) else None
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
    fingerprint = MediaFingerprints(cancel).get
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
        known: dict[str, list[tuple[dict, dict, dict]]] = {}
        for result_id, receipt_digest in project.metadata.job_results.items():
            row = store.get_result(result_id)
            if row is None:
                raise StaleJobResult("Committed result payload is missing")
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
            }

        def transcript(current: Project, cid: str):
            value = current.clips_by_id[cid].transcript
            return (
                [segment.to_dict() for segment in value] if value is not None else None
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
            existing = not needs_alignment(clip)
            if existing and not force and cid not in known:
                output["skipped"].append({"clip_id": cid, "reason": "already_aligned"})
                continue
            try:
                basis = inputs(project, cid)
                task = snapshot_alignment_tasks(
                    [clip], project.sources_by_id, skip_existing=False
                )[0]
                identity_inputs: dict = {"basis": basis}
                if force:
                    identity_inputs["refresh_generation"] = len(known.get(cid, []))
                    identity_inputs["previous_transcript"] = transcript(project, cid)
                spec = ResultSpec.build(
                    path,
                    kind="align_words",
                    version=1,
                    target_id=cid,
                    arguments={},
                    inputs=identity_inputs,
                )
                specs = [spec]
                if existing and not force:
                    matches = [
                        row
                        for row, identity, payload in known[cid]
                        if identity["project_path"] == str(path)
                        and identity["inputs"]["basis"] == basis
                        and payload["segments"] == transcript(project, cid)
                    ]
                    if not matches:
                        output["skipped"].append(
                            {"clip_id": cid, "reason": "already_aligned"}
                        )
                        continue
                    specs = [ResultSpec(path, row["spec_json"]) for row in matches]

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
                    outcome = run_alignment((task,), cancel_event=cancel)[0]
                    if outcome.status != "succeeded":
                        raise _OutcomeError(outcome)
                    return {
                        "segments": [
                            segment.to_dict()
                            for segment in aligned_segments(task, outcome.words)
                        ]
                    }

                def apply(current, payload, cid=cid):
                    target = snapshot_alignment_tasks(
                        [current.clips_by_id[cid]],
                        current.sources_by_id,
                        skip_existing=False,
                    )[0].target
                    result = TranscriptionOutcome(
                        cid,
                        "succeeded",
                        tuple(
                            TranscriptSegment.from_dict(value)
                            for value in payload["segments"]
                        ),
                    )
                    if not TranscriptionApplication(current, (target,)).apply(
                        current, result
                    ):
                        raise StaleJobResult(
                            "Alignment target changed during application"
                        )

                def validate(
                    current: Project, cid: str = cid, basis: dict = basis
                ) -> bool:
                    return inputs(current, cid) == basis

                def is_applied(current: Project, payload: dict, cid: str = cid) -> bool:
                    return bool(transcript(current, cid) == payload["segments"])

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

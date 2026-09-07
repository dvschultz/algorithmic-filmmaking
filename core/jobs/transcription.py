"""Restart-safe transcription receipts for saved project jobs."""

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.commits import ResultSpec, StaleJobResult, result_batch
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore
from core.operations.transcription import (
    TranscriptionApplication,
    TranscriptionOptions,
    TranscriptionOutcome,
    run_transcription,
    snapshot_tasks,
)
from core.project import Project
from core.transcription_models import TranscriptSegment


def _stamp(path: Path) -> tuple[int, ...] | None:
    try:
        stat = path.stat()
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
    except OSError:
        return None


def transcription_job_spec(
    project: Project,
    clip_ids: list[str] | None,
    options: TranscriptionOptions,
    *,
    arguments: dict,
) -> OperationSpec:
    from core.transcription import _resolve_backend

    ids = (
        list(dict.fromkeys(clip_ids))
        if clip_ids is not None
        else [c.id for c in project.clips]
    )
    if any(cid not in project.clips_by_id for cid in ids):
        raise ValueError("Unknown transcription clip ID")
    tasks = snapshot_tasks(
        [project.clips_by_id[cid] for cid in ids],
        project.sources_by_id,
        skip_existing=False,
    )
    targets = []
    for task in tasks:
        target = asdict(task)
        target["source_path"] = str(task.source_path) if task.source_path else None
        target["media_stamp"] = _stamp(task.source_path) if task.source_path else None
        targets.append(target)
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="transcribe",
        version=1,
        arguments=arguments,
        inputs={
            "targets": targets,
            "options": asdict(
                replace(options, backend=_resolve_backend(options.backend))
            ),
        },
        persistence="job_history",
        cancellable=True,
        session_id=project.session.session_id,
        input_revision=revision.digest if revision else None,
    )


class _OutcomeError(Exception):
    def __init__(self, outcome: TranscriptionOutcome):
        self.outcome = outcome


class _Cancelled(Exception):
    pass


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
    """Recover managed results, or explicitly skip/refresh populated targets.

    ``skip_existing`` preserves every populated transcript, including managed
    outputs from other models. ``force`` takes precedence for explicit refresh.
    """
    from core.transcription import _resolve_backend

    options = replace(options, backend=_resolve_backend(options.backend))
    fingerprints: dict[Path, tuple[tuple[int, ...], dict]] = {}

    def fingerprint(media: Path | None) -> dict | None:
        if media is None or (stamp := _stamp(media)) is None:
            return None
        cached = fingerprints.get(media)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        digest = sha256()
        with media.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                if cancel.is_set():
                    raise _Cancelled()
                digest.update(block)
        if _stamp(media) != stamp:
            raise StaleJobResult("Transcription media changed during fingerprinting")
        value = {"stamp": list(stamp), "sha256": digest.hexdigest()}
        fingerprints[media] = (stamp, value)
        return value

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
        managed: dict[str, list[dict]] = {}
        for result_id in project.metadata.job_results:
            row = store.get_result(result_id)
            if row is None:
                raise StaleJobResult("Committed result payload is missing")
            identity = json.loads(row["spec_json"])
            if identity["kind"] == "transcribe":
                digest = sha256(row["payload_json"].encode()).hexdigest()
                if (
                    digest != row["payload_digest"]
                    or project.metadata.job_results[result_id] != digest
                ):
                    raise StaleJobResult("Committed transcription payload is corrupt")
                managed.setdefault(identity["target_id"], []).append(
                    json.loads(row["payload_json"])
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
            task = snapshot_tasks([clip], current.sources_by_id, skip_existing=False)[0]
            value = asdict(task)
            value["source_path"] = str(task.source_path) if task.source_path else None
            return {
                "project_id": current.metadata.id,
                "source_id": clip.source_id,
                "task": value,
                "media": fingerprint(task.source_path),
            }

        for index, clip_id in enumerate(ids):
            if cancel.is_set():
                output["unprocessed"].extend(
                    {"clip_id": cid, "code": "cancelled"} for cid in ids[index:]
                )
                break
            clip = project.clips_by_id[clip_id]
            if skip_existing and not force and clip.transcript is not None:
                output["skipped"].append(
                    {"clip_id": clip_id, "reason": "already_populated"}
                )
                continue
            if clip_id in managed and not force:
                current_segments = (
                    None
                    if clip.transcript is None
                    else [s.to_dict() for s in clip.transcript]
                )
                if not any(
                    payload["segments"] == current_segments
                    for payload in managed[clip_id]
                ):
                    raise StaleJobResult(
                        "Previously committed output changed; refusing stale replay"
                    )
            if clip.transcript is not None and clip_id not in managed and not force:
                output["skipped"].append(
                    {"clip_id": clip_id, "reason": "already_populated"}
                )
                continue
            try:
                captured = inputs(project, clip_id)
                identity_inputs = captured
                if force:
                    # A failed save leaves both the previous output and receipt
                    # count unchanged, so retries reuse its pending computation.
                    # A completed refresh advances the count even for silence.
                    identity_inputs = {
                        **captured,
                        "refresh_generation": len(managed.get(clip_id, [])),
                        "previous_transcript": (
                            None
                            if clip.transcript is None
                            else [segment.to_dict() for segment in clip.transcript]
                        ),
                    }
                spec = ResultSpec.build(
                    path,
                    kind="transcribe",
                    version=1,
                    target_id=clip_id,
                    arguments=asdict(options),
                    inputs=identity_inputs,
                )
                task = snapshot_tasks(
                    [clip], project.sources_by_id, skip_existing=False
                )[0]

                def compute(task=task):
                    outcome = run_transcription((task,), options, cancel_event=cancel)[
                        0
                    ]
                    if outcome.status != "succeeded":
                        raise _OutcomeError(outcome)
                    return {
                        "segments": [segment.to_dict() for segment in outcome.segments]
                    }

                def validate(current, clip_id=clip_id, captured=captured):
                    return inputs(current, clip_id) == captured

                def apply(current, payload, clip_id=clip_id):
                    targets = snapshot_tasks(
                        [current.clips_by_id[clip_id]],
                        current.sources_by_id,
                        skip_existing=False,
                    )
                    outcome = TranscriptionOutcome(
                        clip_id,
                        "succeeded",
                        tuple(
                            TranscriptSegment.from_dict(item)
                            for item in payload["segments"]
                        ),
                    )
                    if not TranscriptionApplication(current, targets).apply(
                        current, outcome
                    ):
                        raise StaleJobResult(
                            "Transcription target changed during application"
                        )

                def is_applied(current, payload, clip_id=clip_id):
                    target = current.clips_by_id.get(clip_id)
                    return (
                        target is not None
                        and target.transcript is not None
                        and [segment.to_dict() for segment in target.transcript]
                        == payload["segments"]
                    )

                receipt = batch.commit(
                    spec,
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

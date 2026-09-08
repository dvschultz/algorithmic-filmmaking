"""Saved-project standalone audio transcription with replayable commits."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from typing import Callable

from core.jobs.commits import ResultSpec, StaleJobResult, canonical_json, result_batch
from core.jobs.media import FingerprintCancelled, MediaFingerprints, media_stamp
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore
from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot
from core.operations.transcription_records import transcription_parameters
from core.operations.audio_transcription import (
    AudioTranscriptionApplication,
    AudioTranscriptionOutcome,
    AudioTranscriptionTask,
    run_audio_transcription,
    audio_transcription_runtime as audio_transcription_runtime,
    audio_transcription_identity,
)
from core.operations.transcription import (
    TranscriptionOptions,
    resolve_transcription_options,
)
from core.project import Project


def resolve_audio_options(options: TranscriptionOptions) -> TranscriptionOptions:
    from math import isfinite

    if options.backend not in ("auto", "faster-whisper", "mlx-whisper", "groq"):
        raise ValueError("Unknown transcription backend")
    if not options.model or not options.model.strip():
        raise ValueError("Transcription model must be nonempty")
    if options.segmentation_mode not in (
        "backend",
        "silence",
        "whisper",
        "none",
        "sentence",
        "phrase",
        "fixed",
    ):
        raise ValueError("Unknown transcription segmentation mode")
    if not isfinite(options.segment_max_seconds) or options.segment_max_seconds <= 0:
        raise ValueError("Segment maximum duration must be positive and finite")
    return resolve_transcription_options(options)


def audio_transcription_job_spec(
    project: Project,
    audio_source_id: str,
    options: TranscriptionOptions,
    *,
    force: bool = False,
) -> OperationSpec:
    audio = project.get_audio_source(audio_source_id)
    if audio is None:
        raise ValueError(f"Unknown audio source: {audio_source_id}")
    options = resolve_audio_options(options)
    task = AudioTranscriptionTask.from_audio(audio, verified=True)
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="audio_transcribe",
        version=2,
        arguments={
            "audio_source_id": audio_source_id,
            "options": asdict(options),
            "force": force,
        },
        inputs={
            "audio": audio.to_dict(),
            "media_stamp": media_stamp(audio.file_path),
            "runtime": audio_transcription_runtime(task, options),
        },
        persistence="job_history",
        session_id=project.session.session_id,
        input_revision=revision.digest if revision else None,
    )


class _OutcomeError(Exception):
    def __init__(self, outcome: AudioTranscriptionOutcome) -> None:
        self.outcome = outcome


def run_audio_transcription_job(
    store: JobStore,
    path: Path,
    audio_source_id: str,
    options: TranscriptionOptions,
    progress: Callable[[float, str], None],
    cancel: Event,
    *,
    force: bool = False,
    operation: OperationSpec | None = None,
) -> dict:
    """Verify one audio transcript; force refreshes while retaining save recovery."""
    options = resolve_audio_options(options)
    fingerprints = MediaFingerprints(cancel)
    analysis_fingerprints = AnalysisFingerprints(
        cancel, media_fingerprints=fingerprints
    )
    try:
        with result_batch(store, path, max_items=1) as batch:
            project = batch.project
            live = audio_transcription_job_spec(
                project, audio_source_id, options, force=force
            )
            if operation is not None and (
                live.inputs_json != operation.inputs_json
                or live.arguments_json != operation.arguments_json
                or live.input_revision != operation.input_revision
            ):
                raise StaleJobResult("Audio transcription inputs changed while queued")
            audio = project.get_audio_source(audio_source_id)
            assert audio is not None
            if cancel.is_set():
                return {
                    "success": False,
                    "error": "cancelled",
                    "audio_source_id": audio_source_id,
                }

            def inputs(current: Project) -> dict:
                target = current.get_audio_source(audio_source_id)
                if target is None:
                    return {"missing": True}
                metadata = target.to_dict()
                metadata.pop("transcript", None)
                metadata.pop("analysis_records", None)
                return {
                    "project_id": current.metadata.id,
                    "audio": metadata,
                    "media": fingerprints.get(target.file_path),
                    "runtime": audio_transcription_runtime(
                        AudioTranscriptionTask.from_audio(target, verified=True),
                        options,
                    ),
                }

            def current_output(current: Project) -> dict:
                target = current.get_audio_source(audio_source_id)
                assert target is not None
                record = target.analysis_records.get("transcribe")
                return {
                    "segments": [s.to_dict() for s in target.transcript]
                    if target.transcript is not None
                    else None,
                    "record": record.to_dict() if record is not None else None,
                }

            def payload_output(payload: dict) -> dict:
                outcome = AudioTranscriptionOutcome.from_dict(payload)
                return {
                    "segments": [s.to_dict() for s in outcome.segments],
                    "record": json.loads(outcome.record_json)
                    if outcome.record_json is not None
                    else None,
                }

            captured = inputs(project)
            previous = current_output(project)
            known = []
            for result_id, digest in project.metadata.job_results.items():
                row = store.get_result(result_id)
                if row is None:
                    continue
                if sha256(row["spec_json"].encode()).hexdigest() != result_id:
                    raise StaleJobResult(
                        "Committed result identity is missing or corrupt"
                    )
                identity = json.loads(row["spec_json"])
                if (
                    identity["kind"] != "audio_transcribe"
                    or identity["target_id"] != audio_source_id
                ):
                    continue
                if (
                    sha256(row["payload_json"].encode()).hexdigest() != digest
                    or row["payload_digest"] != digest
                ):
                    raise StaleJobResult("Committed audio transcription is corrupt")
                known.append((row, identity))

            if (
                known
                and not force
                and not any(
                    payload_output(json.loads(row["payload_json"]))["segments"]
                    == previous["segments"]
                    for row, _ in known
                )
            ):
                raise StaleJobResult(
                    "Previously committed audio text changed; use force to replace it"
                )

            task = AudioTranscriptionTask.from_audio(
                audio, verified=True, skip_existing=False
            )
            application = AudioTranscriptionApplication(project, task, options)
            snapshot = (
                AnalysisSnapshot.from_json(task.analysis_json)
                if task.analysis_json
                else None
            )
            reused = None
            if not force and snapshot is not None and snapshot.inputs.unchanged():
                from core.transcription import _has_audio_stream

                runtime = audio_transcription_runtime(task, options)
                if _has_audio_stream(task.path) is False:
                    runtime = audio_transcription_runtime(
                        task,
                        options,
                        execution={
                            "backend": "audio-probe",
                            "model": None,
                            "input_mode": "no-audio",
                        },
                    )
                semantic = audio_transcription_identity(
                    snapshot, options, analysis_fingerprints, runtime
                )
                reused = snapshot.reusable_record(semantic)
                if inputs(project) != captured or not snapshot.inputs.unchanged():
                    raise StaleJobResult("Audio inputs changed during verification")
            if cancel.is_set():
                raise _OutcomeError(
                    AudioTranscriptionOutcome(audio_source_id, "unprocessed")
                )
            arguments = transcription_parameters(options)

            # A save may succeed before checkpointing fails. Reconcile that exact
            # saved output before treating force as a request for another generation.
            pending = next(
                (
                    row
                    for row, identity in known
                    if not row["committed"]
                    and identity["project_path"] == str(path.resolve())
                    and identity["inputs"].get("basis") == captured
                    and identity["arguments"] == arguments
                    and payload_output(json.loads(row["payload_json"])) == previous
                    and (force or reused is not None)
                ),
                None,
            )
            if reused is not None and pending is None:
                if reused != audio.analysis_records.get("transcribe"):
                    batch.stage_analysis(
                        apply=lambda current: current.record_analysis(
                            "audio", audio_source_id, "transcribe", reused
                        ),
                        validate_input=lambda current: inputs(current) == captured,
                        is_applied=lambda current: current_output(current)["record"]
                        == reused.to_dict(),
                    )
                return {
                    "success": True,
                    "result": {
                        "audio_source_id": audio_source_id,
                        "status": "skipped",
                        "reason": "valid_analysis",
                        "segment_count": len(previous["segments"]),
                    },
                }
            spec = (
                ResultSpec(path.resolve(), pending["spec_json"])
                if pending
                else ResultSpec.build(
                    path,
                    kind="audio_transcribe",
                    version=2,
                    target_id=audio_source_id,
                    arguments=arguments,
                    inputs={
                        "basis": captured,
                        "generation": len(known),
                        "previous_transcript": previous,
                    },
                )
            )

            def compute() -> dict:
                outcome = run_audio_transcription(
                    task,
                    options,
                    cancel_event=cancel,
                    fingerprints=analysis_fingerprints,
                    progress=lambda n, total: progress(
                        n / total if total else 1.0, "Transcribing audio"
                    ),
                )
                if outcome.status != "succeeded" or cancel.is_set():
                    raise _OutcomeError(outcome)
                payload: dict = json.loads(canonical_json(asdict(outcome)))
                AudioTranscriptionOutcome.from_dict(payload)
                return payload

            def apply(current: Project, payload: dict) -> None:
                if cancel.is_set():
                    raise _OutcomeError(
                        AudioTranscriptionOutcome(audio_source_id, "unprocessed")
                    )
                if not application.apply(
                    current, AudioTranscriptionOutcome.from_dict(payload)
                ):
                    raise StaleJobResult("Audio transcription target changed")

            def is_applied(current: Project, payload: dict) -> bool:
                outcome = AudioTranscriptionOutcome.from_dict(payload)
                target = current.get_audio_source(audio_source_id)
                return bool(
                    outcome.status == "succeeded"
                    and outcome.audio_source_id == audio_source_id
                    and target is not None
                    and target.transcript is not None
                    and current_output(current) == payload_output(payload)
                )

            try:
                receipt = batch.commit(
                    spec,
                    compute=compute,
                    validate_input=lambda current: inputs(current) == captured,
                    apply=apply,
                    is_applied=is_applied,
                )
            except _OutcomeError as exc:
                outcome = exc.outcome
                if (
                    outcome.can_apply
                    and outcome.record_json is not None
                    and not cancel.is_set()
                ):
                    expected_record = json.loads(outcome.record_json)

                    def publish_failure(current: Project) -> None:
                        if not application.apply(current, outcome):
                            raise StaleJobResult(
                                "Audio changed during failure publication"
                            )

                    batch.stage_analysis(
                        apply=publish_failure,
                        validate_input=lambda current: inputs(current) == captured,
                        is_applied=lambda current: current_output(current)["record"]
                        == expected_record,
                    )
                return {
                    "success": False,
                    "audio_source_id": audio_source_id,
                    "error": "cancelled"
                    if cancel.is_set()
                    else outcome.message or "Audio transcription failed",
                }
            progress(1.0, "Audio transcription saved")
            return {
                "success": True,
                "result": {
                    "audio_source_id": audio_source_id,
                    "status": "succeeded" if receipt["applied"] else "recovered",
                    "segment_count": len(receipt["payload"]["segments"]),
                },
            }
    except (FingerprintCancelled, _OutcomeError) as exc:
        message = exc.outcome.message if isinstance(exc, _OutcomeError) else None
        return {
            "success": False,
            "audio_source_id": audio_source_id,
            "error": "cancelled"
            if cancel.is_set()
            else message or "Audio transcription failed",
        }

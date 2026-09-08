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
from core.operations.audio_transcription import (
    AudioTranscriptionApplication,
    AudioTranscriptionOutcome,
    AudioTranscriptionTask,
    run_audio_transcription,
)
from core.operations.transcription import TranscriptionOptions, resolve_transcription_options
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


def audio_transcription_runtime() -> dict:
    """Identify installed transcription providers and FFmpeg without loading them."""
    from importlib.metadata import PackageNotFoundError, version
    from core.binary_resolver import find_binary

    packages: dict[str, str | None] = {}
    for package in (
        "faster-whisper",
        "ctranslate2",
        "lightning-whisper-mlx",
        "mlx-whisper",
        "mlx",
        "groq",
        "numpy",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    binary = find_binary("ffmpeg")
    return {
        "algorithm": "audio-transcription/v1",
        "packages": packages,
        "ffmpeg": str(binary) if binary else None,
        "ffmpeg_stamp": list(media_stamp(Path(binary)) or ()) if binary else None,
    }


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
    revision = project.session.file_revision
    return OperationSpec.build(
        kind="audio_transcribe",
        version=1,
        arguments={
            "audio_source_id": audio_source_id,
            "options": asdict(options),
            "force": force,
        },
        inputs={
            "audio": audio.to_dict(),
            "media_stamp": media_stamp(audio.file_path),
            "runtime": audio_transcription_runtime(),
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
    """Transcribe one exact audio ID; preserve existing text unless force is set."""
    options = resolve_audio_options(options)
    fingerprints = MediaFingerprints(cancel)
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
                    "runtime": audio_transcription_runtime(),
                }

            captured = inputs(project)
            previous = (
                [s.to_dict() for s in audio.transcript]
                if audio.transcript is not None
                else None
            )
            known = []
            for result_id, digest in project.metadata.job_results.items():
                row = store.get_result(result_id)
                if (
                    row is None
                    or sha256(row["spec_json"].encode()).hexdigest() != result_id
                ):
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

            # A save may succeed before checkpointing fails. Reconcile that exact
            # saved output before treating force as a request for another generation.
            pending = next(
                (
                    row
                    for row, identity in known
                    if not row["committed"]
                    and identity["project_path"] == str(path.resolve())
                    and identity["inputs"]["basis"] == captured
                    and identity["arguments"] == asdict(options)
                    and [
                        s.to_dict()
                        for s in AudioTranscriptionOutcome.from_dict(
                            json.loads(row["payload_json"])
                        ).segments
                    ]
                    == previous
                ),
                None,
            )
            if previous is not None and not force and pending is None:
                return {
                    "success": True,
                    "result": {
                        "audio_source_id": audio_source_id,
                        "status": "skipped",
                        "reason": "already_populated",
                        "segment_count": len(previous),
                    },
                }
            spec = (
                ResultSpec(path.resolve(), pending["spec_json"])
                if pending
                else ResultSpec.build(
                    path,
                    kind="audio_transcribe",
                    version=1,
                    target_id=audio_source_id,
                    arguments=asdict(options),
                    inputs={
                        "basis": captured,
                        "generation": len(known),
                        "previous_transcript": previous,
                    },
                )
            )
            task = AudioTranscriptionTask.from_audio(audio)
            application = AudioTranscriptionApplication(project, task)

            def compute() -> dict:
                outcome = run_audio_transcription(
                    task,
                    options,
                    cancel_event=cancel,
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
                    and [s.to_dict() for s in target.transcript]
                    == [s.to_dict() for s in outcome.segments]
                )

            receipt = batch.commit(
                spec,
                compute=compute,
                validate_input=lambda current: inputs(current) == captured,
                apply=apply,
                is_applied=is_applied,
            )
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

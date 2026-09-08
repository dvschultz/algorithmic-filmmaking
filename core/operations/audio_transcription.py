"""Detached standalone-audio transcription and owner-thread publication."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
from math import isfinite
from threading import Event
from typing import TYPE_CHECKING, Callable

from core.operations.contracts import OutcomeStatus
from core.operations.transcription import (
    TranscriptionOptions,
    _media_stamp,
    resolve_transcription_options,
)
from core.analysis_records import AnalysisFingerprints, AnalysisSnapshot
from core.operations.transcription_records import (
    transcription_value,
    transcription_parameters,
    transcription_runtime,
    transcription_segments_value,
)
from models.analysis_record import AnalysisRecord, AnalysisIdentity

if TYPE_CHECKING:
    from core.project import Project
    from core.transcription_models import TranscriptSegment
    from models.audio_source import AudioSource


@dataclass(frozen=True)
class AudioTranscriptionTask:
    audio_source_id: str
    path: Path
    media_stamp: tuple[int, int, int, int, int] | None
    analysis_json: str | None = None
    skip: bool = False

    @classmethod
    def from_audio(
        cls, audio: AudioSource, *, verified: bool = False, skip_existing: bool = True
    ) -> AudioTranscriptionTask:
        path = Path(audio.file_path)
        snapshot = (
            AnalysisSnapshot.capture(
                audio,
                "transcribe",
                {"audio": path},
                {
                    "duration_seconds": audio.duration_seconds,
                    "sample_rate": audio.sample_rate,
                    "channels": audio.channels,
                },
                transcription_value(audio),
            )
            if verified
            else None
        )
        return cls(
            audio.id,
            path,
            _media_stamp(path),
            snapshot.to_json() if snapshot else None,
            verified and skip_existing,
        )


def audio_transcription_runtime(
    task: AudioTranscriptionTask,
    options: TranscriptionOptions,
    *,
    execution: dict | None = None,
) -> dict:
    """Identify whole-file decoding separately from clip-range extraction."""
    runtime = transcription_runtime(options, execution=execution)
    backend = runtime["execution"]["backend"]
    direct = backend in ("audio-probe", "faster-whisper") or (
        backend == "groq"
        and task.path.suffix.lower()
        in (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm")
    )
    if direct:
        runtime["binaries"].pop("ffmpeg", None)
    runtime["extraction"] = (
        "no-audio/v1"
        if backend == "audio-probe"
        else "pyav-direct/v1"
        if backend == "faster-whisper"
        else "container-upload/v1"
        if direct
        else "whole-file-pcm-s16le-16000-mono/v1"
    )
    return runtime


def audio_transcription_identity(
    snapshot: AnalysisSnapshot,
    options: TranscriptionOptions,
    fingerprints: AnalysisFingerprints,
    runtime: dict,
) -> AnalysisIdentity:
    return fingerprints.identity(
        snapshot.inputs,
        operation="transcribe",
        operation_version=2,
        model=runtime,
        parameters=transcription_parameters(options),
        sampling={"policy": "whole-audio-file/v1"},
    )


@dataclass(frozen=True)
class AudioTranscriptionOutcome:
    audio_source_id: str
    status: OutcomeStatus
    segments: tuple[TranscriptSegment, ...] = ()
    message: str | None = None
    record_json: str | None = None

    @property
    def has_result(self) -> bool:
        return self.status == "succeeded" or (
            self.status == "skipped" and self.record_json is not None
        )

    @property
    def can_apply(self) -> bool:
        return self.has_result or (
            self.status == "failed" and self.record_json is not None
        )

    @classmethod
    def from_dict(cls, data: dict) -> AudioTranscriptionOutcome:
        """Decode durable results without silently defaulting malformed segments."""
        from core.transcription_models import TranscriptSegment

        if (
            not isinstance(data.get("audio_source_id"), str)
            or not data["audio_source_id"]
            or data.get("status")
            not in ("succeeded", "failed", "skipped", "unprocessed")
            or not isinstance(data.get("segments"), list)
        ):
            raise ValueError("Invalid audio transcription outcome")
        for segment in data["segments"]:
            if not isinstance(segment, dict) or not isinstance(
                segment.get("text"), str
            ):
                raise ValueError("Invalid audio transcript segment")
            for key in ("start_time", "end_time", "confidence"):
                value = segment.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not isfinite(value)
                ):
                    raise ValueError("Invalid audio transcript timing or confidence")
            if segment["start_time"] < 0 or segment["end_time"] < segment["start_time"]:
                raise ValueError("Invalid audio transcript range")
        outcome = cls(
            data["audio_source_id"],
            data["status"],
            tuple(TranscriptSegment.from_dict(segment) for segment in data["segments"]),
            data.get("message"),
            data.get("record_json"),
        )
        transcription_segments_value(outcome.segments)
        return outcome


def _run_audio_transcription(
    task: AudioTranscriptionTask,
    options: TranscriptionOptions,
    *,
    cancel_event: Event | None = None,
    progress: Callable[[int, int], None] | None = None,
    on_execution: Callable[[dict[str, str | None]], None] | None = None,
) -> AudioTranscriptionOutcome:
    """Compute without project access; cancellation closes result publication.

    Existing native backends cannot be interrupted mid-call. We retain their
    worker until they return and discard results after cancellation.
    """
    cancelled = cancel_event or Event()
    if cancelled.is_set():
        return AudioTranscriptionOutcome(task.audio_source_id, "unprocessed")
    if task.media_stamp is None:
        return AudioTranscriptionOutcome(
            task.audio_source_id,
            "failed",
            message=f"Audio file is missing on disk: {task.path.name}",
        )
    if _media_stamp(task.path) != task.media_stamp:
        return AudioTranscriptionOutcome(
            task.audio_source_id,
            "failed",
            message="Audio file changed before transcription",
        )

    def report(fraction: float, _message: str = "") -> None:
        if progress and not cancelled.is_set():
            progress(max(0, min(100, int(fraction * 100))), 100)

    try:
        from core.transcription import transcribe_video

        options = resolve_transcription_options(options)
        report(0)
        if cancelled.is_set():
            return AudioTranscriptionOutcome(task.audio_source_id, "unprocessed")
        segments = transcribe_video(
            task.path,
            model_name=options.model,
            language=options.language or "auto",
            backend=options.backend,
            segmentation_mode=options.segmentation_mode,
            segment_max_seconds=options.segment_max_seconds,
            progress_callback=report,
            cloud_model=options.cloud_model,
            on_execution=on_execution,
        )
        if cancelled.is_set():
            return AudioTranscriptionOutcome(task.audio_source_id, "unprocessed")
        if _media_stamp(task.path) != task.media_stamp:
            raise ValueError("Audio file changed during transcription")
        report(1)
        return AudioTranscriptionOutcome(
            task.audio_source_id,
            "succeeded",
            tuple(deepcopy(segments)),
        )
    except Exception as exc:
        if cancelled.is_set():
            return AudioTranscriptionOutcome(task.audio_source_id, "unprocessed")
        return AudioTranscriptionOutcome(
            task.audio_source_id,
            "failed",
            message=f"Transcription failed: {exc}",
        )


def run_audio_transcription(
    task: AudioTranscriptionTask,
    options: TranscriptionOptions,
    *,
    cancel_event: Event | None = None,
    progress: Callable[[int, int], None] | None = None,
    fingerprints: AnalysisFingerprints | None = None,
) -> AudioTranscriptionOutcome:
    """Verify whole-audio records before inference and retain failed attempts."""
    cancel = cancel_event or Event()
    if task.analysis_json is None or task.media_stamp is None or cancel.is_set():
        return _run_audio_transcription(
            task, options, cancel_event=cancel, progress=progress
        )
    try:
        from core.transcription import _has_audio_stream
        from core.transcription_models import TranscriptSegment

        options = resolve_transcription_options(options)
        snapshot = AnalysisSnapshot.from_json(task.analysis_json)
        fingerprints = fingerprints or AnalysisFingerprints(cancel)
        if not snapshot.inputs.unchanged():
            raise ValueError("Audio file changed before transcription")
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
        identity = audio_transcription_identity(
            snapshot, options, fingerprints, runtime
        )
        reused = snapshot.reusable_record(identity) if task.skip else None
        if reused is not None:
            if cancel.is_set():
                return AudioTranscriptionOutcome(task.audio_source_id, "unprocessed")
            if (
                not snapshot.inputs.unchanged()
                or audio_transcription_runtime(
                    task, options, execution=runtime["execution"]
                )
                != runtime
            ):
                raise ValueError("Audio inputs changed during verification")
            return AudioTranscriptionOutcome(
                task.audio_source_id,
                "skipped",
                tuple(
                    TranscriptSegment.from_dict(s) for s in reused.value["transcript"]
                ),
                record_json=json.dumps(reused.to_dict(), sort_keys=True),
            )
        actual_runtime = None

        def execution(value: dict[str, str | None]) -> None:
            nonlocal actual_runtime
            actual_runtime = audio_transcription_runtime(
                task, options, execution=dict(value)
            )

        outcome = _run_audio_transcription(
            task,
            options,
            cancel_event=cancel,
            progress=progress,
            on_execution=execution,
        )
        if cancel.is_set() or outcome.status == "unprocessed":
            return AudioTranscriptionOutcome(task.audio_source_id, "unprocessed")
        actual = actual_runtime if actual_runtime is not None else runtime
        if (
            not snapshot.inputs.unchanged()
            or audio_transcription_runtime(
                task, options, execution=runtime["execution"]
            )
            != runtime
            or audio_transcription_runtime(task, options, execution=actual["execution"])
            != actual
        ):
            raise ValueError("Audio media or runtime changed during transcription")
        if outcome.status == "succeeded":
            try:
                transcription_segments_value(outcome.segments)
            except (ValueError, TypeError, AttributeError) as exc:
                outcome = replace(
                    outcome, status="failed", segments=(), message=str(exc)
                )
        identity = audio_transcription_identity(snapshot, options, fingerprints, actual)
        record = (
            AnalysisRecord.success(
                identity,
                transcription_segments_value(outcome.segments),
                input_snapshot=snapshot.inputs.to_dict(),
            )
            if outcome.status == "succeeded"
            else replace(
                AnalysisRecord.failure(
                    identity, outcome.message or "Audio transcription failed"
                ),
                input_json=json.dumps(snapshot.inputs.to_dict(), sort_keys=True),
            )
        )
        return replace(
            outcome, record_json=json.dumps(record.to_dict(), sort_keys=True)
        )
    except Exception as exc:
        return AudioTranscriptionOutcome(
            task.audio_source_id,
            "unprocessed" if cancel.is_set() else "failed",
            message=str(exc),
        )


class AudioTranscriptionApplication:
    """Bind a single result to unchanged audio in the original project session."""

    def __init__(
        self,
        project: Project,
        task: AudioTranscriptionTask,
        options: TranscriptionOptions | None = None,
    ) -> None:
        project.session.assert_owner()
        self.project = project
        self.session_id = project.session.session_id
        self.path = project.path.resolve() if project.path is not None else None
        self.task = task
        self.options = (
            resolve_transcription_options(options) if options is not None else None
        )
        self.audio = project.get_audio_source(task.audio_source_id)
        self.expected = deepcopy(self.audio.to_dict()) if self.audio else None
        self.consumed = False

    def is_current(self, project: Project) -> bool:
        return (
            project is self.project
            and project.session.session_id == self.session_id
            and (project.path.resolve() if project.path is not None else None)
            == self.path
        )

    def apply(self, project: Project, outcome: AudioTranscriptionOutcome) -> bool:
        if (
            not self.is_current(project)
            or self.consumed
            or not outcome.can_apply
            or outcome.audio_source_id != self.task.audio_source_id
        ):
            return False

        def publish() -> bool:
            self.consumed = True
            if (
                self.audio is None
                or project.get_audio_source(outcome.audio_source_id) is not self.audio
                or self.audio.to_dict() != self.expected
                or self.audio.file_path != self.task.path
                or self.task.media_stamp is None
                or _media_stamp(self.task.path) != self.task.media_stamp
            ):
                return False
            try:
                value = transcription_segments_value(outcome.segments)
                record = (
                    AnalysisRecord.from_dict(json.loads(outcome.record_json))
                    if outcome.record_json is not None
                    else AnalysisRecord.legacy(value)
                )
                if outcome.record_json is not None:
                    if self.task.analysis_json is None:
                        return False
                    snapshot = AnalysisSnapshot.from_json(self.task.analysis_json)
                    current_json = AudioTranscriptionTask.from_audio(
                        self.audio, verified=True
                    ).analysis_json
                    previous = self.audio.analysis_records.get("transcribe")
                    if not isinstance(previous, AnalysisRecord):
                        previous = None
                    if (
                        record.identity is None
                        or record.identity.operation != "transcribe"
                        or record.identity.to_dict()["operation_version"] != 2
                        or record.identity.to_dict()["schema_version"] != 1
                        or record.identity.to_dict()["sampling"]
                        != {"policy": "whole-audio-file/v1"}
                        or previous != snapshot.record
                        or transcription_value(self.audio)
                        != json.loads(snapshot.value_json)
                        or current_json is None
                        or snapshot.inputs
                        != AnalysisSnapshot.from_json(current_json).inputs
                        or json.loads(record.input_json or "null")
                        != snapshot.inputs.to_dict()
                        or record.identity.to_dict()["source_range"]
                        != json.loads(snapshot.inputs.range_json)
                        or (
                            self.options is not None
                            and record.identity.to_dict()["parameters"]
                            != transcription_parameters(self.options)
                        )
                        or (
                            outcome.has_result
                            and (record.state != "succeeded" or record.value != value)
                        )
                        or (outcome.status == "failed" and record.state != "failed")
                    ):
                        return False
            except (ValueError, TypeError, KeyError, AttributeError):
                return False
            if outcome.status == "succeeded":
                project.set_audio_transcript(
                    outcome.audio_source_id,
                    list(deepcopy(outcome.segments)),
                    analysis_record=record,
                )
            else:
                project.record_analysis(
                    "audio", outcome.audio_source_id, "transcribe", record
                )
            return True

        return project.session.apply_external(publish)

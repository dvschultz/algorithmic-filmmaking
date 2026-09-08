"""Detached standalone-audio transcription and owner-thread publication."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
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

if TYPE_CHECKING:
    from core.project import Project
    from core.transcription_models import TranscriptSegment
    from models.audio_source import AudioSource


@dataclass(frozen=True)
class AudioTranscriptionTask:
    audio_source_id: str
    path: Path
    media_stamp: tuple[int, int, int, int, int] | None

    @classmethod
    def from_audio(cls, audio: AudioSource) -> AudioTranscriptionTask:
        path = Path(audio.file_path)
        return cls(audio.id, path, _media_stamp(path))


@dataclass(frozen=True)
class AudioTranscriptionOutcome:
    audio_source_id: str
    status: OutcomeStatus
    segments: tuple[TranscriptSegment, ...] = ()
    message: str | None = None

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
        return cls(
            data["audio_source_id"],
            data["status"],
            tuple(TranscriptSegment.from_dict(segment) for segment in data["segments"]),
            data.get("message"),
        )


def run_audio_transcription(
    task: AudioTranscriptionTask,
    options: TranscriptionOptions,
    *,
    cancel_event: Event | None = None,
    progress: Callable[[int, int], None] | None = None,
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


class AudioTranscriptionApplication:
    """Bind a single result to unchanged audio in the original project session."""

    def __init__(self, project: Project, task: AudioTranscriptionTask) -> None:
        project.session.assert_owner()
        self.project = project
        self.session_id = project.session.session_id
        self.path = project.path.resolve() if project.path is not None else None
        self.task = task
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
            or outcome.status != "succeeded"
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
            project.set_audio_transcript(
                outcome.audio_source_id, list(deepcopy(outcome.segments))
            )
            return True

        return project.session.apply_external(publish)

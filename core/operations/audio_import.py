"""Detached audio probing and owner-bound import publication."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from core.operations.contracts import OutcomeStatus
from core.operations.transcription import _media_stamp

if TYPE_CHECKING:
    from core.project import Project
    from models.audio_source import AudioSource


@dataclass(frozen=True)
class AudioImportTask:
    audio_source_id: str
    path: Path
    media_stamp: tuple[int, ...] | None

    @classmethod
    def from_path(cls, path: Path) -> "AudioImportTask":
        path = path.expanduser().resolve()
        return cls(uuid4().hex, path, _media_stamp(path))


@dataclass(frozen=True)
class AudioImportOutcome:
    audio_source_id: str
    status: OutcomeStatus
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    message: str | None = None

    def to_model(self, task: AudioImportTask) -> "AudioSource":
        from models.audio_source import AudioSource

        return AudioSource(
            id=self.audio_source_id,
            file_path=task.path,
            duration_seconds=self.duration,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )


def run_audio_import(
    task: AudioImportTask,
    *,
    cancel_event: Event | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> AudioImportOutcome:
    from core.audio_formats import is_audio_file

    cancel = cancel_event or Event()
    try:
        if cancel.is_set():
            return AudioImportOutcome(task.audio_source_id, "unprocessed")
        if not task.path.is_file():
            raise ValueError(f"File not found: {task.path}")
        if not is_audio_file(task.path):
            raise ValueError(
                f"Unsupported audio format: {task.path.suffix or '<no extension>'}"
            )
        if task.media_stamp is None or _media_stamp(task.path) != task.media_stamp:
            raise ValueError("Audio file changed while queued")
        if progress:
            progress(0, 1)
        from core.ffmpeg import FFmpegProcessor

        try:
            processor = FFmpegProcessor()
        except RuntimeError as exc:
            raise ValueError(f"FFmpeg unavailable: {exc}") from exc
        if not processor.ffprobe_available:
            raise ValueError("FFprobe is not available")
        try:
            info = processor.get_audio_info(task.path)
        except ValueError as exc:
            raise ValueError(f"Not an audio file: {task.path.name}") from exc
        except RuntimeError as exc:
            raise ValueError(f"Failed to probe audio: {exc}") from exc
        if cancel.is_set():
            return AudioImportOutcome(task.audio_source_id, "unprocessed")
        if _media_stamp(task.path) != task.media_stamp:
            raise ValueError("Audio file changed during probing")
        duration = info.get("duration", 0.0)
        rate, channels = info.get("sample_rate", 0), info.get("channels", 0)
        if (
            type(duration) not in (int, float)
            or not isfinite(duration)
            or duration <= 0
        ):
            raise ValueError(
                f"Audio file has invalid or zero duration: {task.path.name}"
            )
        if any(type(value) is not int or value < 0 for value in (rate, channels)):
            raise ValueError(
                "Audio probe returned invalid sample rate or channel count"
            )
        if progress:
            progress(1, 1)
        if cancel.is_set():
            return AudioImportOutcome(task.audio_source_id, "unprocessed")
        return AudioImportOutcome(
            task.audio_source_id, "succeeded", float(duration), rate, channels
        )
    except Exception as exc:
        return AudioImportOutcome(
            task.audio_source_id,
            "unprocessed" if cancel.is_set() else "failed",
            message=str(exc),
        )


class AudioImportApplication:
    def __init__(self, project: "Project", task: AudioImportTask) -> None:
        project.session.assert_owner()
        self.project = project
        self.session_id = project.session.session_id
        self.path = project.path.resolve() if project.path else None
        self.task = task
        self.consumed = False
        self.audio: "AudioSource | None" = None

    def is_current(self, project: "Project") -> bool:
        return (
            project is self.project
            and project.session.session_id == self.session_id
            and (project.path.resolve() if project.path else None) == self.path
        )

    def apply(self, project: "Project", outcome: AudioImportOutcome) -> bool:
        if (
            not self.is_current(project)
            or self.consumed
            or outcome.audio_source_id != self.task.audio_source_id
            or outcome.status != "succeeded"
        ):
            return False

        def publish() -> bool:
            self.consumed = True
            if (
                self.task.media_stamp is None
                or _media_stamp(self.task.path) != self.task.media_stamp
            ):
                return False
            existing = next(
                (
                    audio
                    for audio in project.audio_sources
                    if audio.file_path.expanduser().resolve() == self.task.path
                ),
                None,
            )
            if existing is not None:
                self.audio = existing
                return False
            if project.get_audio_source(outcome.audio_source_id) is not None:
                raise ValueError("Imported audio ID already exists")
            self.audio = outcome.to_model(self.task)
            project.add_audio_source(self.audio)
            return True

        return project.session.apply_external(publish)

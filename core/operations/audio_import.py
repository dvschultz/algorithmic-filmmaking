"""Detached audio probing and owner-bound import publication."""

from dataclasses import asdict, dataclass, replace
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

    def to_dict(self) -> dict:
        return {**asdict(self), "path": str(self.path)}

    @classmethod
    def from_dict(cls, data: dict) -> "AudioImportTask":
        aid, path, stamp = (
            data.get("audio_source_id"),
            data.get("path"),
            data.get("media_stamp"),
        )
        if (
            not isinstance(aid, str)
            or not aid
            or not isinstance(path, str)
            or not Path(path).is_absolute()
        ):
            raise ValueError("Invalid audio import identity or path")
        if stamp is not None and (
            not isinstance(stamp, (list, tuple))
            or len(stamp) != 5
            or any(type(value) is not int for value in stamp)
        ):
            raise ValueError("Invalid audio import media stamp")
        return cls(aid, Path(path), tuple(stamp) if stamp is not None else None)


@dataclass(frozen=True)
class AudioImportOutcome:
    audio_source_id: str
    status: OutcomeStatus
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    message: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "AudioImportOutcome":
        aid, status = data.get("audio_source_id"), data.get("status")
        if (
            not isinstance(aid, str)
            or not aid
            or status not in ("succeeded", "failed", "unprocessed")
        ):
            raise ValueError("Invalid audio import outcome identity or status")
        duration, rate, channels = (
            data.get("duration", 0.0),
            data.get("sample_rate", 0),
            data.get("channels", 0),
        )
        if (
            type(duration) not in (int, float)
            or not isfinite(duration)
            or duration < 0
            or (status == "succeeded" and duration == 0)
            or any(type(value) is not int or value < 0 for value in (rate, channels))
        ):
            raise ValueError("Invalid audio import metadata")
        message = data.get("message")
        if message is not None and not isinstance(message, str):
            raise ValueError("Invalid audio import message")
        return cls(aid, status, float(duration), rate, channels, message)

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

    def apply(
        self,
        project: "Project",
        outcome: AudioImportOutcome,
        *,
        recovered_task: AudioImportTask | None = None,
    ) -> bool:
        task = recovered_task or self.task
        if replace(self.task, audio_source_id=task.audio_source_id) != task:
            return False
        if (
            not self.is_current(project)
            or self.consumed
            or outcome.audio_source_id != task.audio_source_id
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
            self.audio = outcome.to_model(task)
            project.add_audio_source(self.audio)
            return True

        return project.session.apply_external(publish)

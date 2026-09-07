"""Detached frame extraction with isolated artifacts and owner publication."""

from dataclasses import asdict, dataclass, replace
import logging
from pathlib import Path
import shutil
from threading import Event
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from core.operations.contracts import OutcomeStatus
from core.operations.transcription import _media_stamp

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip, Source
    from models.frame import Frame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameExtractionTask:
    request_id: str
    source_id: str
    path: Path
    fps: float
    width: int
    height: int
    duration: float
    clip_id: str | None
    start_frame: int
    end_frame: int | None
    mode: str
    interval: int
    artifact_dir: Path
    media_stamp: tuple[int, ...] | None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["path"] = str(self.path)
        data["artifact_dir"] = str(self.artifact_dir)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "FrameExtractionTask":
        from math import isfinite

        values = dict(data)
        for name in ("path", "artifact_dir"):
            if not isinstance(values.get(name), str) or not values[name]:
                raise ValueError("Invalid extraction path")
            values[name] = Path(values[name])
        values["media_stamp"] = _decode_stamp(values.get("media_stamp"))
        task = cls(**values)
        if (
            not isinstance(task.request_id, str)
            or len(task.request_id) != 32
            or any(c not in "0123456789abcdef" for c in task.request_id)
            or not isinstance(task.source_id, str)
            or not task.source_id
            or (
                task.clip_id is not None
                and (not isinstance(task.clip_id, str) or not task.clip_id)
            )
            or not task.artifact_dir.is_absolute()
            or task.artifact_dir.name != task.request_id
            or task.mode not in ("all", "interval", "smart")
            or type(task.interval) is not int
            or task.interval < 1
            or type(task.start_frame) is not int
            or task.start_frame < 0
            or (
                task.end_frame is not None
                and (
                    type(task.end_frame) is not int
                    or task.end_frame <= task.start_frame
                )
            )
            or type(task.fps) not in (float, int)
            or not isfinite(task.fps)
            or task.fps <= 0
            or type(task.duration) not in (float, int)
            or not isfinite(task.duration)
            or task.duration < 0
            or any(
                type(value) is not int or value < 0
                for value in (task.width, task.height)
            )
        ):
            raise ValueError("Invalid extraction task")
        return task

    @classmethod
    def from_source(
        cls,
        source: "Source",
        clip: "Clip | None",
        mode: str,
        interval: int,
        output_dir: Path,
    ) -> "FrameExtractionTask":
        if clip is not None and clip.source_id != source.id:
            raise ValueError("Clip belongs to another source")
        request_id = uuid4().hex
        path = Path(source.file_path)
        return cls(
            request_id,
            source.id,
            path,
            source.fps,
            source.width,
            source.height,
            source.duration_seconds,
            clip.id if clip else None,
            clip.start_frame if clip else 0,
            clip.end_frame if clip else None,
            mode,
            interval,
            output_dir.expanduser().resolve() / request_id,
            _media_stamp(path),
        )


@dataclass(frozen=True)
class ExtractedFrame:
    id: str
    frame_number: int
    path: Path
    thumbnail_path: Path | None
    width: int
    height: int
    media_stamp: tuple[int, ...] | None
    thumbnail_stamp: tuple[int, ...] | None

    def to_model(self, task: FrameExtractionTask) -> "Frame":
        from models.frame import Frame

        return Frame(
            id=self.id,
            file_path=self.path,
            source_id=task.source_id,
            clip_id=task.clip_id,
            frame_number=self.frame_number,
            thumbnail_path=self.thumbnail_path,
            width=self.width,
            height=self.height,
        )


@dataclass(frozen=True)
class FrameExtractionOutcome:
    request_id: str
    status: OutcomeStatus
    frames: tuple[ExtractedFrame, ...] = ()
    message: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["frames"] = [
            {
                **asdict(frame),
                "path": str(frame.path),
                "thumbnail_path": str(frame.thumbnail_path)
                if frame.thumbnail_path
                else None,
            }
            for frame in self.frames
        ]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "FrameExtractionOutcome":
        if (
            not isinstance(data.get("request_id"), str)
            or not data["request_id"]
            or data.get("status") not in ("succeeded", "failed", "unprocessed")
            or not isinstance(data.get("frames"), list)
            or (
                data.get("message") is not None and not isinstance(data["message"], str)
            )
        ):
            raise ValueError("Invalid extraction outcome")
        frames = []
        for value in data["frames"]:
            fields = dict(value)
            if (
                not isinstance(fields.get("id"), str)
                or not fields["id"]
                or type(fields.get("frame_number")) is not int
                or fields["frame_number"] < 0
                or any(
                    type(fields.get(name)) is not int or fields[name] <= 0
                    for name in ("width", "height")
                )
            ):
                raise ValueError("Invalid extracted frame metadata")
            for name in ("path", "thumbnail_path"):
                if name == "thumbnail_path" and fields.get(name) is None:
                    continue
                if (
                    not isinstance(fields.get(name), str)
                    or not Path(fields[name]).is_absolute()
                ):
                    raise ValueError("Invalid extracted frame path")
                fields[name] = Path(fields[name])
            fields["media_stamp"] = _decode_stamp(fields.get("media_stamp"))
            fields["thumbnail_stamp"] = _decode_stamp(fields.get("thumbnail_stamp"))
            frames.append(ExtractedFrame(**fields))
        if data["status"] != "succeeded" and frames:
            raise ValueError("Incomplete extraction contains published frames")
        return cls(
            data["request_id"], data["status"], tuple(frames), data.get("message")
        )


def _decode_stamp(value) -> tuple[int, ...] | None:
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 5
        or any(type(v) is not int for v in value)
    ):
        raise ValueError("Invalid extraction media stamp")
    return tuple(value)


def run_frame_extraction(
    task: FrameExtractionTask,
    *,
    cancel_event: Event | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> FrameExtractionOutcome:
    """Extract into a new request directory, never replacing published images."""
    cancel = cancel_event or Event()
    created = keep = False
    try:
        if cancel.is_set():
            return FrameExtractionOutcome(task.request_id, "unprocessed")
        if task.media_stamp is None or _media_stamp(task.path) != task.media_stamp:
            raise ValueError("Extraction source is missing or changed")
        task.artifact_dir.mkdir(parents=True, exist_ok=False)
        created = True
        from core.ffmpeg import extract_frames_batch
        from core.thumbnail import generate_image_thumbnail
        from PIL import Image

        def report(current: int, total: int) -> None:
            if progress and not cancel.is_set():
                progress(current, total)

        paths = extract_frames_batch(
            task.path,
            task.artifact_dir / "frames",
            task.fps,
            mode=task.mode,
            interval=task.interval,
            start_frame=task.start_frame,
            end_frame=task.end_frame,
            progress_callback=report,
            cancel_event=cancel,
        )
        frames = []
        for index, path in enumerate(paths):
            if cancel.is_set():
                return FrameExtractionOutcome(task.request_id, "unprocessed")
            number = int(path.stem.removeprefix("frame_"))
            thumbnail_path = task.artifact_dir / "thumbnails" / f"{path.stem}.jpg"
            thumbnail: Path | None = thumbnail_path
            with Image.open(path) as image:
                width, height = image.size
            try:
                generate_image_thumbnail(path, thumbnail_path)
            except Exception as exc:
                logger.warning("Frame thumbnail generation failed: %s", exc)
                thumbnail = None
            frames.append(
                ExtractedFrame(
                    uuid4().hex,
                    number,
                    path,
                    thumbnail,
                    width,
                    height,
                    _media_stamp(path),
                    _media_stamp(thumbnail),
                )
            )
            report(index + 1, len(paths))
        if cancel.is_set():
            return FrameExtractionOutcome(task.request_id, "unprocessed")
        if _media_stamp(task.path) != task.media_stamp:
            raise ValueError("Extraction source changed during processing")
        keep = True
        return FrameExtractionOutcome(task.request_id, "succeeded", tuple(frames))
    except Exception as exc:
        return FrameExtractionOutcome(
            task.request_id,
            "unprocessed" if cancel.is_set() else "failed",
            message=str(exc),
        )
    finally:
        if created and not keep:
            try:
                shutil.rmtree(task.artifact_dir)
            except OSError as exc:
                logger.warning("Could not remove incomplete extraction: %s", exc)


class FrameExtractionApplication:
    """Apply a complete frame batch once to the originating unchanged source."""

    def __init__(self, project: "Project", task: FrameExtractionTask) -> None:
        project.session.assert_owner()
        self.project = project
        self.session_id = project.session.session_id
        self.path = project.path.resolve() if project.path is not None else None
        self.task = task
        self.source = project.sources_by_id.get(task.source_id)
        self.clip = project.clips_by_id.get(task.clip_id) if task.clip_id else None
        self.consumed = False

    def is_current(self, project: "Project") -> bool:
        return (
            project is self.project
            and project.session.session_id == self.session_id
            and (project.path.resolve() if project.path is not None else None)
            == self.path
        )

    def apply(
        self,
        project: "Project",
        outcome: FrameExtractionOutcome,
        *,
        recovered_task: FrameExtractionTask | None = None,
    ) -> bool:
        task = recovered_task or self.task
        if recovered_task is not None and (
            task.artifact_dir.parent != self.task.artifact_dir.parent
            or replace(
                self.task, request_id=task.request_id, artifact_dir=task.artifact_dir
            )
            != task
        ):
            return False
        if (
            not self.is_current(project)
            or self.consumed
            or outcome.request_id != task.request_id
            or outcome.status != "succeeded"
        ):
            return False

        def publish() -> bool:
            self.consumed = True
            source = self.source
            if (
                source is None
                or source.id != task.source_id
                or project.sources_by_id.get(task.source_id) is not source
                or (
                    source.file_path,
                    source.fps,
                    source.width,
                    source.height,
                    source.duration_seconds,
                )
                != (task.path, task.fps, task.width, task.height, task.duration)
                or task.media_stamp is None
                or _media_stamp(task.path) != task.media_stamp
            ):
                return False
            if task.clip_id is not None and (
                self.clip is None
                or self.clip.id != task.clip_id
                or project.clips_by_id.get(task.clip_id) is not self.clip
                or (self.clip.source_id, self.clip.start_frame, self.clip.end_frame)
                != (task.source_id, task.start_frame, task.end_frame)
            ):
                return False
            validate_frame_artifacts(task, outcome)
            if any(frame.id in project.frames_by_id for frame in outcome.frames):
                raise ValueError("Extracted frame ID already exists")
            if outcome.frames:
                project.add_frames([frame.to_model(task) for frame in outcome.frames])
            return True

        return project.session.apply_external(publish)


def validate_frame_artifacts(
    task: FrameExtractionTask, outcome: FrameExtractionOutcome
) -> None:
    """Reject incomplete, changed, or out-of-request artifact batches."""
    if outcome.request_id != task.request_id or outcome.status != "succeeded":
        raise ValueError("Extraction outcome does not match its task")
    numbers = [frame.frame_number for frame in outcome.frames]
    ids = [frame.id for frame in outcome.frames]
    if numbers != sorted(set(numbers)) or len(ids) != len(set(ids)):
        raise ValueError("Duplicate or unordered extracted frames")
    for frame in outcome.frames:
        if (
            frame.frame_number < task.start_frame
            or (task.end_frame is not None and frame.frame_number >= task.end_frame)
            or (
                task.mode == "interval"
                and (frame.frame_number - task.start_frame) % task.interval
            )
            or frame.path.resolve().parent != task.artifact_dir / "frames"
            or frame.path.stem != f"frame_{frame.frame_number:06d}"
            or frame.media_stamp is None
            or _media_stamp(frame.path) != frame.media_stamp
            or (
                frame.thumbnail_path is not None
                and (
                    frame.thumbnail_path.resolve().parent
                    != task.artifact_dir / "thumbnails"
                    or frame.thumbnail_stamp is None
                    or _media_stamp(frame.thumbnail_path) != frame.thumbnail_stamp
                )
            )
        ):
            raise ValueError(
                "Extracted frame artifacts changed or do not match the request"
            )

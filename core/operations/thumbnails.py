"""Detached thumbnail generation and owner-thread publication."""

from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from pathlib import Path
from threading import Event
from typing import Callable, TYPE_CHECKING
from uuid import uuid4

from core.jobs.media import media_stamp
from core.operations.contracts import OutcomeStatus

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip, Source


@dataclass(frozen=True)
class ThumbnailTask:
    clip_id: str
    source_id: str
    source_path: Path | None
    start_frame: int
    end_frame: int
    fps: float
    previous_path: Path | None
    source_stamp: tuple | None = field(init=False)
    previous_stamp: tuple | None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_stamp",
            media_stamp(self.source_path) if self.source_path else None,
        )
        object.__setattr__(
            self,
            "previous_stamp",
            media_stamp(self.previous_path) if self.previous_path else None,
        )

    @classmethod
    def capture(cls, clip: "Clip", source: "Source | None") -> "ThumbnailTask":
        return cls(
            clip.id,
            clip.source_id,
            source.file_path if source else None,
            clip.start_frame,
            clip.end_frame,
            source.fps if source else 0,
            Path(clip.thumbnail_path) if clip.thumbnail_path else None,
        )


@dataclass(frozen=True)
class ThumbnailOptions:
    cache_dir: Path
    width: int = 320
    height: int = 180
    force: bool = False

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Thumbnail dimensions must be positive")


@dataclass(frozen=True)
class ThumbnailOutcome:
    clip_id: str
    status: OutcomeStatus
    path: str | None = None
    code: str | None = None
    message: str | None = None


def run_thumbnails(
    tasks: tuple[ThumbnailTask, ...],
    options: ThumbnailOptions,
    cancel: Event,
    progress: Callable[[int, int, ThumbnailOutcome], None] | None = None,
) -> tuple[ThumbnailOutcome, ...]:
    """Compute from immutable inputs; never receive or mutate live models."""
    generator = None
    unavailable = None
    outcomes = []
    for index, task in enumerate(tasks):

        def outcome(status: OutcomeStatus, **kwargs) -> ThumbnailOutcome:
            return ThumbnailOutcome(task.clip_id, status, **kwargs)

        if cancel.is_set():
            result = outcome("unprocessed", code="cancelled")
        elif not options.force and task.previous_path and task.previous_path.is_file():
            result = outcome(
                "skipped", path=str(task.previous_path), code="already_exists"
            )
        elif (
            task.source_path is None
            or task.source_stamp is None
            or not task.source_path.is_file()
        ):
            result = outcome("failed", code="source_file_missing")
        elif media_stamp(task.source_path) != task.source_stamp:
            result = outcome("failed", code="stale_input")
        elif (
            not isinstance(task.fps, (int, float))
            or isinstance(task.fps, bool)
            or not isfinite(task.fps)
            or task.fps <= 0
            or not isinstance(task.start_frame, int)
            or not isinstance(task.end_frame, int)
            or task.end_frame <= task.start_frame
        ):
            result = outcome("failed", code="invalid_time_range")
        else:
            if generator is None and unavailable is None:
                try:
                    from core.thumbnail import ThumbnailGenerator

                    generator = ThumbnailGenerator(cache_dir=options.cache_dir)
                except Exception as exc:
                    unavailable = str(exc)
            if generator is None:
                result = outcome(
                    "failed",
                    code="thumbnail_generation_unavailable",
                    message=unavailable,
                )
            else:
                temporary = None
                try:
                    identity = (
                        str(task.source_path.resolve()),
                        task.source_stamp,
                        task.start_frame,
                        task.end_frame,
                        task.fps,
                        options.width,
                        options.height,
                    )
                    digest = sha256(repr(identity).encode()).hexdigest()
                    suffix = f"-{uuid4().hex}" if options.force else ""
                    output = generator.cache_dir / f"clip_{digest}{suffix}.jpg"
                    if not output.is_file() or output.stat().st_size == 0:
                        temporary = (
                            generator.cache_dir / f".thumbnail-{uuid4().hex}.jpg"
                        )
                        generated = Path(
                            generator.generate_clip_thumbnail(
                                video_path=task.source_path,
                                start_seconds=task.start_frame / task.fps,
                                end_seconds=task.end_frame / task.fps,
                                output_path=temporary,
                                width=options.width,
                                height=options.height,
                            )
                        )
                        if (
                            generated != temporary
                            or not generated.is_file()
                            or generated.stat().st_size == 0
                        ):
                            raise ValueError(
                                "Thumbnail generator did not produce the requested file"
                            )
                        if (
                            not cancel.is_set()
                            and media_stamp(task.source_path) == task.source_stamp
                        ):
                            generated.replace(output)
                    if cancel.is_set():
                        result = outcome("unprocessed", code="cancelled")
                    elif media_stamp(task.source_path) != task.source_stamp:
                        result = outcome("failed", code="stale_input")
                    else:
                        result = outcome("succeeded", path=str(output))
                except Exception as exc:
                    result = outcome(
                        "failed", code="thumbnail_generation_failed", message=str(exc)
                    )
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        outcomes.append(result)
        if progress is not None:
            progress(index + 1, len(tasks), result)
    return tuple(outcomes)


class ThumbnailApplication:
    """Apply each result once while preserving target, media and editorial identity."""

    def __init__(
        self,
        project: "Project",
        tasks: tuple[ThumbnailTask, ...],
        *,
        clips: list["Clip"] | None = None,
    ) -> None:
        project.session.assert_owner()
        self.project = project
        self.session_id = project.session.session_id
        self.path = project.path
        self.tasks = {task.clip_id: task for task in tasks}
        selected = (
            {clip.id: clip for clip in clips}
            if clips is not None
            else project.clips_by_id
        )
        self.bindings = {
            task.clip_id: (
                selected.get(task.clip_id),
                project.sources_by_id.get(task.source_id),
            )
            for task in tasks
        }
        self.consumed: set[str] = set()

    def apply(self, outcome: ThumbnailOutcome) -> bool:
        project = self.project
        task = self.tasks.get(outcome.clip_id)
        if (
            task is None
            or outcome.clip_id in self.consumed
            or outcome.status != "succeeded"
            or outcome.path is None
            or project.session.session_id != self.session_id
            or project.path != self.path
        ):
            return False

        output_path = Path(outcome.path)

        def publish() -> bool:
            self.consumed.add(outcome.clip_id)
            clip = project.clips_by_id.get(outcome.clip_id)
            source = project.sources_by_id.get(task.source_id)
            expected_clip, expected_source = self.bindings[outcome.clip_id]
            if (
                clip is None
                or clip is not expected_clip
                or source is not expected_source
                or ThumbnailTask.capture(clip, source) != task
                or not output_path.is_file()
            ):
                return False
            clip.thumbnail_path = output_path
            project.update_clips([clip])
            return True

        return project.session.apply_external(publish)


def thumbnail_payload(outcomes: tuple[ThumbnailOutcome, ...]) -> dict:
    """Preserve the existing headless thumbnail response shape."""
    return {
        "succeeded": [
            {"clip_id": o.clip_id, "path": o.path}
            for o in outcomes
            if o.status == "succeeded"
        ],
        "failed": [
            dict(
                clip_id=o.clip_id,
                code=o.code,
                **({"message": o.message} if o.message else {}),
            )
            for o in outcomes
            if o.status in ("failed", "unprocessed")
        ],
        "skipped": [
            {"clip_id": o.clip_id, "reason": o.code}
            for o in outcomes
            if o.status == "skipped"
        ],
        "total_clips": len(outcomes),
    }

"""Shared detection computation; callers own project publication and thumbnails."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Callable

from core.spine.sources import find_source_by_path, same_source_path

if TYPE_CHECKING:
    from core.project import Project
    from core.scene_detect import DetectionConfig, KaraokeDetectionConfig
    from models.clip import Clip, Source


class DetectionCancelled(RuntimeError):
    """Detection was cancelled before its result could be delivered."""


class StaleDetectionResult(RuntimeError):
    """Detection inputs changed before the result could be accepted."""


def _media_stamp(path: Path) -> tuple[int, int, int, int, int] | None:
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


def _target_digest(project: Project, path: Path) -> str:
    sources = [s for s in project.sources if same_source_path(s.file_path, path)]
    ids = {s.id for s in sources}
    payload = {
        "sources": [s.to_dict() for s in sources],
        "clips": [c.to_dict() for c in project.clips if c.source_id in ids],
    }
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class DetectionGuard:
    """Owner-thread snapshot of the target that detection may replace."""

    session_id: str
    video_path: Path
    source_id: str | None
    target_digest: str
    media_stamp: tuple[int, int, int, int, int] | None

    @classmethod
    def capture(
        cls, project: Project, video_path: Path, *, source_id: str | None = None
    ) -> DetectionGuard:
        project.session.assert_owner()
        path = Path(video_path).resolve()
        source = (
            project.sources_by_id.get(source_id)
            if source_id
            else find_source_by_path(project, path)
        )
        if source_id and (
            source is None or not same_source_path(source.file_path, path)
        ):
            raise StaleDetectionResult("Detection source no longer matches its media")
        return cls(
            project.session.session_id,
            path,
            source.id if source else None,
            _target_digest(project, path),
            _media_stamp(path),
        )

    def validate(self, project: Project) -> None:
        project.session.assert_owner()
        if (
            project.session.session_id != self.session_id
            or _media_stamp(self.video_path) != self.media_stamp
            or _target_digest(project, self.video_path) != self.target_digest
        ):
            raise StaleDetectionResult("Detection target changed; run detection again")


class DetectionApplication:
    """Publish one detached result to the original owner and target objects."""

    def __init__(self, project: Project, guard: DetectionGuard) -> None:
        project.session.assert_owner()
        self.project = project
        self.guard = guard
        self.path = project.path
        self.source = (
            project.sources_by_id.get(guard.source_id) if guard.source_id else None
        )
        self.clips = (
            tuple(project.clips_by_source.get(guard.source_id, ()))
            if guard.source_id
            else ()
        )
        self.consumed = False

    def apply(
        self,
        source: Source,
        clips: list[Clip],
        *,
        still_current: Callable[[], bool],
    ) -> Source | None:
        """Retain imported source identity and reject replaced targets or runs."""
        project = self.project
        self.guard.validate(project)

        def publish() -> Source | None:
            if self.consumed or not still_current():
                return None
            source_id = self.guard.source_id
            current = project.sources_by_id.get(source_id) if source_id else None
            current_clips = (
                project.clips_by_source.get(source_id, ()) if source_id else ()
            )
            if (
                project.path != self.path
                or current is not self.source
                or len(current_clips) != len(self.clips)
                or any(a is not b for a, b in zip(current_clips, self.clips))
                or not same_source_path(source.file_path, self.guard.video_path)
            ):
                raise StaleDetectionResult(
                    "Detection target changed; run detection again"
                )
            self.consumed = True
            if current is None:
                if source.id in project.sources_by_id:
                    raise StaleDetectionResult(
                        "Detection returned an existing source ID"
                    )
                source.analyzed = True
                project.add_source(source)
                current = source
            else:
                current.duration_seconds = source.duration_seconds
                current.fps = source.fps
                current.width = source.width
                current.height = source.height
                current.analyzed = True
            # A source-added observer can cancel or replace the workflow.
            if not still_current():
                return None
            for clip in clips:
                clip.source_id = current.id
            project.replace_source_clips(current.id, clips)
            if not still_current():
                return None
            if project.sources_by_id.get(current.id) is not current or any(
                project.clips_by_id.get(clip.id) is not clip for clip in clips
            ):
                raise StaleDetectionResult(
                    "Detection result was replaced during publication"
                )
            return current

        return project.session.apply_external(publish)


@dataclass(frozen=True)
class DetectionRequest:
    """Detached configuration snapshot, safe to pass to a worker."""

    video_path: Path
    config_json: str = "{}"
    mode: str = "adaptive"
    karaoke_config_json: str = "{}"
    media_stamp: tuple[int, int, int, int, int] | None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_path", Path(self.video_path).resolve())
        object.__setattr__(self, "media_stamp", _media_stamp(self.video_path))

    @classmethod
    def build(
        cls,
        video_path: Path,
        config: DetectionConfig | None = None,
        *,
        mode: str = "adaptive",
        karaoke_config: KaraokeDetectionConfig | None = None,
    ) -> DetectionRequest:
        return cls(
            Path(video_path).resolve(),
            json.dumps(asdict(config) if config is not None else {}, allow_nan=False),
            mode,
            json.dumps(
                asdict(karaoke_config) if karaoke_config is not None else {},
                allow_nan=False,
            ),
        )

    def validate_media(self) -> None:
        if _media_stamp(self.video_path) != self.media_stamp:
            raise StaleDetectionResult("Detection media changed; run detection again")


def run_detection(
    request: DetectionRequest,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_event: Event | None = None,
) -> tuple[Source, list[Clip]]:
    """Compute fresh source/clips without accessing a live project.

    Cancellation is cooperative at the native-call boundary. Callers must also
    check it before subsequent stages or publishing into their owning session.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise DetectionCancelled()

    request.validate_media()

    from core.scene_detect import DetectionConfig, KaraokeDetectionConfig, SceneDetector

    detector = SceneDetector(config=DetectionConfig(**json.loads(request.config_json)))
    progress = progress_callback or (lambda _fraction, _message: None)
    if request.mode == "karaoke":
        result = detector.detect_karaoke_scenes_with_progress(
            request.video_path,
            progress,
            KaraokeDetectionConfig(**json.loads(request.karaoke_config_json)),
        )
    else:
        result = detector.detect_scenes_with_progress(request.video_path, progress)
    if cancel_event is not None and cancel_event.is_set():
        raise DetectionCancelled()
    request.validate_media()
    return result

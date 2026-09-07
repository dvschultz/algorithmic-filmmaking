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

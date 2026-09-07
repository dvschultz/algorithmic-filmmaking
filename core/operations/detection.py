"""Shared detection computation; callers own project publication and thumbnails."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from core.scene_detect import DetectionConfig, KaraokeDetectionConfig
    from models.clip import Clip, Source


class DetectionCancelled(RuntimeError):
    """Detection was cancelled before its result could be delivered."""


@dataclass(frozen=True)
class DetectionRequest:
    """Detached configuration snapshot, safe to pass to a worker."""

    video_path: Path
    config_json: str = "{}"
    mode: str = "adaptive"
    karaoke_config_json: str = "{}"

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
            Path(video_path),
            json.dumps(asdict(config) if config is not None else {}, allow_nan=False),
            mode,
            json.dumps(
                asdict(karaoke_config) if karaoke_config is not None else {},
                allow_nan=False,
            ),
        )


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
    return result

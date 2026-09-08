"""Cached continuous preview rendering for timeline sequences."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
import tempfile
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

from core.sequence_export import ExportConfig, SequenceExporter
from core.settings import load_settings
from models.sequence import Sequence
from models.clip import Source
from core.artifacts import ArtifactLease
from core.media_cache import MediaCache

if TYPE_CHECKING:
    from models.frame import Frame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SequencePreviewSettings:
    """Settings that affect cached sequence preview output."""

    width: int = 1280
    height: int = 720
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    video_bitrate: str = "4M"
    audio_bitrate: str = "160k"
    preset: str = "fast"
    crf: int = 20
    profile_label: str = "720p proxy"


@dataclass(frozen=True)
class SequencePreviewRender:
    """Result of rendering or locating a cached sequence preview."""

    path: Path
    signature: str
    from_cache: bool
    profile_label: str
    lease: ArtifactLease | None = field(default=None, compare=False, repr=False)
    file_stamp: tuple[int, ...] | None = field(default=None, compare=False, repr=False)


def get_sequence_preview_cache_dir(cache_root: Optional[Path] = None) -> Path:
    """Return the directory where sequence preview renders are cached."""
    if cache_root is None:
        settings = load_settings()
        cache_root = settings.thumbnail_cache_dir.parent
    return cache_root / "sequence_previews"


def _preview_cache(cache_root: Path | None) -> MediaCache:
    directory = get_sequence_preview_cache_dir(cache_root)
    if cache_root is None:
        from core.paths import get_artifact_store_dir

        artifact_root = get_artifact_store_dir()
    else:
        artifact_root = cache_root / "artifacts"
    return MediaCache(directory, artifact_root)


def get_sequence_preview_path(
    sequence: Sequence,
    signature: str,
    cache_root: Optional[Path] = None,
) -> Path:
    """Return the preview path for a sequence/signature pair."""
    sequence_id = getattr(sequence, "id", None) or "default"
    safe_sequence_id = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in str(sequence_id)
    )
    return get_sequence_preview_cache_dir(cache_root) / safe_sequence_id / f"{signature}.mp4"


def compute_sequence_preview_signature(
    sequence: Sequence,
    sources: dict[str, Source],
    clips: dict[str, tuple],
    settings: SequencePreviewSettings | None = None,
    frames: Optional[dict[str, "Frame"]] = None,
) -> str:
    """Compute a stable signature for the rendered preview's meaningful inputs."""
    settings = settings or SequencePreviewSettings()
    payload = {
        "render_plan_version": 2,
        "preview_settings": {
            "width": settings.width,
            "height": settings.height,
            "video_codec": settings.video_codec,
            "audio_codec": settings.audio_codec,
            "video_bitrate": settings.video_bitrate,
            "audio_bitrate": settings.audio_bitrate,
            "preset": settings.preset,
            "crf": settings.crf,
            "profile_label": settings.profile_label,
        },
        "sequence": {
            "fps": sequence.fps,
            "algorithm": sequence.algorithm,
            "show_chromatic_color_bar": bool(
                getattr(sequence, "show_chromatic_color_bar", False)
            ),
            "music_path": _path_fingerprint(
                Path(sequence.music_path)
                if sequence.music_path
                else None
            ),
            "clips": [
                _sequence_clip_payload(seq_clip, sources, clips, frames)
                for seq_clip in sequence.get_all_clips()
            ],
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def render_sequence_preview(
    sequence: Sequence,
    sources: dict[str, Source],
    clips: dict[str, tuple],
    cache_root: Optional[Path] = None,
    settings: SequencePreviewSettings | None = None,
    progress_callback=None,
    frames: Optional[dict[str, "Frame"]] = None,
    cancel_check: Callable[[], bool] | None = None,
) -> SequencePreviewRender:
    """Render a cached continuous preview for a sequence, or return an existing one."""
    from core.render_plan import compile_render_plan
    if cancel_check is not None and cancel_check():
        raise RuntimeError("Sequence preview render cancelled")
    plan = compile_render_plan(sequence, sources, clips, frames=frames)
    settings = settings or SequencePreviewSettings()
    signature = compute_sequence_preview_signature(
        sequence=sequence,
        sources=sources,
        clips=clips,
        settings=settings,
        frames=frames,
    )
    # Hash content in the worker; the GUI signature stays a cheap change guard.
    content = hashlib.sha256(signature.encode())
    from core.binary_resolver import find_binary

    binary = find_binary("ffmpeg")
    content.update(json.dumps(_path_fingerprint(Path(binary) if binary else None), sort_keys=True).encode())
    for path, _stamp in plan.media_stamps:
        file_digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                if cancel_check is not None and cancel_check():
                    raise RuntimeError("Sequence preview render cancelled")
                file_digest.update(block)
        content.update(file_digest.digest())
    plan.validate_media_unchanged()
    if compute_sequence_preview_signature(sequence, sources, clips, settings, frames) != signature:
        raise RuntimeError("Sequence preview inputs changed during fingerprinting")
    cache = _preview_cache(cache_root)
    cached = cache.get(sequence.id, content.hexdigest())
    if cached is not None:
        plan.validate_media_unchanged()
        if compute_sequence_preview_signature(sequence, sources, clips, settings, frames) != signature:
            raise RuntimeError("Sequence preview inputs changed during cache verification")
        return SequencePreviewRender(
            path=cached.path,
            signature=signature,
            from_cache=True,
            profile_label=settings.profile_label,
            lease=cached.lease,
            file_stamp=cached.stamp,
        )

    staging = tempfile.TemporaryDirectory(prefix="render-", dir=cache.root)
    output_path = Path(staging.name) / "preview.mp4"

    config = ExportConfig(
        output_path=output_path,
        fps=sequence.fps,
        width=settings.width,
        height=settings.height,
        video_codec=settings.video_codec,
        audio_codec=settings.audio_codec,
        video_bitrate=settings.video_bitrate,
        audio_bitrate=settings.audio_bitrate,
        preset=settings.preset,
        crf=settings.crf,
        show_chromatic_color_bar=(
            bool(getattr(sequence, "show_chromatic_color_bar", False))
            and sequence.algorithm == "color"
        ),
        music_path=plan.music_path,
        cancel_check=cancel_check,
    )

    exporter = SequenceExporter()
    try:
        success = exporter.export(
            sequence=sequence,
            sources=sources,
            clips=clips,
            config=config,
            progress_callback=progress_callback,
            frames=frames,
        )
        if not success:
            raise RuntimeError("Sequence preview render failed")
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Sequence preview render cancelled")
        plan.validate_media_unchanged()
        if compute_sequence_preview_signature(sequence, sources, clips, settings, frames) != signature:
            raise RuntimeError("Sequence preview inputs changed during rendering")
        cached = cache.publish(sequence.id, content.hexdigest(), output_path)
    finally:
        staging.cleanup()

    return SequencePreviewRender(
        path=cached.path,
        signature=signature,
        from_cache=False,
        profile_label=settings.profile_label,
        lease=cached.lease,
        file_stamp=cached.stamp,
    )


def cleanup_sequence_preview_cache(cache_root: Optional[Path] = None, keep_latest: int = 5) -> None:
    """Retire only registered previews; active consumers retain their files."""
    cache_dir = get_sequence_preview_cache_dir(cache_root)
    if not cache_dir.exists():
        return

    _preview_cache(cache_root).prune(keep_latest)


def _sequence_clip_payload(seq_clip, sources, clips, frames):
    base = {
        "id": seq_clip.id,
        "source_clip_id": seq_clip.source_clip_id,
        "source_id": seq_clip.source_id,
        "track_index": seq_clip.track_index,
        "start_frame": seq_clip.start_frame,
        "in_point": seq_clip.in_point,
        "out_point": seq_clip.out_point,
        "frame_id": seq_clip.frame_id,
        "hold_frames": seq_clip.hold_frames,
        "hflip": seq_clip.hflip,
        "vflip": seq_clip.vflip,
        "reverse": seq_clip.reverse,
        "media_time": {
            "source_rate": seq_clip.source_rate,
            "timeline_rate": seq_clip.timeline_rate,
            "timeline_start": seq_clip.timeline_start,
            "source_presentation": seq_clip.source_presentation,
            "hold_duration": seq_clip.hold_duration,
            "legacy_timing": seq_clip.legacy_timing,
        },
        "prerendered_path": _path_fingerprint(
            Path(seq_clip.prerendered_path) if seq_clip.prerendered_path else None
        ),
    }
    if seq_clip.is_frame_entry:
        frame = frames.get(seq_clip.frame_id) if frames and seq_clip.frame_id else None
        base["frame_path"] = _path_fingerprint(getattr(frame, "file_path", None))
        base["frame_dominant_colors"] = getattr(frame, "dominant_colors", None)
        return base

    clip_data = clips.get(seq_clip.source_clip_id)
    if clip_data:
        source_clip, source = clip_data
        base["source"] = _source_payload(source)
        base["clip"] = {
            "start_frame": getattr(source_clip, "start_frame", None),
            "end_frame": getattr(source_clip, "end_frame", None),
            "dominant_colors": getattr(source_clip, "dominant_colors", None),
        }
    else:
        base["source"] = _source_payload(sources.get(seq_clip.source_id))
    return base


def _source_payload(source: Optional[Source]) -> dict | None:
    if source is None:
        return None
    return {
        "id": source.id,
        "path": _path_fingerprint(source.file_path),
        "fps": source.fps,
        "width": source.width,
        "height": source.height,
        "variable_frame_rate": source.variable_frame_rate,
        "frame_timestamps": source.frame_timestamps,
    }


def _path_fingerprint(path: Optional[Path]) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    try:
        stat = p.stat()
        return {
            "path": str(p.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "device": stat.st_dev,
            "inode": stat.st_ino,
        }
    except OSError:
        return {
            "path": str(p),
            "missing": True,
        }

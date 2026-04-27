"""Cached continuous preview rendering for timeline sequences."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from core.sequence_export import ExportConfig, SequenceExporter
from core.settings import load_settings
from models.sequence import Sequence
from models.clip import Source

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


def get_sequence_preview_cache_dir(cache_root: Optional[Path] = None) -> Path:
    """Return the directory where sequence preview renders are cached."""
    if cache_root is None:
        settings = load_settings()
        cache_root = settings.thumbnail_cache_dir.parent
    return cache_root / "sequence_previews"


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
                if getattr(sequence, "music_path", None)
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
) -> SequencePreviewRender:
    """Render a cached continuous preview for a sequence, or return an existing one."""
    settings = settings or SequencePreviewSettings()
    signature = compute_sequence_preview_signature(
        sequence=sequence,
        sources=sources,
        clips=clips,
        settings=settings,
        frames=frames,
    )
    output_path = get_sequence_preview_path(sequence, signature, cache_root)
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info("Using cached sequence preview: %s", output_path)
        return SequencePreviewRender(
            path=output_path,
            signature=signature,
            from_cache=True,
            profile_label=settings.profile_label,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    music_path = None
    raw_music = getattr(sequence, "music_path", None)
    if raw_music:
        candidate = Path(raw_music)
        if candidate.exists():
            music_path = candidate
        else:
            logger.warning("Sequence preview music file is missing: %s", raw_music)

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
        music_path=music_path,
    )

    exporter = SequenceExporter()
    success = exporter.export(
        sequence=sequence,
        sources=sources,
        clips=clips,
        config=config,
        progress_callback=progress_callback,
        frames=frames,
    )
    if not success:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Sequence preview render failed")

    return SequencePreviewRender(
        path=output_path,
        signature=signature,
        from_cache=False,
        profile_label=settings.profile_label,
    )


def cleanup_sequence_preview_cache(cache_root: Optional[Path] = None, keep_latest: int = 5) -> None:
    """Keep only the newest preview files per sequence cache directory."""
    cache_dir = get_sequence_preview_cache_dir(cache_root)
    if not cache_dir.exists():
        return

    for sequence_dir in (p for p in cache_dir.iterdir() if p.is_dir()):
        previews = sorted(
            sequence_dir.glob("*.mp4"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        for old_preview in previews[max(0, keep_latest):]:
            old_preview.unlink(missing_ok=True)


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
        }
    except OSError:
        return {
            "path": str(p),
            "missing": True,
        }

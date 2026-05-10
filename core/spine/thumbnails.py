"""Thumbnail generation spine.

Provides headless thumbnail backfill for clips. The implementation keeps
FFmpeg/settings imports inside functions so the spine remains cheap and
GUI-free at import time.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _check_cancel(cancel_event: Optional[threading.Event]) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def _resolve_clip_ids(project, clip_ids: Optional[list[str]]):
    if clip_ids is None:
        return list(project.clips)
    out = []
    for clip_id in clip_ids:
        clip = project.clips_by_id.get(clip_id)
        if clip is not None:
            out.append(clip)
    return out


def generate_clip_thumbnails(
    clip_source_pairs: list[tuple[object, object]],
    *,
    force: bool = False,
    width: int = 320,
    height: int = 180,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    progress_start: float = 0.0,
    progress_end: float = 1.0,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Generate thumbnails for explicit ``(clip, source)`` pairs.

    Returns a per-clip result payload and mutates each successful clip's
    ``thumbnail_path``. Existing thumbnails are skipped unless ``force`` is
    true. Missing source files and thumbnail failures are reported per clip.
    """
    if not clip_source_pairs:
        return {"succeeded": [], "failed": [], "skipped": [], "total_clips": 0}

    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []

    pending: list[tuple[object, object]] = []
    for clip, source in clip_source_pairs:
        thumbnail_path = getattr(clip, "thumbnail_path", None)
        if not force and thumbnail_path and Path(thumbnail_path).exists():
            skipped.append({"clip_id": clip.id, "reason": "already_exists"})
            continue
        pending.append((clip, source))

    if not pending:
        return {
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "total_clips": len(clip_source_pairs),
        }

    try:
        from core.settings import load_settings
        from core.thumbnail import ThumbnailGenerator

        settings = load_settings()
        generator = ThumbnailGenerator(cache_dir=settings.thumbnail_cache_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Thumbnail generation unavailable: %s", exc)
        failed.extend(
            {
                "clip_id": clip.id,
                "code": "thumbnail_generation_unavailable",
                "message": str(exc),
            }
            for clip, _source in pending
        )
        return {
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "total_clips": len(clip_source_pairs),
        }

    total = len(pending)
    progress_span = max(0.0, progress_end - progress_start)
    for i, (clip, source) in enumerate(pending):
        if _check_cancel(cancel_event):
            failed.extend({"clip_id": c.id, "code": "cancelled"} for c, _s in pending[i:])
            break

        if progress_callback is not None:
            progress = progress_start + (progress_span * (i / total))
            progress_callback(progress, f"Generating thumbnails ({i + 1}/{total})")

        if source is None or not source.file_path.exists():
            failed.append({"clip_id": clip.id, "code": "source_file_missing"})
            continue

        try:
            output_path = generator.cache_dir / f"clip_{clip.id}.jpg"
            thumbnail_path = generator.generate_clip_thumbnail(
                video_path=source.file_path,
                start_seconds=clip.start_time(source.fps),
                end_seconds=clip.end_time(source.fps),
                output_path=output_path,
                width=width,
                height=height,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to generate thumbnail for clip %s: %s", clip.id, exc)
            failed.append(
                {
                    "clip_id": clip.id,
                    "code": "thumbnail_generation_failed",
                    "message": str(exc),
                }
            )
            continue

        clip.thumbnail_path = Path(thumbnail_path)
        succeeded.append({"clip_id": clip.id, "path": str(clip.thumbnail_path)})

    return {
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "total_clips": len(clip_source_pairs),
    }


def generate_thumbnails(
    project,
    clip_ids: Optional[list[str]] = None,
    *,
    force: bool = False,
    width: int = 320,
    height: int = 180,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Generate or backfill thumbnails for project clips."""
    clips = _resolve_clip_ids(project, clip_ids)
    pairs = [(clip, project.sources_by_id.get(clip.source_id)) for clip in clips]

    result = generate_clip_thumbnails(
        pairs,
        force=force,
        width=width,
        height=height,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )

    updated = [
        project.clips_by_id[item["clip_id"]]
        for item in result["succeeded"]
        if item["clip_id"] in project.clips_by_id
    ]
    if updated:
        project.update_clips(updated)

    if progress_callback is not None:
        progress_callback(
            1.0,
            (
                f"Done: {len(result['succeeded'])} generated, "
                f"{len(result['failed'])} failed, {len(result['skipped'])} skipped"
            ),
        )

    return {"success": True, "result": result}

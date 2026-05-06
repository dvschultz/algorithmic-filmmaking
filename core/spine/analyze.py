"""Analysis op spine.

Per-clip analysis operations (colors, shots, transcription) routed through
the same canonical signature: sync ``def`` taking ``project`` plus op args,
optional ``progress_callback`` and ``cancel_event`` (R15).

Cancellation granularity is **per-clip**. The plan calls out that pushing
cancel_event inside the per-frame iteration loops in ``core/analysis/*.py``
is a follow-up; coarse-grained cancel between clips is sufficient for v1
and matches the existing pure-function APIs.

Skip-existing semantics: each op checks ``clip.<analysis_field> is not
None`` and skips already-populated clips. This makes re-issuing the same op
after a crashed/cancelled run a no-op for clips that succeeded (R18 —
preserve on-disk progress).
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _check_cancel(cancel_event: Optional[threading.Event]) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def _resolve_clip_ids(project, clip_ids: Optional[list[str]]):
    """Return the clip objects matching ``clip_ids``, or all project clips
    when ``clip_ids`` is None."""
    if clip_ids is None:
        return list(project.clips)
    out = []
    for clip_id in clip_ids:
        clip = project.clips_by_id.get(clip_id)
        if clip is not None:
            out.append(clip)
    return out


def analyze_colors(
    project,
    clip_ids: Optional[list[str]] = None,
    num_colors: int = 5,
    *,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Extract dominant colors for each clip in ``clip_ids`` (or all clips).

    Per-clip granularity: cancel checked between clips. Skips clips whose
    ``dominant_colors`` is already set unless ``skip_existing=False``.
    """
    from core.analysis.color import extract_dominant_colors

    clips = _resolve_clip_ids(project, clip_ids)
    if not clips:
        return {
            "success": True,
            "result": {"succeeded": [], "failed": [], "skipped": []},
        }

    sources_by_id = project.sources_by_id
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    updated = []

    total = len(clips)
    for i, clip in enumerate(clips):
        if _check_cancel(cancel_event):
            break

        if progress_callback is not None:
            progress_callback(
                i / total,
                f"Color analysis ({i + 1}/{total}): {clip.id}",
            )

        if skip_existing and clip.dominant_colors:
            skipped.append({"clip_id": clip.id, "reason": "already_populated"})
            continue

        source = sources_by_id.get(clip.source_id)
        if source is None or not source.file_path.exists():
            failed.append(
                {"clip_id": clip.id, "code": "source_file_missing"}
            )
            continue

        try:
            colors = extract_dominant_colors(
                video_path=source.file_path,
                start_frame=clip.start_frame,
                end_frame=clip.end_frame,
                n_colors=num_colors,
            )
        except Exception as exc:  # noqa: BLE001 — per-item resilience
            failed.append(
                {"clip_id": clip.id, "code": "extraction_failed", "message": str(exc)}
            )
            continue

        if colors:
            clip.dominant_colors = colors
            updated.append(clip)
            succeeded.append({"clip_id": clip.id, "color_count": len(colors)})
        else:
            failed.append({"clip_id": clip.id, "code": "no_colors_extracted"})

    if updated:
        project.update_clips(updated)

    if progress_callback is not None:
        progress_callback(
            1.0,
            f"Done: {len(succeeded)} ok, {len(failed)} failed, {len(skipped)} skipped",
        )

    return {
        "success": True,
        "result": {
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "total_clips": total,
        },
    }


def analyze_shots(
    project,
    clip_ids: Optional[list[str]] = None,
    *,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Classify shot type (wide/medium/close-up/extreme close-up) per clip.

    Uses ``core.analysis.shots.classify_shot_type``. Requires
    ``clip.thumbnail_path`` to point at an existing file; clips without
    thumbnails surface as ``thumbnail_missing`` failures (this op does
    not generate thumbnails — that's a separate concern).
    """
    from pathlib import Path

    from core.analysis.shots import classify_shot_type

    clips = _resolve_clip_ids(project, clip_ids)
    if not clips:
        return {
            "success": True,
            "result": {"succeeded": [], "failed": [], "skipped": []},
        }

    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    updated = []

    total = len(clips)
    for i, clip in enumerate(clips):
        if _check_cancel(cancel_event):
            break

        if progress_callback is not None:
            progress_callback(
                i / total,
                f"Shot classification ({i + 1}/{total}): {clip.id}",
            )

        if skip_existing and clip.shot_type:
            skipped.append({"clip_id": clip.id, "reason": "already_populated"})
            continue

        thumbnail_path = clip.thumbnail_path
        if not thumbnail_path:
            failed.append({"clip_id": clip.id, "code": "thumbnail_missing"})
            continue
        thumb_p = Path(thumbnail_path)
        if not thumb_p.exists():
            failed.append({"clip_id": clip.id, "code": "thumbnail_missing"})
            continue

        try:
            outcome = classify_shot_type(thumb_p)
        except Exception as exc:  # noqa: BLE001
            failed.append(
                {"clip_id": clip.id, "code": "classification_failed", "message": str(exc)}
            )
            continue

        # ``classify_shot_type`` returns ``(shot_type, confidence)`` per
        # the existing API in ``core/analysis/shots.py``. Tolerate either
        # shape so spine callers stubbing the function in tests don't
        # have to care.
        if isinstance(outcome, tuple):
            shot_type, _confidence = outcome
        else:
            shot_type = outcome

        if shot_type and shot_type != "unknown":
            clip.shot_type = shot_type
            updated.append(clip)
            succeeded.append({"clip_id": clip.id, "shot_type": shot_type})
        else:
            failed.append({"clip_id": clip.id, "code": "no_classification"})

    if updated:
        project.update_clips(updated)

    if progress_callback is not None:
        progress_callback(
            1.0,
            f"Done: {len(succeeded)} ok, {len(failed)} failed, {len(skipped)} skipped",
        )

    return {
        "success": True,
        "result": {
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "total_clips": total,
        },
    }


def transcribe(
    project,
    clip_ids: Optional[list[str]] = None,
    model: str = "base",
    language: Optional[str] = None,
    *,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Transcribe audio per clip via faster-whisper / lightning-whisper-mlx.

    Heavier dependency than colors / shots; the implementation lazy-imports
    ``core.transcription`` to keep the spine import boundary clean.
    """
    from core.transcription import transcribe_clip

    clips = _resolve_clip_ids(project, clip_ids)
    if not clips:
        return {
            "success": True,
            "result": {"succeeded": [], "failed": [], "skipped": []},
        }

    sources_by_id = project.sources_by_id
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    updated = []

    total = len(clips)
    for i, clip in enumerate(clips):
        if _check_cancel(cancel_event):
            break

        if progress_callback is not None:
            progress_callback(
                i / total,
                f"Transcribing ({i + 1}/{total}): {clip.id}",
            )

        if skip_existing and clip.transcript:
            skipped.append({"clip_id": clip.id, "reason": "already_populated"})
            continue

        source = sources_by_id.get(clip.source_id)
        if source is None or not source.file_path.exists():
            failed.append(
                {"clip_id": clip.id, "code": "source_file_missing"}
            )
            continue

        try:
            segments = transcribe_clip(
                video_path=source.file_path,
                start_frame=clip.start_frame,
                end_frame=clip.end_frame,
                fps=source.fps,
                model_size=model,
                language=language,
            )
        except Exception as exc:  # noqa: BLE001
            failed.append(
                {"clip_id": clip.id, "code": "transcription_failed", "message": str(exc)}
            )
            continue

        if segments:
            clip.transcript = segments
            updated.append(clip)
            succeeded.append(
                {"clip_id": clip.id, "segment_count": len(segments)}
            )
        else:
            # Empty transcript is a valid outcome (silent clip); track it.
            clip.transcript = []
            updated.append(clip)
            succeeded.append({"clip_id": clip.id, "segment_count": 0})

    if updated:
        project.update_clips(updated)

    if progress_callback is not None:
        progress_callback(
            1.0,
            f"Done: {len(succeeded)} ok, {len(failed)} failed, {len(skipped)} skipped",
        )

    return {
        "success": True,
        "result": {
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "total_clips": total,
        },
    }

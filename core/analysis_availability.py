"""Utilities for determining per-operation analysis availability."""

from __future__ import annotations

from collections.abc import Iterable


def operation_is_complete_for_clip(op_key: str, clip) -> bool:
    """Return True when a clip already has results for the given operation."""
    if op_key == "colors":
        return clip.dominant_colors is not None
    if op_key == "shots":
        return clip.shot_type is not None
    if op_key == "classify":
        return clip.object_labels is not None
    if op_key == "detect_objects":
        return clip.detected_objects is not None
    if op_key == "extract_text":
        # Matches current pipeline behavior: empty OCR list is treated as needing rerun.
        return bool(clip.extracted_texts)
    if op_key == "transcribe":
        return clip.transcript is not None
    if op_key == "describe":
        return clip.description is not None
    if op_key == "cinematography":
        return clip.cinematography is not None
    return False


def compute_operation_need_counts(clips: Iterable, op_keys: Iterable[str]) -> dict[str, int]:
    """Count how many clips still need each operation."""
    clip_list = list(clips)
    counts: dict[str, int] = {}
    for op_key in op_keys:
        counts[op_key] = sum(
            1 for clip in clip_list if not operation_is_complete_for_clip(op_key, clip)
        )
    return counts


def compute_disabled_operations(clips: Iterable, op_keys: Iterable[str]) -> set[str]:
    """Return operations that are already complete for all clips in scope."""
    counts = compute_operation_need_counts(clips, op_keys)
    return {op_key for op_key, needing in counts.items() if needing == 0}

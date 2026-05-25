"""Utilities for determining per-operation analysis availability."""

from __future__ import annotations

from collections.abc import Iterable


_ANALYSIS_RESULT_FIELDS: dict[str, tuple[str, ...]] = {
    "colors": ("dominant_colors",),
    "shots": ("shot_type",),
    "classify": ("object_labels",),
    "detect_objects": ("detected_objects", "person_count"),
    "extract_text": ("extracted_texts",),
    "transcribe": ("transcript",),
    "describe": ("description", "description_model", "description_frames"),
    "cinematography": ("cinematography",),
    "face_embeddings": ("face_embeddings",),
    "gaze": ("gaze_yaw", "gaze_pitch", "gaze_category"),
    "embeddings": (
        "embedding",
        "first_frame_embedding",
        "last_frame_embedding",
        "embedding_model",
    ),
}


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
    if op_key == "face_embeddings":
        return clip.face_embeddings is not None
    if op_key == "gaze":
        return clip.gaze_category is not None
    if op_key == "embeddings":
        return clip.embedding is not None
    if op_key == "custom_query":
        # Each custom query is unique — never auto-skip, always allow rerun
        return False
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


def clear_operation_result(clip, op_key: str) -> bool:
    """Clear stored result fields for one analysis operation on a clip.

    Returns True when at least one populated field was cleared. Unknown
    operations and custom queries are ignored because custom-query results are
    query-specific and should not be globally removed for a new query run.
    """
    changed = False
    for field in _ANALYSIS_RESULT_FIELDS.get(op_key, ()):
        if hasattr(clip, field) and getattr(clip, field) is not None:
            setattr(clip, field, None)
            changed = True
    return changed


def clear_operation_results(clips: Iterable, op_keys: Iterable[str]) -> int:
    """Clear stored result fields for selected operations across clips."""
    cleared = 0
    for clip in clips:
        for op_key in op_keys:
            if clear_operation_result(clip, op_key):
                cleared += 1
    return cleared

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
    "boundary_embeddings": ("first_frame_embedding", "last_frame_embedding"),
}


def operation_has_result(op_key: str, clip) -> bool:
    """Include legacy field projections when filtering or displaying results."""
    if op_key == "colors":
        return clip.dominant_colors is not None
    if op_key == "shots":
        return clip.shot_type is not None
    if op_key == "classify":
        return clip.object_labels is not None
    if op_key == "detect_objects":
        return clip.detected_objects is not None
    if op_key == "extract_text":
        return clip.extracted_texts is not None
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
    if op_key == "boundary_embeddings":
        return clip.first_frame_embedding is not None and clip.last_frame_embedding is not None
    if op_key == "custom_query":
        # Each custom query is unique — never auto-skip, always allow rerun
        return False
    return False


def operation_is_complete_for_clip(op_key: str, clip, *, runtime: dict | None = None) -> bool:
    """Report reusable completion; existing fields alone do not prove provenance."""
    if op_key == "colors":
        from core.analysis_records import current_record, model_runtime

        record = current_record(clip, op_key)
        if record is None or record.identity is None or not clip.dominant_colors:
            return False
        data = record.identity.to_dict()
        return bool(
            data["operation_version"] == 2 and data["schema_version"] == 1
            and data["model"] == (runtime if runtime is not None else model_runtime("kmeans-rgb", ("numpy", "scikit-learn", "opencv-python")))
            and data["parameters"] == {"num_colors": 5, "random_state": 42, "n_init": 1, "max_iter": 100}
            and record.value == {"dominant_colors": [list(color) for color in clip.dominant_colors]}
        )
    if op_key == "embeddings":
        from hashlib import sha256
        import json
        from core.analysis_records import current_record
        from core.analysis_model_identity import embedding_runtime

        record = current_record(clip, op_key)
        if record is None or record.identity is None or clip.embedding is None:
            return False
        data = record.identity.to_dict()
        if (
            data["operation_version"] != 2 or data["schema_version"] != 1
            or data["model"] != (runtime if runtime is not None else embedding_runtime())
            or data["parameters"] != {}
            or data["sampling"] != {"policy": "thumbnail/v1", "processor": "dinov2-default"}
        ):
            return False
        value = {"embedding": clip.embedding, "embedding_model": clip.embedding_model}
        if record.artifact is None:
            return bool(record.value == value)
        from core.artifacts import ArtifactStore

        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return sha256(encoded).hexdigest() == record.artifact.digest and ArtifactStore().available_fast(record.artifact)
    return operation_has_result(op_key, clip)


def compute_operation_need_counts(clips: Iterable, op_keys: Iterable[str]) -> dict[str, int]:
    """Count how many clips still need each operation."""
    clip_list = list(clips)
    counts: dict[str, int] = {}
    for op_key in op_keys:
        runtime = None
        if op_key == "colors":
            from core.analysis_records import model_runtime

            runtime = model_runtime("kmeans-rgb", ("numpy", "scikit-learn", "opencv-python"))
        elif op_key == "embeddings":
            from core.analysis_model_identity import embedding_runtime

            runtime = embedding_runtime()
        counts[op_key] = sum(
            1 for clip in clip_list if not operation_is_complete_for_clip(op_key, clip, runtime=runtime)
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
    records = getattr(clip, "analysis_records", {})
    if op_key in _ANALYSIS_RESULT_FIELDS and op_key in records:
        del records[op_key]
        changed = True
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

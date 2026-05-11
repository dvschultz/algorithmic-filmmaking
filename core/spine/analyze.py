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


def _thumbnail_for_clip(clip):
    """Return an existing thumbnail path for thumbnail-based analysis."""
    from pathlib import Path

    thumbnail_path = getattr(clip, "thumbnail_path", None)
    if not thumbnail_path:
        return None
    path = Path(thumbnail_path)
    return path if path.exists() else None


def classify_content(
    project,
    clip_ids: Optional[list[str]] = None,
    top_k: int = 5,
    threshold: float = 0.1,
    *,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Classify thumbnail content with ImageNet labels."""
    from core.analysis.classification import classify_frame

    clips = _resolve_clip_ids(project, clip_ids)
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    updated = []
    total = len(clips)

    for i, clip in enumerate(clips):
        if _check_cancel(cancel_event):
            break
        if progress_callback is not None and total:
            progress_callback(i / total, f"Content classification ({i + 1}/{total}): {clip.id}")
        if skip_existing and clip.object_labels is not None:
            skipped.append({"clip_id": clip.id, "reason": "already_populated"})
            continue
        thumbnail_path = _thumbnail_for_clip(clip)
        if thumbnail_path is None:
            failed.append({"clip_id": clip.id, "code": "thumbnail_missing"})
            continue
        try:
            labels = classify_frame(thumbnail_path, top_k=top_k, threshold=threshold)
        except Exception as exc:  # noqa: BLE001
            failed.append({"clip_id": clip.id, "code": "classification_failed", "message": str(exc)})
            continue
        clip.object_labels = [label for label, _confidence in labels]
        updated.append(clip)
        succeeded.append({"clip_id": clip.id, "label_count": len(labels)})

    if updated:
        project.update_clips(updated)
    if progress_callback is not None:
        progress_callback(1.0, f"Done: {len(succeeded)} ok, {len(failed)} failed, {len(skipped)} skipped")
    return {"success": True, "result": {"succeeded": succeeded, "failed": failed, "skipped": skipped, "total_clips": total}}


def detect_objects(
    project,
    clip_ids: Optional[list[str]] = None,
    confidence: float = 0.5,
    detect_all: bool = True,
    *,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Detect objects and person count on clip thumbnails."""
    from core.analysis.detection import count_people, detect_objects as detect_objects_in_image

    clips = _resolve_clip_ids(project, clip_ids)
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    updated = []
    total = len(clips)

    for i, clip in enumerate(clips):
        if _check_cancel(cancel_event):
            break
        if progress_callback is not None and total:
            progress_callback(i / total, f"Object detection ({i + 1}/{total}): {clip.id}")
        if skip_existing and clip.detected_objects is not None:
            skipped.append({"clip_id": clip.id, "reason": "already_populated"})
            continue
        thumbnail_path = _thumbnail_for_clip(clip)
        if thumbnail_path is None:
            failed.append({"clip_id": clip.id, "code": "thumbnail_missing"})
            continue
        try:
            if detect_all:
                detections = detect_objects_in_image(thumbnail_path, confidence_threshold=confidence)
                person_count = sum(1 for d in detections if d.get("label") == "person")
            else:
                detections = []
                person_count = count_people(thumbnail_path, confidence_threshold=confidence)
        except Exception as exc:  # noqa: BLE001
            failed.append({"clip_id": clip.id, "code": "detection_failed", "message": str(exc)})
            continue
        clip.detected_objects = detections
        clip.person_count = person_count
        updated.append(clip)
        succeeded.append({"clip_id": clip.id, "object_count": len(detections), "person_count": person_count})

    if updated:
        project.update_clips(updated)
    if progress_callback is not None:
        progress_callback(1.0, f"Done: {len(succeeded)} ok, {len(failed)} failed, {len(skipped)} skipped")
    return {"success": True, "result": {"succeeded": succeeded, "failed": failed, "skipped": skipped, "total_clips": total}}


def extract_text(
    project,
    clip_ids: Optional[list[str]] = None,
    num_keyframes: int = 3,
    use_vlm_fallback: bool = True,
    vlm_model: Optional[str] = None,
    vlm_only: bool = False,
    *,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Extract visible text from clips using OCR/VLM fallback."""
    from core.analysis.ocr import extract_text_from_clip

    clips = _resolve_clip_ids(project, clip_ids)
    sources_by_id = project.sources_by_id
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    updated = []
    total = len(clips)

    for i, clip in enumerate(clips):
        if _check_cancel(cancel_event):
            break
        if progress_callback is not None and total:
            progress_callback(i / total, f"Text extraction ({i + 1}/{total}): {clip.id}")
        if skip_existing and clip.extracted_texts is not None:
            skipped.append({"clip_id": clip.id, "reason": "already_populated"})
            continue
        source = sources_by_id.get(clip.source_id)
        if source is None or not source.file_path.exists():
            failed.append({"clip_id": clip.id, "code": "source_file_missing"})
            continue
        try:
            texts = extract_text_from_clip(
                clip=clip,
                source=source,
                num_keyframes=min(max(1, num_keyframes), 5),
                use_vlm_fallback=use_vlm_fallback,
                vlm_model=vlm_model,
                vlm_only=vlm_only,
            )
        except Exception as exc:  # noqa: BLE001
            failed.append({"clip_id": clip.id, "code": "text_extraction_failed", "message": str(exc)})
            continue
        clip.extracted_texts = texts
        updated.append(clip)
        succeeded.append({"clip_id": clip.id, "text_count": len(texts)})

    if updated:
        project.update_clips(updated)
    if progress_callback is not None:
        progress_callback(1.0, f"Done: {len(succeeded)} ok, {len(failed)} failed, {len(skipped)} skipped")
    return {"success": True, "result": {"succeeded": succeeded, "failed": failed, "skipped": skipped, "total_clips": total}}


def describe(
    project,
    clip_ids: Optional[list[str]] = None,
    tier: Optional[str] = None,
    prompt: Optional[str] = None,
    *,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Generate VLM descriptions for clip thumbnails/video snippets."""
    from core.analysis.description import describe_frame

    default_prompt = (
        "Describe this video frame in 3 sentences or less. "
        "Focus on the main subjects, action, and setting."
    )
    clips = _resolve_clip_ids(project, clip_ids)
    sources_by_id = project.sources_by_id
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    updated = []
    total = len(clips)

    for i, clip in enumerate(clips):
        if _check_cancel(cancel_event):
            break
        if progress_callback is not None and total:
            progress_callback(i / total, f"Description ({i + 1}/{total}): {clip.id}")
        if skip_existing and clip.description is not None:
            skipped.append({"clip_id": clip.id, "reason": "already_populated"})
            continue
        thumbnail_path = _thumbnail_for_clip(clip)
        if thumbnail_path is None:
            failed.append({"clip_id": clip.id, "code": "thumbnail_missing"})
            continue
        source = sources_by_id.get(clip.source_id)
        try:
            description, model = describe_frame(
                thumbnail_path,
                tier=tier,
                prompt=prompt or default_prompt,
                source_path=source.file_path if source else None,
                start_frame=clip.start_frame,
                end_frame=clip.end_frame,
                fps=source.fps if source else None,
            )
        except Exception as exc:  # noqa: BLE001
            failed.append({"clip_id": clip.id, "code": "description_failed", "message": str(exc)})
            continue
        clip.description = description
        clip.description_model = model
        clip.description_frames = 1
        updated.append(clip)
        succeeded.append({"clip_id": clip.id, "model": model})

    if updated:
        project.update_clips(updated)
    if progress_callback is not None:
        progress_callback(1.0, f"Done: {len(succeeded)} ok, {len(failed)} failed, {len(skipped)} skipped")
    return {"success": True, "result": {"succeeded": succeeded, "failed": failed, "skipped": skipped, "total_clips": total}}


def cinematography(
    project,
    clip_ids: Optional[list[str]] = None,
    mode: Optional[str] = None,
    model: Optional[str] = None,
    *,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Run rich cinematography analysis for clips."""
    from core.analysis.cinematography import analyze_cinematography

    clips = _resolve_clip_ids(project, clip_ids)
    sources_by_id = project.sources_by_id
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    updated = []
    total = len(clips)

    for i, clip in enumerate(clips):
        if _check_cancel(cancel_event):
            break
        if progress_callback is not None and total:
            progress_callback(i / total, f"Cinematography ({i + 1}/{total}): {clip.id}")
        if skip_existing and clip.cinematography is not None:
            skipped.append({"clip_id": clip.id, "reason": "already_populated"})
            continue
        thumbnail_path = _thumbnail_for_clip(clip)
        if thumbnail_path is None:
            failed.append({"clip_id": clip.id, "code": "thumbnail_missing"})
            continue
        source = sources_by_id.get(clip.source_id)
        try:
            analysis = analyze_cinematography(
                thumbnail_path=thumbnail_path,
                source_path=source.file_path if source and source.file_path.exists() else None,
                start_frame=clip.start_frame,
                end_frame=clip.end_frame,
                fps=source.fps if source else None,
                mode=mode,
                model=model,
            )
        except Exception as exc:  # noqa: BLE001
            failed.append({"clip_id": clip.id, "code": "cinematography_failed", "message": str(exc)})
            continue
        clip.cinematography = analysis
        if hasattr(analysis, "get_simple_shot_type"):
            clip.shot_type = analysis.get_simple_shot_type()
        updated.append(clip)
        succeeded.append({"clip_id": clip.id, "shot_size": getattr(analysis, "shot_size", None)})

    if updated:
        project.update_clips(updated)
    if progress_callback is not None:
        progress_callback(1.0, f"Done: {len(succeeded)} ok, {len(failed)} failed, {len(skipped)} skipped")
    return {"success": True, "result": {"succeeded": succeeded, "failed": failed, "skipped": skipped, "total_clips": total}}


def face_embeddings(
    project,
    clip_ids: Optional[list[str]] = None,
    sample_interval: float = 1.0,
    *,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Extract face embeddings from clip frame samples."""
    from core.analysis.faces import extract_faces_from_clip

    clips = _resolve_clip_ids(project, clip_ids)
    sources_by_id = project.sources_by_id
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    updated = []
    total = len(clips)

    for i, clip in enumerate(clips):
        if _check_cancel(cancel_event):
            break
        if progress_callback is not None and total:
            progress_callback(i / total, f"Face detection ({i + 1}/{total}): {clip.id}")
        if skip_existing and clip.face_embeddings is not None:
            skipped.append({"clip_id": clip.id, "reason": "already_populated"})
            continue
        source = sources_by_id.get(clip.source_id)
        if source is None or not source.file_path.exists():
            failed.append({"clip_id": clip.id, "code": "source_file_missing"})
            continue
        try:
            faces = extract_faces_from_clip(source.file_path, clip.start_frame, clip.end_frame, source.fps, sample_interval)
        except Exception as exc:  # noqa: BLE001
            failed.append({"clip_id": clip.id, "code": "face_detection_failed", "message": str(exc)})
            continue
        clip.face_embeddings = faces if faces else []
        updated.append(clip)
        succeeded.append({"clip_id": clip.id, "face_count": len(clip.face_embeddings)})

    if updated:
        project.update_clips(updated)
    if progress_callback is not None:
        progress_callback(1.0, f"Done: {len(succeeded)} ok, {len(failed)} failed, {len(skipped)} skipped")
    return {"success": True, "result": {"succeeded": succeeded, "failed": failed, "skipped": skipped, "total_clips": total}}


def gaze(
    project,
    clip_ids: Optional[list[str]] = None,
    sample_interval: float = 1.0,
    *,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Estimate gaze direction for clips."""
    from core.analysis.gaze import extract_gaze_from_clip

    clips = _resolve_clip_ids(project, clip_ids)
    sources_by_id = project.sources_by_id
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    updated = []
    total = len(clips)

    for i, clip in enumerate(clips):
        if _check_cancel(cancel_event):
            break
        if progress_callback is not None and total:
            progress_callback(i / total, f"Gaze analysis ({i + 1}/{total}): {clip.id}")
        if skip_existing and clip.gaze_category is not None:
            skipped.append({"clip_id": clip.id, "reason": "already_populated"})
            continue
        source = sources_by_id.get(clip.source_id)
        if source is None or not source.file_path.exists():
            failed.append({"clip_id": clip.id, "code": "source_file_missing"})
            continue
        try:
            result = extract_gaze_from_clip(str(source.file_path), clip.start_frame, clip.end_frame, source.fps, sample_interval)
        except Exception as exc:  # noqa: BLE001
            failed.append({"clip_id": clip.id, "code": "gaze_failed", "message": str(exc)})
            continue
        if result is None:
            failed.append({"clip_id": clip.id, "code": "no_gaze_detected"})
            continue
        clip.gaze_yaw = result["gaze_yaw"]
        clip.gaze_pitch = result["gaze_pitch"]
        clip.gaze_category = result["gaze_category"]
        updated.append(clip)
        succeeded.append({"clip_id": clip.id, "gaze_category": clip.gaze_category})

    if updated:
        project.update_clips(updated)
    if progress_callback is not None:
        progress_callback(1.0, f"Done: {len(succeeded)} ok, {len(failed)} failed, {len(skipped)} skipped")
    return {"success": True, "result": {"succeeded": succeeded, "failed": failed, "skipped": skipped, "total_clips": total}}


def embeddings(
    project,
    clip_ids: Optional[list[str]] = None,
    *,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Extract DINOv2 embeddings from clip thumbnails."""
    from core.analysis.embeddings import _EMBEDDING_MODEL_TAG, extract_clip_embeddings_batch

    clips = _resolve_clip_ids(project, clip_ids)
    to_process = []
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []

    for clip in clips:
        if skip_existing and clip.embedding is not None:
            skipped.append({"clip_id": clip.id, "reason": "already_populated"})
            continue
        thumbnail_path = _thumbnail_for_clip(clip)
        if thumbnail_path is None:
            failed.append({"clip_id": clip.id, "code": "thumbnail_missing"})
            continue
        to_process.append((clip, thumbnail_path))

    total = len(clips)
    if to_process and not _check_cancel(cancel_event):
        try:
            vectors = extract_clip_embeddings_batch([path for _clip, path in to_process])
        except Exception as exc:  # noqa: BLE001
            failed.extend({"clip_id": clip.id, "code": "embedding_failed", "message": str(exc)} for clip, _path in to_process)
        else:
            for i, ((clip, _path), vector) in enumerate(zip(to_process, vectors)):
                if _check_cancel(cancel_event):
                    break
                clip.embedding = vector
                clip.embedding_model = _EMBEDDING_MODEL_TAG
                succeeded.append({"clip_id": clip.id, "embedding_dim": len(vector)})
                if progress_callback is not None and to_process:
                    progress_callback((i + 1) / len(to_process), f"Embeddings ({i + 1}/{len(to_process)}): {clip.id}")
            if succeeded:
                project.update_clips([clip for clip, _path in to_process if clip.embedding is not None])

    if progress_callback is not None:
        progress_callback(1.0, f"Done: {len(succeeded)} ok, {len(failed)} failed, {len(skipped)} skipped")
    return {"success": True, "result": {"succeeded": succeeded, "failed": failed, "skipped": skipped, "total_clips": total}}


def custom_query(
    project,
    clip_ids: Optional[list[str]] = None,
    query: Optional[str] = None,
    tier: Optional[str] = None,
    *,
    skip_existing: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Evaluate a yes/no VLM visual query against clip thumbnails."""
    from core.analysis.custom_query import evaluate_custom_query

    if not query or not query.strip():
        return {"success": False, "error": {"code": "missing_query", "message": "query is required"}}

    clips = _resolve_clip_ids(project, clip_ids)
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    updated = []
    total = len(clips)

    for i, clip in enumerate(clips):
        if _check_cancel(cancel_event):
            break
        if progress_callback is not None and total:
            progress_callback(i / total, f"Custom query ({i + 1}/{total}): {clip.id}")
        if skip_existing and clip.custom_queries and any(q.get("query") == query for q in clip.custom_queries):
            skipped.append({"clip_id": clip.id, "reason": "already_populated"})
            continue
        thumbnail_path = _thumbnail_for_clip(clip)
        if thumbnail_path is None:
            failed.append({"clip_id": clip.id, "code": "thumbnail_missing"})
            continue
        try:
            match, confidence, model = evaluate_custom_query(thumbnail_path, query.strip(), tier=tier)
        except Exception as exc:  # noqa: BLE001
            failed.append({"clip_id": clip.id, "code": "custom_query_failed", "message": str(exc)})
            continue
        if clip.custom_queries is None:
            clip.custom_queries = []
        result = {
            "query": query.strip(),
            "match": match,
            "confidence": round(confidence, 4),
            "model": model,
        }
        clip.custom_queries.append(result)
        updated.append(clip)
        succeeded.append({"clip_id": clip.id, **result})

    if updated:
        project.update_clips(updated)
    if progress_callback is not None:
        progress_callback(1.0, f"Done: {len(succeeded)} ok, {len(failed)} failed, {len(skipped)} skipped")
    return {"success": True, "result": {"succeeded": succeeded, "failed": failed, "skipped": skipped, "total_clips": total}}


ANALYZE_CLIP_OPERATION_MAP = {
    "colors": analyze_colors,
    "shots": analyze_shots,
    "classify": classify_content,
    "detect_objects": detect_objects,
    "extract_text": extract_text,
    "transcribe": transcribe,
    "describe": describe,
    "cinematography": cinematography,
    "face_embeddings": face_embeddings,
    "gaze": gaze,
    "embeddings": embeddings,
    "custom_query": custom_query,
}


def analyze_clips(
    project,
    clip_ids: Optional[list[str]] = None,
    operations: Optional[list[str]] = None,
    *,
    query: Optional[str] = None,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Run one or more canonical UI analysis operations headlessly."""
    if not operations:
        return {"success": False, "error": {"code": "no_operations", "message": "operations is required"}}

    invalid = [op for op in operations if op not in ANALYZE_CLIP_OPERATION_MAP]
    if invalid:
        return {"success": False, "error": {"code": "invalid_operations", "operations": invalid}}

    results = {}
    total_ops = len(operations)
    for index, op in enumerate(operations):
        if _check_cancel(cancel_event):
            break
        if progress_callback is not None:
            progress_callback(index / total_ops, f"Starting {op} ({index + 1}/{total_ops})")
        kwargs = {"skip_existing": skip_existing}
        if op == "custom_query":
            kwargs["query"] = query
            # Custom query appends user-authored query runs by default.
            kwargs["skip_existing"] = False
        result = ANALYZE_CLIP_OPERATION_MAP[op](
            project,
            clip_ids,
            progress_callback=None,
            cancel_event=cancel_event,
            **kwargs,
        )
        results[op] = result
        if result.get("success") is False:
            return {"success": False, "error": result.get("error"), "result": results}

    if progress_callback is not None:
        progress_callback(1.0, f"Done: {len(results)} operation(s)")
    return {"success": True, "result": {"operations": results}}

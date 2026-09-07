"""Analysis op spine.

Per-clip analysis operations (colors, shots, transcription) routed through
the same canonical signature: sync ``def`` taking ``project`` plus op args,
optional ``progress_callback`` and ``cancel_event`` (R15).

Cancellation granularity is **per-clip**. The plan calls out that pushing
cancel_event inside the per-frame iteration loops in ``core/analysis/*.py``
is a follow-up; coarse-grained cancel between clips is sufficient for v1
and matches the existing pure-function APIs.

Skip-existing semantics: operations skip already-populated analysis fields.
Colors retain their historical truthy-palette check; an empty palette is retried.
This makes re-issuing the same op
after a crashed/cancelled run a no-op for clips that succeeded (R18 —
preserve on-disk progress).
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.project import Project

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
    from core.operations.colors import ColorApplication, color_request, compute_colors

    request = color_request(project, clip_ids, num_colors, skip_existing=skip_existing)
    application = ColorApplication(project, request)

    def progress(completed, total, outcome):
        if progress_callback is not None:
            progress_callback(completed / total, f"Color analysis ({completed}/{total}): {outcome.target_id}")

    result = application.apply(compute_colors(
        request, cancel_event=cancel_event, progress_callback=progress,
    ))
    succeeded = []
    failed = []
    skipped = []
    unprocessed = []
    for outcome in result.outcomes:
        if outcome.status == "succeeded":
            succeeded.append({"clip_id": outcome.target_id, "color_count": len(outcome.colors)})
        elif outcome.status == "skipped":
            skipped.append({"clip_id": outcome.target_id, "reason": outcome.code})
        elif outcome.status == "unprocessed":
            unprocessed.append({"clip_id": outcome.target_id, "reason": outcome.code})
        else:
            error = {"clip_id": outcome.target_id, "code": outcome.code}
            if outcome.message is not None:
                error["message"] = outcome.message
            failed.append(error)
    if progress_callback is not None:
        progress_callback(1.0, f"Done: {len(succeeded)} ok, {len(failed)} failed, {len(skipped)} skipped")
    return {
        "success": True,
        "result": {
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "unprocessed": unprocessed,
            "total_clips": len(request.targets),
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
    from core.operations.transcription import (
        TranscriptionApplication, TranscriptionOptions, run_transcription, snapshot_tasks,
    )

    clips = _resolve_clip_ids(project, clip_ids)
    tasks = snapshot_tasks(clips, project.sources_by_id, skip_existing=skip_existing)
    application = TranscriptionApplication(project, tasks)
    outcomes = run_transcription(
        tasks, TranscriptionOptions(model=model, language=language),
        cancel_event=cancel_event,
        progress=(lambda current, total: progress_callback(current / total, f"Transcribing ({current}/{total})")) if progress_callback else None,
    )
    accepted = application.apply_batch(project, outcomes)
    succeeded, failed, skipped, unprocessed = [], [], [], []
    for clip, outcome, applied in zip(clips, outcomes, accepted):
        if outcome.status == "succeeded":
            if not applied:
                failed.append({"clip_id": clip.id, "code": "stale_target", "message": "Transcription target changed during execution"})
                continue
            succeeded.append({"clip_id": clip.id, "segment_count": len(outcome.segments)})
        elif outcome.status == "skipped":
            skipped.append({"clip_id": clip.id, "reason": outcome.code})
        elif outcome.status == "failed":
            failed.append({"clip_id": clip.id, "code": outcome.code, "message": outcome.message})
        elif outcome.status == "unprocessed":
            unprocessed.append({"clip_id": clip.id, "code": outcome.code})
    return {"success": True, "result": {
        "succeeded": succeeded, "failed": failed, "skipped": skipped,
        "total_clips": len(clips), "unprocessed": unprocessed,
    }}


def align_words(
    project: Project,
    clip_ids: Optional[list[str]] = None,
    *,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Align existing transcripts without installing optional dependencies."""
    from core.operations.alignment import (
        AlignmentApplication, AlignmentOutcome, run_alignment, snapshot_alignment_tasks,
    )

    if clip_ids is not None:
        clip_ids = list(dict.fromkeys(clip_ids))
        unknown = [cid for cid in clip_ids if cid not in project.clips_by_id]
        if unknown:
            raise ValueError(f"Unknown alignment clip IDs: {', '.join(unknown)}")
    clips = _resolve_clip_ids(project, clip_ids)
    tasks = snapshot_alignment_tasks(clips, project.sources_by_id, skip_existing=skip_existing)
    application = AlignmentApplication(project, tasks)
    missing: list[str] = []
    if any(task.skip_reason is None for task in tasks) and not _check_cancel(cancel_event):
        from core.feature_registry import check_feature_ready
        ready, missing = check_feature_ready("word_alignment")
        if ready:
            missing = []
        elif not missing:
            missing = ["word_alignment"]
    if _check_cancel(cancel_event):
        missing = []
    if missing:
        outcomes = tuple(
            AlignmentOutcome(task.clip_id, "skipped", code=task.skip_reason)
            if task.skip_reason is not None else AlignmentOutcome(
                task.clip_id, "failed", code="dependency_missing",
                message=f"Word alignment dependencies unavailable: {', '.join(missing)}. Install them from Settings > Dependencies.",
            )
            for task in tasks
        )
    else:
        outcomes = run_alignment(
            tasks, cancel_event=cancel_event,
            progress=(lambda current, total: progress_callback(current / total if total else 1.0, f"Aligning words ({current}/{total})")) if progress_callback else None,
        )
    output: dict = {"succeeded": [], "failed": [], "skipped": [], "unprocessed": [], "total_clips": len(tasks)}
    for outcome in outcomes:
        if outcome.status == "succeeded":
            if application.apply(project, outcome):
                output["succeeded"].append({"clip_id": outcome.clip_id, "word_count": len(outcome.words)})
            else:
                output["failed"].append({"clip_id": outcome.clip_id, "code": "stale_target", "message": "Alignment target changed during execution"})
        elif outcome.status == "skipped":
            output["skipped"].append({"clip_id": outcome.clip_id, "reason": outcome.code})
        else:
            output[outcome.status].append({"clip_id": outcome.clip_id, "code": outcome.code, "message": outcome.message})
    return {"success": True, "result": output}


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
    from core.operations.description import (
        DEFAULT_PROMPT,
        DescriptionTask,
        DescriptionApplication,
        resolve_options,
        run_description,
    )

    clips = _resolve_clip_ids(project, clip_ids)
    sources_by_id = project.sources_by_id
    tasks = []
    for clip in clips:
        source = sources_by_id.get(clip.source_id)
        tasks.append(
            DescriptionTask(
                clip.id,
                _thumbnail_for_clip(clip),
                source.file_path if source else None,
                clip.start_frame,
                clip.end_frame,
                source.fps if source else None,
                skip_existing and clip.description is not None,
            )
        )
    total = len(clips)
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    def progress(current: int, count: int) -> None:
        if progress_callback is not None and count:
            progress_callback(current / count, f"Description ({current}/{count})")

    application = DescriptionApplication(project, tuple(tasks))
    outcomes = run_description(
        tuple(tasks),
        resolve_options(tier, prompt or DEFAULT_PROMPT),
        cancel_event=cancel_event,
        progress=progress,
    )
    accepted = application.apply_batch(project, outcomes)
    for clip, outcome, applied in zip(clips, outcomes, accepted):
        if outcome.status == "skipped":
            skipped.append({"clip_id": clip.id, "reason": outcome.code})
        elif outcome.status == "failed":
            failure = {"clip_id": clip.id, "code": outcome.code}
            if outcome.message is not None:
                failure["message"] = outcome.message
            failed.append(failure)
        elif outcome.status == "succeeded":
            if applied:
                succeeded.append({"clip_id": clip.id, "model": outcome.model})
            else:
                failed.append({"clip_id": clip.id, "code": "stale_result"})

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
    from core.analysis.faces import extract_faces_from_clip, unload_model

    clips = _resolve_clip_ids(project, clip_ids)
    sources_by_id = project.sources_by_id
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    updated = []
    total = len(clips)

    try:
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
    finally:
        # Long-lived MCP servers must not retain InsightFace state across jobs.
        unload_model()


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
    from core.analysis.gaze import extract_gaze_from_clip, unload_model

    clips = _resolve_clip_ids(project, clip_ids)
    sources_by_id = project.sources_by_id
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    updated = []
    total = len(clips)

    try:
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
    finally:
        # Long-lived MCP servers must not retain MediaPipe state across jobs.
        unload_model()


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
    from core.operations.custom_query import CustomQueryTask, resolve_options, run_custom_query

    if not query or not query.strip():
        return {"success": False, "error": {"code": "missing_query", "message": "query is required"}}

    clips = _resolve_clip_ids(project, clip_ids)
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    updated = []
    total = len(clips)

    query = query.strip()
    tasks = tuple(CustomQueryTask(
        clip.id, _thumbnail_for_clip(clip), query,
        skip=bool(skip_existing and clip.custom_queries and any(q.get("query") == query for q in clip.custom_queries)),
    ) for clip in clips)

    def report(current: int, count: int) -> None:
        if progress_callback is not None:
            progress_callback(current / count if count else 1.0, f"Custom query ({current}/{count})")

    outcomes = run_custom_query(tasks, resolve_options(tier), cancel_event=cancel_event, progress=report)
    for clip, outcome in zip(clips, outcomes):
        if outcome.status == "skipped":
            skipped.append({"clip_id": clip.id, "reason": outcome.code})
            continue
        if outcome.status != "succeeded":
            if outcome.status == "failed":
                failure = {"clip_id": clip.id, "code": outcome.code}
                if outcome.message:
                    failure["message"] = outcome.message
                failed.append(failure)
            continue
        if clip.custom_queries is None:
            clip.custom_queries = []
        result = {
            "query": query,
            "match": outcome.match,
            "confidence": round(outcome.confidence or 0.0, 4),
            "model": outcome.model,
        }
        clip.custom_queries.append(result)
        updated.append(clip)
        succeeded.append({"clip_id": clip.id, **result})

    if updated:
        project.update_clips(updated)
    if progress_callback is not None:
        progress_callback(1.0, f"Done: {len(succeeded)} ok, {len(failed)} failed, {len(skipped)} skipped")
    return {"success": True, "result": {"succeeded": succeeded, "failed": failed, "skipped": skipped, "total_clips": total}}


ANALYZE_CLIP_OPERATION_MAP: dict[str, Callable[..., dict]] = {
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
    from core.operations.analysis_plan import run_analysis_plan

    def execute(op, progress):
        kwargs: dict[str, object] = {"skip_existing": skip_existing}
        if op == "custom_query":
            kwargs["query"] = query
            # Custom query appends user-authored query runs by default.
            kwargs["skip_existing"] = False
        return ANALYZE_CLIP_OPERATION_MAP[op](
            project,
            clip_ids,
            progress_callback=progress,
            cancel_event=cancel_event,
            **kwargs,
        )
    return run_analysis_plan(
        operations, execute, progress=progress_callback, cancel=cancel_event
    )

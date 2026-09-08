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

    from core.operations.shots import (
        ShotTypeTask, ShotTypeOptions, ShotTypeApplication, run_shot_types,
    )

    clips = _resolve_clip_ids(project, clip_ids)
    if not clips:
        return {
            "success": True,
            "result": {"succeeded": [], "failed": [], "skipped": []},
        }

    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    total = len(clips)
    tasks = tuple(
        ShotTypeTask(
            clip.id, Path(clip.thumbnail_path) if clip.thumbnail_path else None,
            source.file_path if (source := project.sources_by_id.get(clip.source_id)) else None,
            clip.start_frame, clip.end_frame, source.fps if source else None,
            skip=skip_existing and bool(clip.shot_type),
        ) for clip in clips
    )
    application = ShotTypeApplication(project, tasks)

    def report(current, count):
        if progress_callback is not None:
            progress_callback(current / count if count else 1.0,
                              f"Shot classification ({current}/{count})")

    outcomes = run_shot_types(tasks, ShotTypeOptions(), cancel_event=cancel_event, progress=report)
    unprocessed = []
    for outcome in outcomes:
        if outcome.status == "succeeded":
            if application.apply(project, outcome):
                succeeded.append({"clip_id": outcome.clip_id, "shot_type": outcome.shot_type})
            else:
                failed.append({"clip_id": outcome.clip_id, "code": "stale_input"})
        elif outcome.status == "skipped":
            skipped.append({"clip_id": outcome.clip_id, "reason": outcome.code})
        elif outcome.status == "unprocessed":
            unprocessed.append({"clip_id": outcome.clip_id, "code": outcome.code})
        else:
            failure = {"clip_id": outcome.clip_id, "code": outcome.code}
            if outcome.message:
                failure["message"] = outcome.message
            failed.append(failure)

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
            "unprocessed": unprocessed,
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
    from core.operations.classification import ClassificationApplication, ClassificationTask, ClassificationOptions, run_classification

    clips = _resolve_clip_ids(project, clip_ids)
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    total = len(clips)

    tasks = tuple(ClassificationTask(clip.id, _thumbnail_for_clip(clip), skip=skip_existing and clip.object_labels is not None) for clip in clips)
    application = ClassificationApplication(project, tasks)

    def report(current: int, count: int) -> None:
        if progress_callback is not None:
            progress_callback(current / count if count else 1.0, f"Content classification ({current}/{count})")

    outcomes = run_classification(tasks, ClassificationOptions(top_k, threshold), cancel_event=cancel_event, progress=report)
    for clip, outcome, accepted in zip(clips, outcomes, application.apply_batch(project, outcomes)):
        if outcome.status == "skipped":
            skipped.append({"clip_id": clip.id, "reason": outcome.code})
        elif outcome.status == "failed":
            failure = {"clip_id": clip.id, "code": outcome.code}
            if outcome.message:
                failure["message"] = outcome.message
            failed.append(failure)
        elif outcome.status == "succeeded":
            if accepted:
                succeeded.append({"clip_id": clip.id, "label_count": len(outcome.labels)})
            else:
                failed.append({"clip_id": clip.id, "code": "stale_result"})
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
    from core.operations.object_detection import ObjectDetectionApplication, ObjectDetectionOptions, object_detection_task, run_object_detection

    clips = _resolve_clip_ids(project, clip_ids)
    clips_by_id = {clip.id: clip for clip in clips}
    result = {"succeeded": [], "failed": [], "skipped": [], "unprocessed": [], "total_clips": len(clips)}
    tasks = tuple(object_detection_task(clip, project.sources_by_id.get(clip.source_id), image_path=_thumbnail_for_clip(clip), skip_existing=skip_existing, detect_all=detect_all) for clip in clips)
    options = ObjectDetectionOptions(confidence, detect_all)
    application = ObjectDetectionApplication(project, tasks, options)

    def deliver(outcome):
        clip = clips_by_id[outcome.clip_id]
        if outcome.has_result:
            if application.apply(project, outcome):
                if outcome.status == "skipped":
                    result["skipped"].append({"clip_id": clip.id, "reason": outcome.code})
                else:
                    result["succeeded"].append({"clip_id": clip.id, "object_count": len(outcome.detections), "person_count": outcome.person_count})
            else:
                result["failed"].append({"clip_id": clip.id, "code": "stale_result"})
        elif outcome.status == "skipped":
            result["skipped"].append({"clip_id": clip.id, "reason": outcome.code})
        elif outcome.status == "failed":
            if outcome.can_apply:
                application.apply(project, outcome)
            failure = {"clip_id": clip.id, "code": outcome.code}
            if outcome.message:
                failure["message"] = outcome.message
            result["failed"].append(failure)

    def report(current, total):
        if progress_callback:
            progress_callback(current / total if total else 1.0, f"Object detection ({current}/{total})")

    outcomes = run_object_detection(tasks, options,
        cancel_event=cancel_event, on_outcome=deliver, progress=report)
    result["unprocessed"] = [{"clip_id": outcome.clip_id, "code": outcome.code}
        for outcome in outcomes if outcome.status == "unprocessed"]
    if progress_callback:
        progress_callback(1.0, f"Done: {len(result['succeeded'])} ok, {len(result['failed'])} failed, {len(result['skipped'])} skipped")
    return {"success": True, "result": result}


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
    from core.operations.ocr import OcrTask, OcrOptions, OcrApplication, run_ocr

    clips = _resolve_clip_ids(project, clip_ids)
    tasks = tuple(
        OcrTask.from_clip(clip, project.sources_by_id.get(clip.source_id),
                          skip=skip_existing and clip.extracted_texts is not None)
        for clip in clips
    )
    application = OcrApplication(project, tasks)
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []

    def deliver(outcome):
        if outcome.status == "skipped":
            skipped.append({"clip_id": outcome.clip_id, "reason": "already_populated"})
        elif outcome.status == "failed":
            entry = {"clip_id": outcome.clip_id, "code": outcome.code}
            if outcome.message:
                entry["message"] = outcome.message
            failed.append(entry)
        elif outcome.status == "succeeded":
            if application.apply(project, outcome):
                succeeded.append({"clip_id": outcome.clip_id, "text_count": len(outcome.texts)})
            else:
                failed.append({"clip_id": outcome.clip_id, "code": "target_changed"})

    run_ocr(
        tasks, OcrOptions(min(max(1, num_keyframes), 5), use_vlm_fallback, vlm_model, vlm_only),
        cancel_event=cancel_event, on_outcome=deliver,
        progress=(lambda n, total, cid: progress_callback((n - 1) / total, f"Text extraction ({n}/{total}): {cid}"))
        if progress_callback else None,
    )
    if progress_callback is not None:
        progress_callback(1.0, f"Done: {len(succeeded)} ok, {len(failed)} failed, {len(skipped)} skipped")
    return {"success": True, "result": {"succeeded": succeeded, "failed": failed, "skipped": skipped, "total_clips": len(tasks)}}


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
    from core.operations.cinematography import CinematographyApplication, CinematographyTask, resolve_options, run_cinematography

    clips = _resolve_clip_ids(project, clip_ids)
    sources_by_id = project.sources_by_id
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    total = len(clips)

    tasks = []
    for clip in clips:
        source = sources_by_id.get(clip.source_id)
        tasks.append(CinematographyTask(
            clip.id, _thumbnail_for_clip(clip),
            source.file_path if source and source.file_path.exists() else None,
            clip.start_frame, clip.end_frame, source.fps if source else None,
            skip=skip_existing and clip.cinematography is not None,
        ))

    def report(current: int, count: int, cid: str) -> None:
        if progress_callback is not None:
            progress_callback(current / count if count else 1.0, f"Cinematography ({current}/{count}): {cid}")

    application = CinematographyApplication(project, tuple(tasks))
    outcomes = run_cinematography(tuple(tasks), resolve_options(mode, model), cancel_event=cancel_event, progress=report)
    accepted = application.apply_batch(project, outcomes)
    for clip, outcome, applied in zip(clips, outcomes, accepted):
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
        analysis = outcome.analysis
        if not applied:
            failed.append({"clip_id": clip.id, "code": "stale_result"})
            continue
        succeeded.append({"clip_id": clip.id, "shot_size": getattr(analysis, "shot_size", None)})

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
    from core.operations.faces import FaceTask, FaceOptions, FaceApplication, run_faces

    clips = _resolve_clip_ids(project, clip_ids)
    sources = project.sources_by_id
    tasks = tuple(FaceTask(c.id, c.source_id,
        sources[c.source_id].file_path if c.source_id in sources else None,
        c.start_frame, c.end_frame,
        sources[c.source_id].fps if c.source_id in sources else 0.0,
        skip=skip_existing and c.face_embeddings is not None) for c in clips)
    application = FaceApplication(project, tasks)
    result: dict = {"succeeded": [], "failed": [], "skipped": [], "unprocessed": [], "total_clips": len(clips)}
    def deliver(outcome):
        if outcome.status == "succeeded":
            if application.apply(project, outcome):
                result["succeeded"].append({"clip_id": outcome.clip_id, "face_count": len(outcome.faces)})
            else:
                result["failed"].append({"clip_id": outcome.clip_id, "code": "stale_result"})
        elif outcome.status == "skipped":
            result["skipped"].append({"clip_id": outcome.clip_id, "reason": outcome.code})
        elif outcome.status == "failed":
            result["failed"].append({"clip_id": outcome.clip_id, "code": outcome.code, "message": outcome.message})
    def report(current, total):
        if progress_callback:
            progress_callback(current / total if total else 1.0, f"Face detection ({current}/{total})")
    outcomes = run_faces(tasks, FaceOptions(sample_interval), cancel_event=cancel_event,
                         on_outcome=deliver, progress=report)
    result["unprocessed"] = [{"clip_id": outcome.clip_id, "code": outcome.code} for outcome in outcomes if outcome.status == "unprocessed"]
    if progress_callback:
        progress_callback(1.0, "Face detection finished")
    return {"success": True, "result": result}


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
    from core.operations.gaze import GazeTask, GazeOptions, GazeApplication, run_gaze

    clips = _resolve_clip_ids(project, clip_ids)
    sources = project.sources_by_id
    tasks = tuple(
        GazeTask(c.id, c.source_id,
                 sources[c.source_id].file_path if c.source_id in sources else None,
                 c.start_frame, c.end_frame,
                 sources[c.source_id].fps if c.source_id in sources else 0.0,
                 skip=skip_existing and c.gaze_category is not None)
        for c in clips
    )
    application = GazeApplication(project, tasks)
    result = {"succeeded": [], "failed": [], "skipped": [], "unprocessed": [], "total_clips": len(tasks)}

    def deliver(outcome):
        if outcome.status == "succeeded" and outcome.category is None:
            # Preserve the legacy wire response for an empty observation.
            result["failed"].append({"clip_id": outcome.clip_id, "code": "no_gaze_detected"})
        elif outcome.status == "succeeded":
            if application.apply(project, outcome):
                result["succeeded"].append({"clip_id": outcome.clip_id, "gaze_category": outcome.category})
            else:
                result["failed"].append({"clip_id": outcome.clip_id, "code": "stale_input"})
        elif outcome.status == "skipped":
            result["skipped"].append({"clip_id": outcome.clip_id, "reason": outcome.code})
        elif outcome.status == "failed":
            result["failed"].append({"clip_id": outcome.clip_id, "code": outcome.code, "message": outcome.message})

    outcomes = run_gaze(
        tasks, GazeOptions(sample_interval), cancel_event=cancel_event,
        on_outcome=deliver,
        progress=(lambda current, total: progress_callback(current / total if total else 1.0, f"Gaze analysis ({current}/{total})")) if progress_callback else None,
    )
    result["unprocessed"] = [{"clip_id": o.clip_id, "code": o.code} for o in outcomes if o.status == "unprocessed"]
    if progress_callback is not None:
        progress_callback(1.0, f"Done: {len(result['succeeded'])} ok, {len(result['failed'])} failed, {len(result['skipped'])} skipped")
    return {"success": True, "result": result}


def embeddings(
    project,
    clip_ids: Optional[list[str]] = None,
    *,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Extract DINOv2 embeddings from clip thumbnails."""
    from core.operations.embeddings import (
        EmbeddingApplication, EmbeddingOptions, embedding_task, run_embeddings,
    )

    tasks = tuple(
        embedding_task(c, project.sources_by_id.get(c.source_id), skip_existing=skip_existing)
        for c in _resolve_clip_ids(project, clip_ids)
    )
    application = EmbeddingApplication(project, tasks)
    result = {"succeeded": [], "failed": [], "skipped": [], "unprocessed": [], "total_clips": len(tasks)}

    def deliver(outcome):
        if outcome.status == "succeeded":
            if application.apply(project, outcome):
                result["succeeded"].append({"clip_id": outcome.clip_id, "embedding_dim": len(outcome.vector)})
            else:
                result["failed"].append({"clip_id": outcome.clip_id, "code": "stale_result"})
        elif outcome.status == "skipped":
            if outcome.record_json is not None and not application.apply(project, outcome):
                result["failed"].append({"clip_id": outcome.clip_id, "code": "stale_result"})
            else:
                result["skipped"].append({"clip_id": outcome.clip_id, "reason": outcome.code})
        elif outcome.status == "failed":
            item = {"clip_id": outcome.clip_id, "code": outcome.code}
            if outcome.message:
                item["message"] = outcome.message
            result["failed"].append(item)

    outcomes = run_embeddings(
        tasks, EmbeddingOptions(), cancel_event=cancel_event, on_outcome=deliver,
        progress=lambda n, total: progress_callback(n / total if total else 1.0, f"Embeddings ({n}/{total})") if progress_callback else None,
    )
    result["unprocessed"] = [{"clip_id": o.clip_id, "code": o.code} for o in outcomes if o.status == "unprocessed"]
    return {"success": True, "result": result}


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
    from core.operations.custom_query import CustomQueryApplication, CustomQueryTask, resolve_options, run_custom_query

    if not query or not query.strip():
        return {"success": False, "error": {"code": "missing_query", "message": "query is required"}}

    clips = _resolve_clip_ids(project, clip_ids)
    succeeded: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []
    total = len(clips)

    query = query.strip()
    tasks = tuple(CustomQueryTask(
        clip.id, _thumbnail_for_clip(clip), query,
        skip=bool(skip_existing and clip.custom_queries and any(q.get("query") == query for q in clip.custom_queries)),
    ) for clip in clips)

    def report(current: int, count: int) -> None:
        if progress_callback is not None:
            progress_callback(current / count if count else 1.0, f"Custom query ({current}/{count})")

    application = CustomQueryApplication(project, tasks)
    outcomes = run_custom_query(tasks, resolve_options(tier), cancel_event=cancel_event, progress=report)
    accepted = application.apply_batch(project, outcomes)
    for clip, outcome, applied in zip(clips, outcomes, accepted):
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
        if not applied:
            failed.append({"clip_id": clip.id, "code": "stale_result"})
            continue
        result = {
            "query": query,
            "match": outcome.match,
            "confidence": round(outcome.confidence or 0.0, 4),
            "model": outcome.model,
        }
        succeeded.append({"clip_id": clip.id, **result})

    if progress_callback is not None:
        progress_callback(1.0, f"Done: {len(succeeded)} ok, {len(failed)} failed, {len(skipped)} skipped")
    return {"success": True, "result": {"succeeded": succeeded, "failed": failed, "skipped": skipped, "total_clips": total}}


def boundary_embeddings(
    project: "Project", clip_ids: Optional[list[str]] = None, *,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Analyze first/last source frames and publish each validated pair together."""
    from core.operations.boundary_embeddings import (
        BoundaryEmbeddingApplication, BoundaryEmbeddingTask, run_boundary_embeddings,
    )

    clips = _resolve_clip_ids(project, clip_ids)
    tasks = tuple(BoundaryEmbeddingTask(
        clip.id, project.sources_by_id[clip.source_id].file_path if clip.source_id in project.sources_by_id else None,
        clip.start_frame, clip.end_frame,
        project.sources_by_id[clip.source_id].fps if clip.source_id in project.sources_by_id else 0.0,
        skip_existing and clip.first_frame_embedding is not None and clip.last_frame_embedding is not None,
    ) for clip in clips)
    application = BoundaryEmbeddingApplication(project, tasks)
    result = {"succeeded": [], "failed": [], "skipped": [], "unprocessed": [], "total_clips": len(tasks)}
    for outcome in run_boundary_embeddings(tasks, cancel_event=cancel_event):
        if outcome.status == "succeeded":
            if application.apply(project, outcome):
                result["succeeded"].append({"clip_id": outcome.clip_id})
            else:
                result["failed"].append({"clip_id": outcome.clip_id, "code": "stale_result"})
        else:
            result[outcome.status].append({"clip_id": outcome.clip_id, "code": outcome.code, "message": outcome.message})
    if progress_callback is not None:
        progress_callback(1.0, "Boundary embeddings finished")
    return {"success": True, "result": result}


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
    "boundary_embeddings": boundary_embeddings,
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

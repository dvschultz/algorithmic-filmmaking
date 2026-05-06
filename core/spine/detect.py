"""Scene detection spine.

Reconciles two existing ``detect_scenes`` entry points onto a single
canonical impl plus thin compose-then-detect helpers:

- ``detect_scenes_for_source`` — operates on an existing source, replaces
  its clips. The canonical impl. Used by the GUI agent (chat tool) and the
  MCP bulk job.
- ``detect_scenes_for_video`` — find-or-create source from a video file
  path, then dispatch to ``detect_scenes_for_source``. Matches the GUI
  agent's existing detect_scenes signature.
- ``detect_scenes_new_project`` — Project.new() + add_source() + detect.
  Matches today's MCP "create-then-detect" shape.
- ``detect_scenes_bulk`` — iterate per-source, aggregate per-item failures.
  Used by the MCP ``start_detect_scenes_bulk`` job.

All entry points are sync ``def`` with optional ``progress_callback`` and
``cancel_event`` (R15). The MCP wrappers run them via ``asyncio.to_thread``;
GUI calls them directly from the chat-worker QThread.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _check_cancel(cancel_event: Optional[threading.Event]) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def detect_scenes_for_source(
    project,
    source_id: str,
    sensitivity: float = 3.0,
    *,
    luma_only: Optional[bool] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Detect scenes on a single existing source. Mutates the project in
    place via ``project.replace_source_clips(source_id, new_clips)``.

    Returns ``{"success": bool, "result": {"clips": [...], "skipped": ...}}``
    for happy path, or a structured error dict on failure.
    """
    from core.scene_detect import DetectionConfig, SceneDetector

    source = project.sources_by_id.get(source_id)
    if source is None:
        return {
            "success": False,
            "error": {
                "code": "unknown_source_id",
                "source_id": source_id,
                "message": (
                    f"Source {source_id!r} not in project. "
                    "Use list_sources to discover ids."
                ),
            },
        }

    if not source.file_path.exists():
        return {
            "success": False,
            "error": {
                "code": "source_file_missing",
                "source_id": source_id,
                "path": str(source.file_path),
            },
        }

    if progress_callback is not None:
        progress_callback(0.0, f"Detecting scenes in {source.filename}")

    config = DetectionConfig(threshold=sensitivity, luma_only=luma_only)
    detector = SceneDetector(config)

    if _check_cancel(cancel_event):
        return {"success": False, "error": {"code": "cancelled"}}

    try:
        detected_source, clips = detector.detect_scenes(source.file_path)
    except FileNotFoundError as exc:
        return {
            "success": False,
            "error": {
                "code": "source_file_missing",
                "source_id": source_id,
                "message": str(exc),
            },
        }

    # Reuse the existing source id so downstream references (sequence clips,
    # frames) stay valid.
    for clip in clips:
        clip.source_id = source.id
    source.analyzed = True

    project.replace_source_clips(source.id, clips)

    if progress_callback is not None:
        progress_callback(1.0, f"Detected {len(clips)} clips in {source.filename}")

    return {
        "success": True,
        "result": {
            "source_id": source.id,
            "source_name": source.filename,
            "clip_count": len(clips),
            "clip_ids": [c.id for c in clips],
        },
    }


def detect_scenes_for_video(
    project,
    video_path: str | Path,
    sensitivity: float = 3.0,
    *,
    luma_only: Optional[bool] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Find-or-create the source for ``video_path`` then detect scenes.

    Matches the GUI agent's existing ``detect_scenes(project, video_path)``
    shape: if a source for the resolved file path already exists, its clips
    are replaced; otherwise a new source is added.
    """
    from core.scene_detect import DetectionConfig, SceneDetector

    video = Path(video_path).expanduser()
    if not video.exists():
        return {
            "success": False,
            "error": {
                "code": "source_file_missing",
                "path": str(video),
            },
        }

    resolved = video.resolve()
    existing_source = None
    for s in project.sources:
        try:
            if s.file_path.resolve() == resolved:
                existing_source = s
                break
        except (OSError, ValueError):
            if s.file_path == video:
                existing_source = s
                break

    if _check_cancel(cancel_event):
        return {"success": False, "error": {"code": "cancelled"}}

    config = DetectionConfig(threshold=sensitivity, luma_only=luma_only)
    detector = SceneDetector(config)
    if progress_callback is not None:
        progress_callback(0.0, f"Detecting scenes in {video.name}")

    detected_source, clips = detector.detect_scenes(video)

    if existing_source:
        source = existing_source
        for clip in clips:
            clip.source_id = source.id
        source.analyzed = True
        project.replace_source_clips(source.id, clips)
    else:
        source = detected_source
        source.analyzed = True
        project.add_source(source)
        if clips:
            project.add_clips(clips)

    if progress_callback is not None:
        progress_callback(1.0, f"Detected {len(clips)} clips in {video.name}")

    return {
        "success": True,
        "result": {
            "source_id": source.id,
            "source_name": source.filename,
            "clip_count": len(clips),
            "clip_ids": [c.id for c in clips],
            "is_new_source": existing_source is None,
        },
    }


def detect_scenes_new_project(
    video_path: str | Path,
    output_project_path: str | Path,
    sensitivity: float = 3.0,
    *,
    luma_only: Optional[bool] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Create a fresh project, add ``video_path`` as a source, detect scenes,
    and save to ``output_project_path``.

    Matches today's MCP ``detect_scenes`` shape.
    """
    from core.project import Project, ProjectMetadata

    video = Path(video_path).expanduser()
    output = Path(output_project_path).expanduser()
    if not video.exists():
        return {
            "success": False,
            "error": {
                "code": "source_file_missing",
                "path": str(video),
            },
        }

    project = Project.new(name=video.stem)
    project.metadata = ProjectMetadata(name=video.stem)

    # Reuse detect_scenes_for_video for the actual detection — but it
    # expects an existing project, which we just created.
    detection = detect_scenes_for_video(
        project,
        video,
        sensitivity=sensitivity,
        luma_only=luma_only,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    if not detection.get("success"):
        return detection

    output.parent.mkdir(parents=True, exist_ok=True)
    if not project.save(output):
        return {
            "success": False,
            "error": {"code": "save_failed", "path": str(output)},
        }

    payload = detection["result"]
    payload["project_path"] = str(output)
    return {"success": True, "result": payload}


def detect_scenes_bulk(
    project,
    source_ids: list[str],
    sensitivity: float = 3.0,
    *,
    luma_only: Optional[bool] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Run scene detection on each source in ``source_ids``.

    Per-source granularity for cancellation: between sources we check
    ``cancel_event``; an interrupted run reports the sources that finished
    in ``succeeded`` and the rest in ``cancelled``. Per-source failures are
    aggregated to ``failed`` and never raise mid-batch (R20).
    """
    succeeded: list[dict] = []
    failed: list[dict] = []
    cancelled: list[str] = []

    total = max(len(source_ids), 1)
    for i, source_id in enumerate(source_ids):
        if _check_cancel(cancel_event):
            cancelled.extend(source_ids[i:])
            break

        if progress_callback is not None:
            progress_callback(
                i / total,
                f"Detecting scenes ({i + 1}/{len(source_ids)}): {source_id}",
            )

        result = detect_scenes_for_source(
            project,
            source_id,
            sensitivity=sensitivity,
            luma_only=luma_only,
            cancel_event=cancel_event,
        )

        if result.get("success"):
            succeeded.append(result["result"])
        else:
            err = result.get("error", {})
            failed.append(
                {
                    "source_id": source_id,
                    "code": err.get("code", "unknown"),
                    "message": err.get("message"),
                }
            )

    if progress_callback is not None:
        progress_callback(1.0, f"Done: {len(succeeded)} ok, {len(failed)} failed")

    return {
        "success": True,
        "result": {
            "succeeded": succeeded,
            "failed": failed,
            "cancelled": cancelled,
        },
    }

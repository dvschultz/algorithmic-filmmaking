"""Headless adapters for the shared detached thumbnail operation."""

from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Callable, TYPE_CHECKING

from core.operations.thumbnails import (
    ThumbnailApplication,
    ThumbnailOptions,
    ThumbnailTask,
    run_thumbnails,
    thumbnail_payload,
)

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip, Source


def _prepare(
    pairs,
    *,
    force,
    width,
    height,
    progress_callback,
    progress_start,
    progress_end,
):
    from core.settings import load_settings

    tasks = tuple(ThumbnailTask.capture(clip, source) for clip, source in pairs)
    options = ThumbnailOptions(
        load_settings().thumbnail_cache_dir, width, height, force
    )

    def progress(current, total, outcome):
        if progress_callback is not None:
            progress_callback(
                progress_start
                + max(0.0, progress_end - progress_start) * current / total,
                f"Generating thumbnails ({current}/{total})",
            )

    return tasks, options, progress


def generate_clip_thumbnails(
    clip_source_pairs: list[tuple["Clip", "Source | None"]],
    *,
    force: bool = False,
    width: int = 320,
    height: int = 180,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_start: float = 0.0,
    progress_end: float = 1.0,
    cancel_event: Event | None = None,
) -> dict:
    """Compute thumbnails and update caller-owned detached detection clips."""
    tasks, options, progress = _prepare(
        clip_source_pairs,
        force=force,
        width=width,
        height=height,
        progress_callback=progress_callback,
        progress_start=progress_start,
        progress_end=progress_end,
    )
    cancel = cancel_event if cancel_event is not None else Event()
    outcomes = run_thumbnails(tasks, options, cancel, progress)
    accepted = []
    for (clip, source), task, outcome in zip(clip_source_pairs, tasks, outcomes):
        if outcome.status == "succeeded":
            if cancel.is_set() or ThumbnailTask.capture(clip, source) != task:
                outcome = replace(
                    outcome,
                    status="failed",
                    code="cancelled" if cancel.is_set() else "stale_input",
                    path=None,
                )
            else:
                assert outcome.path is not None
                clip.thumbnail_path = Path(outcome.path)
        accepted.append(outcome)
    return thumbnail_payload(tuple(accepted))


def generate_thumbnails(
    project: "Project",
    clip_ids: list[str] | None = None,
    *,
    force: bool = False,
    width: int = 320,
    height: int = 180,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_event: Event | None = None,
) -> dict:
    """Generate project thumbnails and publish on the calling owner thread."""
    project.session.assert_owner()
    clips = (
        list(project.clips)
        if clip_ids is None
        else [
            project.clips_by_id[cid]
            for cid in dict.fromkeys(clip_ids)
            if cid in project.clips_by_id
        ]
    )
    pairs = [(clip, project.sources_by_id.get(clip.source_id)) for clip in clips]
    tasks, options, progress = _prepare(
        pairs,
        force=force,
        width=width,
        height=height,
        progress_callback=progress_callback,
        progress_start=0.0,
        progress_end=1.0,
    )
    application = ThumbnailApplication(project, tasks)
    cancel = cancel_event if cancel_event is not None else Event()
    outcomes = run_thumbnails(tasks, options, cancel, progress)
    accepted = []
    for outcome in outcomes:
        if outcome.status == "succeeded" and (
            cancel.is_set() or not application.apply(outcome)
        ):
            outcome = replace(
                outcome,
                status="failed",
                code="cancelled" if cancel.is_set() else "stale_input",
                path=None,
            )
        accepted.append(outcome)
    result = thumbnail_payload(tuple(accepted))
    if progress_callback is not None:
        progress_callback(
            1.0,
            f"Done: {len(result['succeeded'])} generated, {len(result['failed'])} failed, {len(result['skipped'])} skipped",
        )
    return {"success": True, "result": result}

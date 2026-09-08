"""Media identity captured before UI gates and asynchronous analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.jobs.media import media_stamp

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip


def clip_input(project: Project, clip: Clip) -> tuple:
    source = project.sources_by_id.get(clip.source_id)
    return (
        id(clip),
        id(source),
        clip.source_id,
        clip.start_frame,
        clip.end_frame,
        clip.thumbnail_path,
        media_stamp(clip.thumbnail_path) if clip.thumbnail_path else None,
        source.file_path if source else None,
        source.fps if source else None,
        media_stamp(source.file_path) if source else None,
    )

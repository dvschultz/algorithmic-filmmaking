"""Source lookup, metadata preparation, and project admission.

Pure-Python helpers for project sources, used by both the chat-tools agent
and the MCP server. No PySide6, no main_window, no GUI state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
import logging

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Source

logger = logging.getLogger(__name__)


def probe_source(path: Path | str) -> Source:
    """Read video metadata without touching a project; fall back to defaults.

    This is a blocking media probe. Background import adapters can prepare the
    source here, then admit it on the project owner thread.
    """
    from models.clip import Source

    path = Path(path).expanduser()
    source = Source(file_path=path)
    try:
        from core.ffmpeg import FFmpegProcessor

        info = FFmpegProcessor().get_video_info(path)
        source.duration_seconds = info.get("duration", 0.0)
        source.fps = info.get("fps", 30.0)
        source.width = info.get("width", 0)
        source.height = info.get("height", 0)
    except Exception as exc:
        logger.warning("Failed to extract metadata for %s: %s", path.name, exc)
    return source


def same_source_path(first: Path | str, second: Path | str) -> bool:
    """Compare canonical paths, then physical identity for available media."""

    def canonical(value: Path | str) -> Path:
        path = Path(value).expanduser()
        try:
            return path.resolve()
        except (OSError, RuntimeError):
            # One inaccessible/offline source must not block unrelated imports.
            return path.absolute()

    first = canonical(first)
    second = canonical(second)
    if first == second:
        return True
    try:
        return first.samefile(second)
    except OSError:
        return False


def find_source_by_path(project: Project, path: Path | str) -> Source | None:
    """Resolve lexical aliases and, for online media, physical file identity."""
    for source in project.sources:
        if same_source_path(source.file_path, path):
            return source
    return None


def add_source_if_missing(project: Project, source: Source) -> tuple[Source, bool]:
    """Admit prepared metadata once on the project owner thread.

    Return the existing source unchanged on retry; do not replace its identity,
    analysis, or clips with freshly probed defaults.
    """
    project.session.assert_owner()
    existing = find_source_by_path(project, source.file_path)
    if existing is not None:
        return existing, False
    if source.id in project.sources_by_id:
        raise ValueError("Source ID already belongs to different media")
    project.add_source(source)
    return source, True


def remove_source(project: Project, source_id: str) -> dict:
    """Resolve a source ID or filename and remove its references reversibly."""
    source = project.sources_by_id.get(source_id)
    if source is None:
        source = next(
            (
                s
                for s in project.sources
                if s.file_path
                and source_id
                in (
                    s.file_path.name,
                    s.file_path.stem,
                )
            ),
            None,
        )
    if source is None:
        return {
            "success": False,
            "error": f"Source '{source_id}' not found. Use list_sources to see available sources.",
        }
    count = len(project.clips_by_source.get(source.id, []))
    name = source.filename if source.file_path else f"Source {source.id[:8]}"
    try:
        project.remove_source(source.id)
    except (ValueError, RuntimeError) as exc:
        return {"success": False, "error": str(exc)}
    return {
        "success": True,
        "message": f"Removed source '{name}' and {count} associated clips",
        "removed_source": name,
        "removed_clips_count": count,
    }


def list_sources(project) -> dict:
    """List all video sources in the project."""
    sources = []
    for s in project.sources:
        clip_count = len(project.clips_by_source.get(s.id, []))
        sources.append(
            {
                "id": s.id,
                "filename": s.file_path.name if s.file_path else "Unknown",
                "duration": s.duration_seconds,
                "fps": s.fps,
                "width": s.width,
                "height": s.height,
                "clip_count": clip_count,
                "analyzed": s.analyzed,
            }
        )

    return {
        "success": True,
        "sources": sources,
        "count": len(sources),
    }


__all__ = [
    "list_sources",
    "remove_source",
    "same_source_path",
    "find_source_by_path",
    "add_source_if_missing",
    "probe_source",
]

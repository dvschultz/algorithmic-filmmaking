"""Source-listing spine impls.

Pure-Python helpers for project sources, used by both the chat-tools agent
and the MCP server. No PySide6, no main_window, no GUI state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.project import Project


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


__all__ = ["list_sources", "remove_source"]

"""Source-listing spine impls.

Pure-Python helpers for project sources, used by both the chat-tools agent
and the MCP server. No PySide6, no main_window, no GUI state.
"""

from __future__ import annotations


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


__all__ = ["list_sources"]

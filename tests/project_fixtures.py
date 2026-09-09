"""Qt-free project fixtures shared by the desktop and MCP test suites."""

from __future__ import annotations

from pathlib import Path


def project_with_thumbnails(tmp_path: Path, count: int = 3):
    """One source, ``count`` clips, every clip pointing at one fake thumbnail."""
    from tests.test_spine_analyze import _build_project

    project = _build_project(tmp_path, count)
    thumbnail = tmp_path / "thumb.jpg"
    thumbnail.write_bytes(b"fake")
    for clip in project.clips:
        clip.thumbnail_path = thumbnail
    return project

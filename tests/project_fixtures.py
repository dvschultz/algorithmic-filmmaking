"""Qt-free project fixtures shared by the desktop and MCP test suites."""

from __future__ import annotations

from pathlib import Path


def build_project(tmp_path: Path, n_clips: int = 3, populate_colors: int = 0):
    """Project with one source and ``n_clips`` clips. The first
    ``populate_colors`` clips have ``dominant_colors`` pre-set to exercise
    skip-existing."""
    from core.project import Project
    from models.clip import Clip, Source

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")

    project = Project.new(name="test")
    source = Source(
        id="src-1",
        file_path=video,
        duration_seconds=120.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    project.add_source(source)

    clips = [
        Clip(
            id=f"c-{i}",
            source_id=source.id,
            start_frame=i * 60,
            end_frame=(i + 1) * 60,
        )
        for i in range(n_clips)
    ]
    for clip in clips[:populate_colors]:
        clip.dominant_colors = [(255, 0, 0)]
    project.add_clips(clips)
    return project


_build_project = build_project


def project_with_thumbnails(tmp_path: Path, count: int = 3):
    """One source, ``count`` clips, every clip pointing at one fake thumbnail."""
    project = build_project(tmp_path, count)
    thumbnail = tmp_path / "thumb.jpg"
    thumbnail.write_bytes(b"fake")
    for clip in project.clips:
        clip.thumbnail_path = thumbnail
    return project

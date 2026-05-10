"""Tests for headless thumbnail generation spine."""

from pathlib import Path
from unittest.mock import patch

from core.spine.thumbnails import generate_thumbnails


class _FakeSettings:
    def __init__(self, thumbnail_cache_dir: Path):
        self.thumbnail_cache_dir = thumbnail_cache_dir


class _FakeThumbnailGenerator:
    calls: list[Path] = []

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_clip_thumbnail(
        self,
        video_path,
        start_seconds,
        end_seconds,
        output_path=None,
        width=320,
        height=180,
    ):
        output_path = output_path or self.cache_dir / "generated.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"thumb")
        self.calls.append(output_path)
        return output_path


def _build_project_with_clips(tmp_path: Path):
    from core.project import Project
    from models.clip import Clip, Source

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")

    project = Project.new(name="thumbs")
    source = Source(
        id="src-1",
        file_path=video,
        duration_seconds=60.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    clips = [
        Clip(id="clip-1", source_id=source.id, start_frame=0, end_frame=30),
        Clip(id="clip-2", source_id=source.id, start_frame=30, end_frame=60),
    ]
    project.add_source(source)
    project.add_clips(clips)
    return project, clips


def _stub_thumbnail_generation(tmp_path: Path):
    _FakeThumbnailGenerator.calls = []
    return patch.multiple(
        "core.thumbnail",
        ThumbnailGenerator=_FakeThumbnailGenerator,
    ), patch(
        "core.settings.load_settings",
        return_value=_FakeSettings(tmp_path / "thumbs"),
    )


def test_generate_thumbnails_sets_clip_thumbnail_paths(tmp_path):
    project, clips = _build_project_with_clips(tmp_path)
    thumb_patch, settings_patch = _stub_thumbnail_generation(tmp_path)

    with thumb_patch, settings_patch:
        result = generate_thumbnails(project)

    assert result["success"] is True
    payload = result["result"]
    assert len(payload["succeeded"]) == 2
    assert payload["failed"] == []
    assert payload["skipped"] == []
    assert all(c.thumbnail_path and c.thumbnail_path.exists() for c in clips)


def test_generate_thumbnails_skips_existing_files_unless_forced(tmp_path):
    project, clips = _build_project_with_clips(tmp_path)
    existing = tmp_path / "existing.jpg"
    existing.write_bytes(b"old")
    clips[0].thumbnail_path = existing
    thumb_patch, settings_patch = _stub_thumbnail_generation(tmp_path)

    with thumb_patch, settings_patch:
        result = generate_thumbnails(project)

    assert result["success"] is True
    assert result["result"]["skipped"] == [
        {"clip_id": "clip-1", "reason": "already_exists"}
    ]
    assert len(result["result"]["succeeded"]) == 1

    with thumb_patch, settings_patch:
        forced = generate_thumbnails(project, force=True)

    assert forced["success"] is True
    assert len(forced["result"]["succeeded"]) == 2

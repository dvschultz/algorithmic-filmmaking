"""File aliases must reuse source identity without discarding editorial state."""

from pathlib import Path
import subprocess
import sys

import pytest

from core.project import Project
from core.spine.sources import add_source_if_missing, find_source_by_path
from models.clip import Clip, Source


@pytest.mark.parametrize("alias_kind", ["relative", "symlink", "hardlink", "offline"])
def test_import_alias_reuses_source_and_preserves_clips(
    tmp_path, monkeypatch, alias_kind
):
    media = tmp_path / "media.mp4"
    media.write_bytes(b"video")
    alias = tmp_path / "alias.mp4"
    if alias_kind == "symlink":
        alias.symlink_to(media)
    elif alias_kind == "hardlink":
        alias.hardlink_to(media)
    else:
        monkeypatch.chdir(tmp_path)
        alias = Path("media.mp4")
        if alias_kind == "offline":
            media.unlink()
    project = Project.new(name="test")
    original = Source(file_path=media, fps=24)
    project.add_source(original)
    clip = Clip(source_id=original.id, start_frame=0, end_frame=24)
    clip.notes = "keep this edit"
    project.add_clips([clip])
    project.mark_clean()
    reused, added = add_source_if_missing(project, Source(file_path=alias, fps=60))
    assert reused is original and not added
    assert find_source_by_path(project, alias) is original
    assert project.sources == [original] and project.clips == [clip]
    assert original.fps == 24 and clip.notes == "keep this edit"
    assert not project.is_dirty


def test_equal_contents_are_not_identity_and_id_collision_is_rejected(tmp_path):
    project = Project.new(name="test")
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    source, added = add_source_if_missing(project, Source(file_path=first))
    assert added
    assert find_source_by_path(project, second) is None
    with pytest.raises(ValueError, match="different media"):
        add_source_if_missing(project, Source(id=source.id, file_path=second))
    assert len(project.sources) == 1


@pytest.mark.parametrize("failure", [OSError, RuntimeError])
def test_unresolvable_old_source_does_not_block_other_imports(
    tmp_path, monkeypatch, failure
):
    loop = tmp_path / "loop.mp4"
    resolve = Path.resolve

    def resolve_path(path, *args, **kwargs):
        if path == loop:
            raise failure("unavailable source path")
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_path)
    project = Project.new(name="test")
    project.add_source(Source(file_path=loop))
    media = tmp_path / "valid.mp4"
    media.write_bytes(b"video")
    source, added = add_source_if_missing(project, Source(file_path=media))
    assert added and source.file_path == media


def test_all_desktop_import_entry_points_reuse_downloaded_alias(tmp_path):
    code = """
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from core.project import Project
from models.clip import Source
from ui.main_window import MainWindow

root = Path(sys.argv[1])
media = root / 'media.mp4'
media.write_bytes(b'video')
alias = root / 'alias.mp4'
alias.hardlink_to(media)
for name in ('_load_video', '_add_video_to_library', '_on_agent_video_finished', '_on_intention_video_downloaded'):
    project = Project.new(name=name)
    source = Source(file_path=media)
    project.add_source(source)
    window = SimpleNamespace(project=project, sources=project.sources, collect_tab=Mock(),
        status_bar=Mock(), intention_workflow=None, _select_source=Mock(),
        _create_source_with_metadata=Mock(return_value=Source(file_path=alias)),
        _generate_source_thumbnail=Mock(), _update_chat_project_state=Mock())
    result = SimpleNamespace(success=True, file_path=alias, duration=1, fps=30, width=100, height=100)
    if name.startswith('_on_'):
        getattr(MainWindow, name)(window, 'url', result)
    else:
        getattr(MainWindow, name)(window, alias)
    assert project.sources == [source], name
    window._create_source_with_metadata.assert_not_called()
    window.collect_tab.add_source.assert_not_called()
    window._generate_source_thumbnail.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("edited", [False, True])
def test_detection_alias_reuses_source_and_guards_its_clips(
    tmp_path, monkeypatch, edited
):
    from core.operations.detection import DetectionGuard
    from core.spine.detect import detect_scenes_for_video

    media = tmp_path / "media.mp4"
    media.write_bytes(b"video")
    alias = tmp_path / "alias.mp4"
    alias.hardlink_to(media)
    project = Project.new(name="test")
    source = Source(file_path=media)
    project.add_source(source)
    old = Clip(source_id=source.id, start_frame=0, end_frame=60)
    project.add_clips([old])
    assert DetectionGuard.capture(project, alias).source_id == source.id

    def detect(*args, **kwargs):
        if edited:
            old.notes = "changed while detecting"
        return Source(file_path=alias), [
            Clip(source_id="generated", start_frame=0, end_frame=30)
        ]

    monkeypatch.setattr(
        "core.scene_detect.SceneDetector.detect_scenes_with_progress", detect
    )
    monkeypatch.setattr(
        "core.spine.detect._generate_detected_clip_thumbnails", lambda *a, **k: {}
    )
    result = detect_scenes_for_video(project, alias)
    assert result["success"] is not edited
    assert project.sources == [source]
    if edited:
        assert result["error"]["code"] == "stale_detection"
        assert project.clips == [old]
    else:
        assert result["result"]["source_id"] == source.id
        assert project.clips[0].source_id == source.id
        assert project.clips[0].end_frame == 30


def test_folder_import_retries_skip_physical_aliases(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from core.chat_tools import import_folder

    media = tmp_path / "media.mp4"
    media.write_bytes(b"video")
    (tmp_path / "alias.mp4").hardlink_to(media)
    load = Mock(return_value={"duration": 2, "fps": 24, "width": 100, "height": 80})
    monkeypatch.setattr(
        "core.ffmpeg.FFmpegProcessor", Mock(return_value=Mock(get_video_info=load))
    )
    project = Project.new(name="test")
    first = import_folder(project, None, str(tmp_path))
    second = import_folder(project, None, str(tmp_path))
    assert first["imported_count"] == 1 and first["skipped_count"] == 1
    assert second["imported_count"] == 0 and second["skipped_count"] == 2
    assert len(project.sources) == 1 and load.call_count == 1
    assert project.sources[0].fps == 24


def test_metadata_probe_failure_keeps_source_defaults(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from core.spine.sources import probe_source

    monkeypatch.setattr(
        "core.ffmpeg.FFmpegProcessor", Mock(side_effect=RuntimeError("unavailable"))
    )
    path = tmp_path / "media.mp4"
    source = probe_source(path)
    assert source.file_path == path
    assert source.fps == 30 and source.duration_seconds == 0
    assert source.width == source.height == 0

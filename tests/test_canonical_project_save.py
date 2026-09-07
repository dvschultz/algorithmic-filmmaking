"""Save complete model state without borrowing editorial data from disk."""

import json
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from core.project import Project, save_project, ProjectSaveError
from models.clip import Source, Clip
from models.frame import Frame
from models.sequence import Sequence, SequenceClip
from models.audio_source import AudioSource


def populated(root):
    media = root / "video.mp4"
    media.touch()
    source = Source(file_path=media, duration_seconds=1)
    clip = Clip(source_id=source.id, start_frame=0, end_frame=30)
    still = root / "still.png"
    still.touch()
    project = Project.new()
    project.add_source(source)
    project.add_clips([clip])
    project.add_frames([Frame(file_path=still)])
    project.add_audio_source(AudioSource(file_path=root / "offline.wav"))
    project.add_sequence(Sequence(name="Alternate"))
    project.sequences[1].music_path = str(root / "offline.wav")
    return project


def test_transcribe_preserves_complete_project(tmp_path, monkeypatch):
    from cli.main import cli, register_commands

    project = populated(tmp_path)
    path = tmp_path / "project.sceneripper"
    assert project.save(path)
    monkeypatch.setattr("core.transcription.is_faster_whisper_available", lambda: True)
    monkeypatch.setattr("core.transcription.transcribe_clip", lambda **kwargs: [])
    register_commands()
    result = CliRunner().invoke(cli, ["transcribe", str(path)])
    assert result.exit_code == 0, result.output
    loaded = Project.load(path)
    assert [f.id for f in loaded.frames] == [f.id for f in project.frames]
    assert [s.id for s in loaded.sequences] == [s.id for s in project.sequences]
    assert [a.id for a in loaded.audio_sources] == [a.id for a in project.audio_sources]
    assert loaded.sequences[1].music_path == project.sequences[1].music_path


def test_single_sequence_writer_does_not_merge_old_editorial_state(tmp_path):
    project = populated(tmp_path)
    path = tmp_path / "project.sceneripper"
    assert project.save(path)
    replacement = Sequence(name="Replacement")
    assert save_project(path, [], [], replacement)
    data = json.loads(path.read_text())
    assert len(data["sequences"]) == 1
    assert data["sequences"][0]["name"] == "Replacement"


def test_bundle_writer_keeps_all_sequences_and_detects_refused_save(
    tmp_path, monkeypatch
):
    from core.project_export import _write_bundle_project_file

    project = populated(tmp_path)
    path = tmp_path / "bundle.sceneripper"
    _write_bundle_project_file(
        path,
        project,
        project.sources,
        project.frames,
        project.audio_sources,
        project.clips,
    )
    assert len(Project.load(path).sequences) == 2
    monkeypatch.setattr("core.project_export.save_project", Mock(return_value=False))
    with pytest.raises(ProjectSaveError):
        _write_bundle_project_file(path, project, [], [], [], [])


def test_non_active_prerenders_are_localized_without_mutating_model(tmp_path):
    project = populated(tmp_path)
    rendered = tmp_path / "external.mp4"
    rendered.write_bytes(b"prerendered content")
    clip = SequenceClip(out_point=10, prerendered_path=str(rendered))
    project.sequences[1].tracks[0].clips.append(clip)
    path = tmp_path / "saved" / "project.sceneripper"
    assert project.save(path)
    data = json.loads(path.read_text())
    saved = data["sequences"][1]["tracks"][0]["clips"][0]["prerendered_path"]
    assert (
        path.parent / saved
    ).resolve() == path.parent / "transformed_clips" / rendered.name
    assert clip.prerendered_path == str(rendered)


def test_synchronous_save_uses_snapshot_and_keeps_later_edit_dirty(tmp_path):
    project = populated(tmp_path)
    original_name = project.metadata.name
    path = tmp_path / "snapshot.sceneripper"

    def progress(fraction, message):
        if fraction == 0.1:
            project.rename("Newer edit")

    assert project.save(path, progress_callback=progress)
    assert json.loads(path.read_text())["project_name"] == original_name
    assert project.metadata.name == "Newer edit"
    assert project.is_dirty


def test_same_size_prerender_collisions_keep_distinct_content(tmp_path):
    project = populated(tmp_path)
    for index, content in enumerate((b"first", b"other")):
        folder = tmp_path / str(index)
        folder.mkdir()
        rendered = folder / "render.mp4"
        rendered.write_bytes(content)
        project.sequences[index].tracks[0].clips.append(
            SequenceClip(out_point=10, prerendered_path=str(rendered))
        )
    path = tmp_path / "saved" / "project.sceneripper"
    assert project.save(path)
    data = json.loads(path.read_text())
    paths = [
        path.parent / seq["tracks"][0]["clips"][0]["prerendered_path"]
        for seq in data["sequences"]
    ]
    assert paths[0] != paths[1]
    assert [item.read_bytes() for item in paths] == [b"first", b"other"]


@pytest.mark.parametrize("collision_error", [FileExistsError, OSError])
def test_prerender_publication_race_does_not_overwrite_winner(
    tmp_path, monkeypatch, collision_error
):
    import os
    from core.project import _prepare_prerendered_clips

    source = tmp_path / "render.mp4"
    source.write_bytes(b"first")
    sequence = Sequence()
    sequence.tracks[0].clips.append(
        SequenceClip(out_point=10, prerendered_path=str(source))
    )
    original_link = os.link
    won = []

    def racing_link(src, dest):
        if not won:
            dest.write_bytes(b"other")
            won.append(dest)
            raise collision_error("another writer won")
        return original_link(src, dest)

    monkeypatch.setattr("core.project.os.link", racing_link)
    mapping = _prepare_prerendered_clips(sequence, tmp_path / "saved")
    assert won[0].read_bytes() == b"other"
    from pathlib import Path

    assert Path(mapping[str(source)]).read_bytes() == b"first"


def test_prerender_exclusive_copy_fallback(tmp_path, monkeypatch):
    from core.project import _prepare_prerendered_clips
    from pathlib import Path

    source = tmp_path / "render.mp4"
    source.write_bytes(b"complete media")
    sequence = Sequence()
    sequence.tracks[0].clips.append(
        SequenceClip(out_point=10, prerendered_path=str(source))
    )

    def unsupported_link(*args):
        raise OSError("hard links unavailable")

    monkeypatch.setattr("core.project.os.link", unsupported_link)
    mapping = _prepare_prerendered_clips(sequence, tmp_path / "saved")
    assert Path(mapping[str(source)]).read_bytes() == source.read_bytes()


def test_bundle_is_portable_before_atomic_publication(tmp_path, monkeypatch):
    from core.project_export import _write_bundle_project_file
    from core.project_lock import replace_project_file
    from pathlib import Path

    project = populated(tmp_path)
    seen = []

    def inspect(temporary, destination):
        data = json.loads(Path(temporary).read_text())
        assert "_absolute_path" not in data["sources"][0]
        seen.append(True)
        replace_project_file(temporary, destination)

    monkeypatch.setattr("core.project_lock.replace_project_file", inspect)
    _write_bundle_project_file(
        tmp_path / "bundle.sceneripper",
        project,
        project.sources,
        project.frames,
        project.audio_sources,
        project.clips,
    )
    assert seen == [True]


def test_bundle_copies_music_used_only_by_alternate_sequence(tmp_path):
    from core.project_export import export_project_bundle

    project = populated(tmp_path)
    music = tmp_path / "score.wav"
    music.write_bytes(b"music")
    project.sequences[1].music_path = str(music)
    destination = tmp_path / "bundle"
    export_project_bundle(project, destination, include_clips=False)
    loaded = Project.load(next(destination.glob("*.sceneripper")))
    from pathlib import Path

    bundled_music = Path(loaded.sequences[1].music_path)
    assert bundled_music.is_relative_to(destination)
    assert bundled_music.read_bytes() == b"music"
    assert project.sequences[1].music_path == str(music)

"""Schema upgrades preserve originals and never overwrite unknown versions."""

import json

import pytest

from core.project import Project, ProjectMetadata, save_project


def write_document(path, version="1.3"):
    data = {
        "version": version,
        "project_name": "Legacy",
        "sources": [],
        "clips": [],
        "sequence": {"id": "sequence-old", "name": "Original", "tracks": []},
    }
    path.write_text(json.dumps(data))
    return path.read_bytes()


def test_legacy_upgrade_is_in_memory_until_save_and_keeps_backup(tmp_path):
    path = tmp_path / "old.sceneripper"
    original = write_document(path)
    project = Project.load(path)
    assert path.read_bytes() == original
    assert project.metadata.version == "1.4"
    assert project.sequence.id == "sequence-old"
    assert project.save()
    saved = json.loads(path.read_text())
    assert saved["version"] == "1.4"
    assert saved["sequences"][0]["id"] == "sequence-old"
    backups = list(tmp_path.glob("old.sceneripper.pre-v1.4-*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original


def test_newer_schema_can_be_inspected_but_never_saved(tmp_path):
    path = tmp_path / "future.sceneripper"
    original = write_document(path, "99.0")
    project = Project.load(path)
    assert project.is_read_only
    assert project.metadata.name == "Legacy"
    with pytest.raises(RuntimeError, match="read-only"):
        project.rename("changed")
    assert not project.save()
    assert path.read_bytes() == original
    assert not project.save(tmp_path / "copy.sceneripper")
    assert not (tmp_path / "copy.sceneripper").exists()
    project.clear()
    assert not project.is_read_only


def test_save_as_cannot_replace_newer_destination(tmp_path):
    path = tmp_path / "future.sceneripper"
    original = write_document(path, "99.0")
    project = Project.new("Current")
    assert not project.save(path)
    assert path.read_bytes() == original
    assert project.path is None


def test_legacy_writer_cannot_replace_newer_destination(tmp_path):
    path = tmp_path / "future.sceneripper"
    original = write_document(path, "99.0")
    assert not save_project(path, [], [], None, metadata=ProjectMetadata())
    assert path.read_bytes() == original


def test_failed_replacement_preserves_legacy_file_and_backup(tmp_path, monkeypatch):
    import core.project as module

    path = tmp_path / "old.sceneripper"
    original = write_document(path)
    project = Project.load(path)
    replace = module.os.replace

    def fail_project_replace(source, destination):
        if destination == path:
            raise OSError("disk full")
        return replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_project_replace)
    assert not project.save()
    assert path.read_bytes() == original
    assert next(tmp_path.glob("*.bak")).read_bytes() == original


def test_backup_failure_aborts_write_before_replacing_project(tmp_path, monkeypatch):
    import core.project_migrations as migrations

    path = tmp_path / "old.sceneripper"
    original = write_document(path)
    project = Project.load(path)
    monkeypatch.setattr(
        migrations.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk full"))
    )
    assert not project.save()
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.bak"))


def test_backup_is_published_only_after_complete_write(tmp_path, monkeypatch):
    import core.project_migrations as migrations

    path = tmp_path / "old.sceneripper"
    original = write_document(path)
    project = Project.load(path)

    def interrupted_publish(source, destination):
        assert source.read_bytes() == original
        assert not destination.exists()
        raise OSError("interrupted backup publication")

    monkeypatch.setattr(migrations.os, "link", interrupted_publish)
    assert not project.save()
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.bak"))
    assert not list(tmp_path.glob(".project_backup_*"))
    monkeypatch.undo()
    assert project.save()
    assert next(tmp_path.glob("*.bak")).read_bytes() == original


@pytest.mark.parametrize("version", [None, True, "1.x", 99, {}, "0.9"])
def test_invalid_schema_does_not_load_or_overwrite(tmp_path, version):
    from core.project import ProjectLoadError

    path = tmp_path / "invalid.sceneripper"
    original = write_document(path, version)
    with pytest.raises(ProjectLoadError):
        Project.load(path)
    assert not Project.new().save(path)
    assert path.read_bytes() == original


def test_legacy_library_mutations_reject_read_only_project(tmp_path):
    from models.frame import Frame

    path = tmp_path / "future.sceneripper"
    write_document(path, "99.0")
    project = Project.load(path)
    with pytest.raises(RuntimeError, match="read-only"):
        project.add_frames([Frame(id="new")])
    assert project.frames == []


def test_mtime_adapter_reports_refused_save(tmp_path):
    from core.project import ProjectSaveError
    from core.spine.project_io import save_with_mtime_check

    path = tmp_path / "future.sceneripper"
    original = write_document(path, "99.0")
    project = Project.load(path)
    with pytest.raises(ProjectSaveError, match="Failed to save"):
        save_with_mtime_check(project, path, path.stat().st_mtime)
    assert path.read_bytes() == original


def test_migration_preserves_ambiguous_trim_values_and_raw_input():
    from copy import deepcopy
    from core.project_migrations import migrate_project_data

    raw = {"version": "1.3", "sequence": {"clips": [{"in_point": 3, "out_point": 9}]}}
    original = deepcopy(raw)
    upgraded = migrate_project_data(raw)
    assert upgraded["sequences"][0] == original["sequence"]
    assert raw == original
    assert migrate_project_data(upgraded) == upgraded


@pytest.mark.parametrize(
    "bad_fields",
    [{"sequence": "invalid"}, {"sources": 1}, {"clips": None}, {"frames": 1}],
)
def test_malformed_legacy_document_fails_before_migration(tmp_path, bad_fields):
    from core.project import ProjectLoadError

    path = tmp_path / "invalid.sceneripper"
    write_document(path)
    data = json.loads(path.read_text())
    data.update(bad_fields)
    path.write_text(json.dumps(data))
    original = path.read_bytes()
    with pytest.raises(ProjectLoadError, match="structure"):
        Project.load(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("replacement", [None, "missing.mp4", "replacement.mp4"])
def test_offline_multi_sequence_references_survive_relink_and_save(
    tmp_path, replacement
):
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "projects" / "v1.4.json"
    path = tmp_path / "project.sceneripper"
    path.write_bytes(fixture.read_bytes())
    (tmp_path / "replacement.mp4").touch()
    project = Project.load(
        path,
        missing_source_callback=lambda *_: tmp_path / replacement
        if replacement
        else None,
    )
    assert len(project.sources) == len(project.clips) == 1
    assert len(project.sequences) == 2
    assert project.frames[0].id == "frame-a"
    assert project.audio_sources[0].id == "audio-a"
    expected = tmp_path / (
        "replacement.mp4" if replacement == "replacement.mp4" else "video.mp4"
    )
    assert project.sources[0].file_path == expected
    entries = [
        [clip.to_dict() for track in sequence.tracks for clip in track.clips]
        for sequence in project.sequences
    ]
    assert project.save()
    restored = Project.load(path)
    assert restored.sources[0].file_path == expected
    assert [
        [clip.to_dict() for track in sequence.tracks for clip in track.clips]
        for sequence in restored.sequences
    ] == entries

"""Headless sessions reject stale history even when timestamps are unchanged."""

import json
import os

import pytest

from models.sequence import Sequence
from core.project import Project
from core.spine.project_io import load_with_mtime, save_with_mtime_check


def change_file(path):
    timestamp = path.stat()
    data = json.loads(path.read_text())
    data["project_name"] = "external edit"
    path.write_text(json.dumps(data))
    os.utime(path, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))


def load_editable(tmp_path):
    path = tmp_path / "project.sceneripper"
    assert Project.new().save(path)
    project, timestamp = load_with_mtime(path)
    project.add_sequence(Sequence(name="Second"))
    return path, project, timestamp


def test_same_timestamp_external_edit_blocks_save(tmp_path):
    path, project, timestamp = load_editable(tmp_path)
    change_file(path)
    expected = path.read_bytes()
    with pytest.raises(RuntimeError, match="changed externally"):
        save_with_mtime_check(project, path, timestamp)
    assert path.read_bytes() == expected
    assert project.is_dirty


@pytest.mark.parametrize("operation", ["undo", "redo"])
def test_external_edit_blocks_history_and_availability(tmp_path, operation):
    path, project, _ = load_editable(tmp_path)
    if operation == "redo":
        project.session.undo()
    change_file(path)
    before = list(project.sequences)
    assert not getattr(project.session, "can_" + operation)
    with pytest.raises(RuntimeError, match="changed externally"):
        getattr(project.session, operation)()
    assert project.sequences == before


def test_own_save_refreshes_revision_without_losing_history(tmp_path):
    path, project, timestamp = load_editable(tmp_path)
    save_with_mtime_check(project, path, timestamp)
    assert project.session.can_undo
    project.session.undo()
    assert len(project.sequences) == 1


def test_deleted_project_blocks_history(tmp_path):
    path, project, _ = load_editable(tmp_path)
    path.unlink()
    assert not project.session.can_undo
    with pytest.raises(RuntimeError, match="changed externally"):
        project.session.undo()


def test_change_during_load_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "project.sceneripper"
    assert Project.new().save(path)
    original = Project.load

    def racing_load(path):
        result = original(path)
        change_file(path)
        return result

    monkeypatch.setattr(Project, "load", racing_load)
    with pytest.raises(RuntimeError, match="changed externally"):
        load_with_mtime(path)


def test_detected_conflict_stays_invalid_after_bytes_restored(tmp_path):
    path, project, _ = load_editable(tmp_path)
    original = path.read_bytes()
    change_file(path)
    assert not project.session.can_undo
    path.write_bytes(original)
    assert not project.session.can_undo
    project.clear()
    assert project.session.file_revision is None
    project.add_sequence(Sequence(name="Fresh"))
    assert project.session.can_undo


def test_revision_conflict_is_structured_for_mcp(tmp_path):
    from core.project_revision import ProjectRevisionConflict
    from core.spine.project_io import project_error

    error = project_error(ProjectRevisionConflict(tmp_path / "p.sceneripper"))
    assert error["code"] == "project_modified_externally"
    assert "Reload" in error["message"]


def test_history_rechecks_revision_after_availability_query(tmp_path):
    path, project, _ = load_editable(tmp_path)
    assert project.session.can_undo
    change_file(path)
    with pytest.raises(RuntimeError, match="changed externally"):
        project.session.undo()
    assert len(project.sequences) == 2


def test_failed_save_keeps_revision_and_history(tmp_path, monkeypatch):
    from core.project import ProjectSaveError

    path, project, timestamp = load_editable(tmp_path)
    original = project.session.file_revision
    monkeypatch.setattr(project, "save", lambda path: False)
    with pytest.raises(ProjectSaveError):
        save_with_mtime_check(project, path, timestamp)
    assert project.session.file_revision is original
    assert project.session.can_undo


def test_own_save_publishes_refreshed_history_availability(tmp_path):
    path, project, timestamp = load_editable(tmp_path)
    availability = []
    project.session.add_observer(lambda: availability.append(project.session.can_undo))
    save_with_mtime_check(project, path, timestamp)
    assert availability[-1] is True

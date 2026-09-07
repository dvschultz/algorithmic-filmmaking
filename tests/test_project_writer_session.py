"""Persistent model ownership and Save As transitions."""

import pytest
from core.project import Project, ProjectLoadError
from core.project_lock import ProjectWriter, ProjectBusyError


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr("core.project_lock._lock_directory", lambda: tmp_path / "locks")
    return tmp_path


def busy(path):
    with pytest.raises(ProjectBusyError):
        with ProjectWriter(path):
            pass


def free(path):
    with ProjectWriter(path):
        pass


def test_owned_load_and_clear(root):
    path = root / "owned.sceneripper"
    assert Project.new().save(path)
    project = Project.load(path, retain_writer=True)
    try:
        busy(path)
        assert project.save()
        busy(path)
        project.clear()
        free(path)
        assert project.save(path)
        busy(path)
    finally:
        project.close_writer()


def test_failed_save_as_keeps_old_owner(root, monkeypatch):
    old, new = root / "old.sceneripper", root / "new.sceneripper"
    project = Project.new(retain_writer=True)
    try:
        assert project.save(old)
        project.rename("Unsaved")
        with ProjectWriter(new):
            with pytest.raises(ProjectBusyError):
                project.save(new)
        assert project.path == old and project.is_dirty
        busy(old)
        monkeypatch.setattr("core.project.save_project", lambda **kwargs: False)
        assert not project.save(new)
        busy(old)
        free(new)
        assert project.path == old and project.is_dirty
    finally:
        project.close_writer()


def test_successful_save_as_transfers_only_after_publication(root):
    old, new = root / "old.sceneripper", root / "new.sceneripper"
    project = Project.new(retain_writer=True)
    try:
        assert project.save(old)
        writer = project.prepare_save(new)
        busy(old)
        busy(new)
        with pytest.raises(RuntimeError):
            project.clear()
        project.finish_save(writer, True)
        busy(new)
        free(old)
    finally:
        project.close_writer()


def test_owned_load_failure_releases_lease(root):
    path = root / "bad.sceneripper"
    path.write_text("invalid")
    with pytest.raises(ProjectLoadError):
        Project.load(path, retain_writer=True)
    free(path)


def test_agent_load_acquires_before_replacing_departing_project(root):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from models.clip import Source
    from core.chat_tools import load_project

    old, target = root / "old.sceneripper", root / "target.sceneripper"
    source_file = root / "video.mp4"
    source_file.touch()
    saved = Project.new()
    saved.add_source(Source(file_path=source_file))
    assert saved.save(target)
    departing = Project.new(retain_writer=True)
    assert departing.save(old)

    def clear():
        busy(old)
        busy(target)
        departing.clear()

    window = SimpleNamespace(
        project=departing,
        _clear_project_state=clear,
        undo_stack=Mock(),
        _project_adapter=Mock(),
        _refresh_ui_from_project=lambda: None,
    )
    try:
        result = load_project(str(target), main_window=window)
        assert result["success"], result
        busy(target)
        free(old)
        assert window.project.path == target
    finally:
        window.project.close_writer()


def test_agent_save_uses_retained_lease_and_reports_busy_destination(root):
    from core.spine.project_save import save_project

    old, target = root / "old.sceneripper", root / "target.sceneripper"
    project = Project.new(retain_writer=True)
    try:
        assert project.save(old)
        assert save_project(project)["success"]
        with ProjectWriter(target):
            result = save_project(project, str(target))
        assert result["error"]["code"] == "project_busy"
        assert project.path == old
        busy(old)
    finally:
        project.close_writer()


def test_pending_save_blocks_clean_project_departure(root):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from ui.main_window import MainWindow
    from core.chat_tools import new_project, load_project

    path = root / "project.sceneripper"
    project = Project.new(retain_writer=True)
    assert project.save(path)
    writer = project.prepare_save(path)
    window = SimpleNamespace(
        project=project, _save_project_context={}, status_bar=Mock(), _is_dirty=False
    )
    try:
        assert not MainWindow._check_unsaved_changes(window)
        assert not new_project(main_window=window)["success"]
        assert not load_project(str(path), main_window=window)["success"]
        busy(path)
    finally:
        project.finish_save(writer, False)
        project.close_writer()


def test_owned_save_records_canonical_path(root, monkeypatch):
    from pathlib import Path

    monkeypatch.chdir(root)
    project = Project.new(retain_writer=True)
    try:
        assert project.save(Path("relative.sceneripper"))
        assert project.path == root / "relative.sceneripper"
        elsewhere = root / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        assert project.save()
        assert not (elsewhere / "relative.sceneripper").exists()
    finally:
        project.close_writer()


def test_desktop_new_and_accepted_close_release_ownership(root, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from ui.main_window import MainWindow

    path = root / "desktop.sceneripper"
    project = Project.new(retain_writer=True)
    assert project.save(path)
    window = SimpleNamespace(
        project=project,
        _check_unsaved_changes=lambda: True,
        _clear_project_state=project.clear,
        _mark_clean=project.mark_clean,
        _update_window_title=lambda: None,
        status_bar=Mock(),
    )
    MainWindow._on_new_project(window)
    free(path)
    assert project.save(path)
    busy(path)
    monkeypatch.setenv("SCENE_RIPPER_STARTUP_SMOKE_TEST", "1")
    event = Mock()
    MainWindow.closeEvent(window, event)
    event.accept.assert_called_once()
    free(path)


def test_desktop_busy_open_keeps_current_project(root, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from ui.main_window import MainWindow, QMessageBox

    old, target = root / "old.sceneripper", root / "busy.sceneripper"
    project = Project.new(retain_writer=True)
    assert project.save(old)
    window = SimpleNamespace(project=project, status_bar=Mock())
    warning = Mock()
    monkeypatch.setattr(QMessageBox, "warning", warning)
    try:
        with ProjectWriter(target):
            MainWindow._load_project_file(window, target)
        assert window.project is project and project.path == old
        busy(old)
        assert "already open for writing" in warning.call_args.args[2]
    finally:
        project.close_writer()

"""Exercise real Qt action delivery and both GUI/agent history entry points."""

from types import SimpleNamespace

import pytest

from core.chat_tools import redo, toggle_clip_disabled, undo
from tests.test_clip_disabled import _make_project_with_clips
from ui.session_history import SessionHistoryAdapter


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_menu_and_agent_share_history_and_saved_marker(qapp):
    from ui.main_window import MainWindow

    project = _make_project_with_clips()
    adapter = SessionHistoryAdapter(project.session)
    undo_action = adapter.createUndoAction(adapter, "Undo")
    redo_action = adapter.createRedoAction(adapter, "Redo")

    class Window:
        _title_build_suffix = "test"
        current_project_path = None
        current_source = None
        _update_window_title = MainWindow._update_window_title
        _is_dirty = property(lambda self: self.project.is_dirty)

        def setWindowTitle(self, title):
            self.title = title

    window = Window()
    window.project = project
    window.undo_stack = adapter
    adapter.changed.connect(window._update_window_title)
    window._update_window_title()
    assert not undo_action.isEnabled()
    assert not redo_action.isEnabled()
    project.toggle_clips_disabled(["c0"])
    assert window.title.endswith("*")
    assert undo_action.text() == "Undo Disable 1 clip"
    assert toggle_clip_disabled(project, ["c1"], True)["success"]
    undo_action.trigger()
    assert not project.clips[1].disabled
    assert undo(window)["success"]
    assert not project.is_dirty
    assert not window.title.endswith("*")
    assert redo(window)["success"]
    assert project.clips[0].disabled
    redo_action.trigger()
    assert project.clips[1].disabled
    project.mark_clean()
    assert not window.title.endswith("*")


def test_switch_project_disconnects_old_history(qapp):
    first = _make_project_with_clips()
    second = _make_project_with_clips()
    adapter = SessionHistoryAdapter(first.session)
    action = adapter.createUndoAction(adapter)
    first.toggle_clips_disabled(["c0"])
    assert action.isEnabled()
    first.session.close()
    adapter.set_session(second.session)
    assert not action.isEnabled()
    assert not undo(SimpleNamespace(undo_stack=adapter))["success"]
    assert first.clips[0].disabled
    assert not second.clips[0].disabled


def test_conflicting_undo_reports_failure_to_agent_and_ui(qapp):
    project = _make_project_with_clips()
    adapter = SessionHistoryAdapter(project.session)
    errors = []
    adapter.action_failed.connect(errors.append)
    project.toggle_clips_disabled(["c0"])
    project.remove_clips(["c0"])
    result = undo(SimpleNamespace(undo_stack=adapter))
    assert not result["success"]
    assert errors == [result["error"]]

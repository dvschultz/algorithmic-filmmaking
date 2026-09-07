"""Future-schema inspection must not offer or dispatch project edits."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from core.project import Project
from core.chat_tools import ToolRegistry
from core.tool_executor import ToolExecutor


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_read_only_executor_blocks_mutation_before_worker_wait():
    registry = ToolRegistry()
    called = Mock()

    @registry.register(
        description="Edit",
        requires_project=True,
        modifies_project_state=True,
        conflicts_with_workers=True,
    )
    def edit(project):
        called()

    project = Project.new()
    project.metadata.version = "99.0"
    busy = Mock(return_value=None)
    result = ToolExecutor(registry, project, busy).execute(
        {"function": {"name": "edit", "arguments": "{}"}}
    )
    assert not result["success"]
    assert "read-only" in result["error"]
    called.assert_not_called()
    busy.assert_not_called()


def test_read_only_controls_resist_reenable_and_restore(qapp):
    from PySide6.QtWidgets import QWidget, QPushButton
    from PySide6.QtGui import QAction
    from ui.project_access import ReadOnlyControls

    parent = QWidget()
    button = QPushButton(parent)
    action = QAction(parent)
    guard = ReadOnlyControls(parent)
    guard.watch(button)
    guard.watch(action)
    guard.set_read_only(True)
    button.setEnabled(True)
    action.setEnabled(True)
    assert not button.isEnabled() and not action.isEnabled()
    guard.set_read_only(False)
    assert button.isEnabled() and action.isEnabled()
    parent.close()


def test_read_only_inspection_tool_remains_available():
    registry = ToolRegistry()

    @registry.register(description="Inspect", requires_project=True)
    def inspect_project(project):
        return {"version": project.metadata.version}

    project = Project.new()
    project.metadata.version = "99.0"
    result = ToolExecutor(registry, project).execute(
        {"function": {"name": "inspect_project", "arguments": "{}"}}
    )
    assert result["success"]


def test_direct_save_handler_refuses_read_only_before_snapshot(
    qapp, tmp_path, monkeypatch
):
    from ui.main_window import MainWindow, QMessageBox

    project = Project.new()
    project.metadata.version = "99.0"
    window = SimpleNamespace(
        project=project,
        status_bar=Mock(),
        save_worker=None,
        sequence_tab=Mock(),
        analyze_tab=Mock(),
    )
    monkeypatch.setattr(QMessageBox, "warning", Mock())
    monkeypatch.setattr(
        project,
        "snapshot_for_save",
        Mock(side_effect=AssertionError("snapshot read-only project")),
    )
    MainWindow._save_project_to_file(window, tmp_path / "no.sceneripper")
    project.snapshot_for_save.assert_not_called()
    window.sequence_tab._persist_current_sequence.assert_not_called()


def test_read_only_guard_restores_nested_control_state(qapp):
    from PySide6.QtWidgets import QWidget, QPushButton
    from ui.project_access import ReadOnlyControls

    parent = QWidget()
    group = QWidget(parent)
    button = QPushButton(group)
    guard = ReadOnlyControls(parent)
    guard.watch(group)
    guard.watch(button)
    guard.set_read_only(True)
    guard.set_read_only(False)
    assert group.isEnabled() and button.isEnabled()
    parent.close()


def test_gui_agent_dispatch_rejects_read_only_before_call(monkeypatch):
    from core.chat_tools import tools
    from ui.main_window import MainWindow

    project = Project.new()
    project.metadata.version = "99.0"
    function = Mock()
    tool = SimpleNamespace(
        name="edit", modifies_project_state=True, modifies_gui_state=True, func=function
    )
    monkeypatch.setattr(tools, "get", lambda name: tool)
    window = SimpleNamespace(project=project, _chat_worker=Mock())
    MainWindow._on_gui_tool_requested(window, "edit", {}, "call")
    function.assert_not_called()
    result = window._chat_worker.set_gui_tool_result.call_args.args[0]
    assert not result["success"] and "read-only" in result["error"]


@pytest.mark.parametrize("name", ["save_project", "export_bundle"])
def test_read_only_dispatch_rejects_project_writers_even_without_mutation_flag(name):
    from core.project_access import read_only_tool_error

    project = Project.new()
    project.metadata.version = "99.0"
    assert read_only_tool_error(
        project, SimpleNamespace(name=name, modifies_project_state=False)
    )


@pytest.mark.parametrize("name", ["update_source", "undo", "redo", "download_videos"])
def test_editorial_agent_tools_declare_mutation(name):
    from core.chat_tools import tools
    from core.project_access import read_only_tool_error

    project = Project.new()
    project.metadata.version = "99.0"
    assert read_only_tool_error(project, tools.get(name))


def test_inspection_filter_has_public_tool_registration():
    from core.chat_tools import tools, apply_filters

    assert tools.get("apply_filters") is not None
    assert tools.get("apply_filters").func is apply_filters
    assert tools.get("_validate_enum_arg") is None


def test_read_only_browser_delete_gestures_are_ignored(qapp):
    from PySide6.QtWidgets import QWidget
    from ui.clip_browser import ClipBrowser
    from ui.source_browser import SourceBrowser

    parent = QWidget()
    parent.setProperty("projectReadOnly", True)
    clips = ClipBrowser()
    sources = SourceBrowser()
    clips.setParent(parent)
    sources.setParent(parent)
    clips._all_entries = Mock(side_effect=AssertionError("edited clips"))
    removed = Mock()
    sources.delete_sources_requested.connect(removed)
    clips.toggle_disabled(["missing"])
    sources._on_thumbnail_delete_requested(Mock())
    removed.assert_not_called()
    assert clips.isEnabled() and sources.isEnabled()
    parent.close()


def test_read_only_import_shortcuts_restore_on_new_project(qapp):
    from PySide6.QtWidgets import QWidget
    from PySide6.QtGui import QAction
    from ui.project_access import refresh_project_access

    window = QWidget()
    window.project = Project.new()
    action = QAction(window)
    window._project_import_actions = (action,)
    window.project.metadata.version = "99.0"
    refresh_project_access(window)
    action.setEnabled(True)
    assert not action.isEnabled()
    window.project = Project.new()
    refresh_project_access(window)
    assert action.isEnabled()
    window.close()

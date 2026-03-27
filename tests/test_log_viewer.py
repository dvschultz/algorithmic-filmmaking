"""Tests for the in-app log viewer bridge and widget."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from ui.log_viewer import InAppLogBridge, LogViewerWidget
from ui.main_window import MainWindow
from ui.widgets.dependency_widgets import DependencyDownloadDialog


def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_log_bridge_buffers_and_emits_recent_lines():
    bridge = InAppLogBridge(max_lines=2)
    logger_obj = logging.getLogger("test.in_app_logs")
    logger_obj.handlers = []
    logger_obj.propagate = False
    logger_obj.setLevel(logging.DEBUG)
    bridge.install(logger_obj)

    seen: list[str] = []
    bridge.line_emitted.connect(seen.append)

    logger_obj.info("first line")
    logger_obj.warning("second line")
    logger_obj.error("third line")

    assert len(seen) == 3
    assert "first line" in seen[0]
    assert "second line" in seen[1]
    assert "third line" in seen[2]
    recent = bridge.recent_lines()
    assert len(recent) == 2
    assert "second line" in recent[0]
    assert "third line" in recent[1]


def test_log_viewer_widget_shows_existing_and_live_logs():
    qapp()
    bridge = InAppLogBridge(max_lines=10)
    bridge.append_line("existing line")

    widget = LogViewerWidget(bridge)
    assert "existing line" in widget.text_edit.toPlainText()

    bridge.append_line("live line")
    assert "live line" in widget.text_edit.toPlainText()


def test_show_log_viewer_reveals_dock():
    calls = []

    class _FakeDock:
        def setVisible(self, visible: bool):
            calls.append(("visible", visible))

        def raise_(self):
            calls.append(("raise", True))

        def activateWindow(self):
            calls.append(("activate", True))

    harness = SimpleNamespace(log_dock=_FakeDock())

    MainWindow.show_log_viewer(harness)

    assert calls == [("visible", True), ("raise", True), ("activate", True)]


def test_dependency_download_dialog_adds_view_logs_button_for_parent_host():
    qapp()

    class _Host(QWidget):
        def __init__(self):
            super().__init__()
            self.opened = 0

        def show_log_viewer(self):
            self.opened += 1

    host = _Host()
    dialog = DependencyDownloadDialog(
        title="Install",
        message="Installing package",
        install_func=lambda _cb: True,
        parent=host,
    )

    buttons = [btn for btn in dialog.findChildren(QPushButton) if btn.text() == "View Logs"]
    assert len(buttons) == 1

    buttons[0].click()
    assert host.opened == 1

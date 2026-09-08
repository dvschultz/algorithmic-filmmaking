"""Verify GUI test cleanup without importing Qt into headless collections."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("gui", [False, True])
def test_test_widget_ownership(tmp_path, gui):
    source = '''
import sys
import pytest
pytest_plugins = ["tests.conftest"]
def test_headless():
    assert "PySide6.QtWidgets" not in sys.modules
    assert "PySide6.QtCore" not in sys.modules
'''
    if gui:
        source = '''
import pytest
import shiboken6
from PySide6.QtWidgets import QWidget, QMenu
pytest_plugins = ["tests.conftest"]
retained = []
deferred = []
class ClosingLater(QWidget):
    def closeEvent(self, event):
        event.ignore()
@pytest.fixture(scope="module")
def shared_widget(qt_application_lifetime):
    widget = QWidget()
    yield widget
    widget.deleteLater()
def test_create(shared_widget):
    widget = QWidget()
    child = QMenu(widget)
    shared_widget.popup = QMenu(shared_widget)
    retained.extend([widget, child])
    deferred.append(ClosingLater())
def test_cleanup(shared_widget):
    assert all(not shiboken6.isValid(widget) for widget in retained)
    assert shiboken6.isValid(shared_widget)
    assert shiboken6.isValid(shared_widget.popup)
    assert shiboken6.isValid(deferred[0])
    deferred[0].deleteLater()
'''
    (tmp_path / "test_child.py").write_text(source)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "test_child.py", "-q"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "QT_QPA_PLATFORM": "offscreen",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

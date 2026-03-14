"""Regression tests for SequenceTab direction dropdown synchronization."""

import os

import pytest

# SequenceTab contains video widgets; offscreen avoids display-dependent failures in CI/headless.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_color_direction_dropdown_reflects_selected_direction(qapp):
    from ui.tabs.sequence_tab import SequenceTab

    tab = SequenceTab()
    tab.algorithm_dropdown.setCurrentText("Chromatics")
    tab._update_direction_dropdown("color", "warm_to_cool")

    assert tab.direction_dropdown.currentText() == "Warm to Cool"
    assert tab._get_current_direction() == "warm_to_cool"


def test_color_direction_dropdown_defaults_to_first_option(qapp):
    from ui.tabs.sequence_tab import SequenceTab

    tab = SequenceTab()
    tab.algorithm_dropdown.setCurrentText("Chromatics")
    tab._update_direction_dropdown("color", "unknown_direction")

    assert tab.direction_dropdown.currentText() == "Rainbow"
    assert tab._get_current_direction() == "rainbow"

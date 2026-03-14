"""Regression tests for clip-browser layout refresh on tab activation."""

import os

import pytest

# Tabs include Qt widgets; offscreen avoids display-dependent failures.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_cut_tab_refreshes_clip_browser_layout_on_activation(qapp):
    from ui.tabs.cut_tab import CutTab

    tab = CutTab()
    tab.state_stack.setCurrentIndex(tab.STATE_CLIPS)
    tab.clip_browser.thumbnails = [object()]

    called = {"value": False}

    def _refresh():
        called["value"] = True

    tab.clip_browser.refresh_layout = _refresh

    tab.on_tab_activated()

    assert called["value"] is True


def test_analyze_tab_refreshes_clip_browser_layout_on_activation(qapp):
    from ui.tabs.analyze_tab import AnalyzeTab

    tab = AnalyzeTab()
    tab.state_stack.setCurrentIndex(tab.STATE_CLIPS)
    tab.clip_browser.thumbnails = [object()]

    called = {"value": False}

    def _refresh():
        called["value"] = True

    tab.clip_browser.refresh_layout = _refresh

    tab.on_tab_activated()

    assert called["value"] is True


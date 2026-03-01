"""Tests for Analyze tab quick-run operation availability."""

from pathlib import Path

import pytest

from models.clip import Source
from tests.conftest import make_test_clip


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def analyze_tab(qapp):
    from ui.tabs.analyze_tab import AnalyzeTab

    return AnalyzeTab()


@pytest.fixture
def source():
    return Source(
        id="src-1",
        file_path=Path("/test/video.mp4"),
        duration_seconds=10.0,
        fps=30.0,
        width=1280,
        height=720,
    )


def test_quick_run_disables_completed_operation_items(analyze_tab, source):
    clips = [
        make_test_clip("c1", dominant_colors=[(1, 2, 3)]),
        make_test_clip("c2", dominant_colors=[(4, 5, 6)]),
    ]
    clips_by_id = {c.id: c for c in clips}
    analyze_tab.set_lookups(clips_by_id, {source.id: source})
    analyze_tab.add_clips([c.id for c in clips])

    colors_idx = analyze_tab.quick_run_combo.findData("colors")
    colors_item = analyze_tab.quick_run_combo.model().item(colors_idx)

    assert colors_idx >= 0
    assert colors_item.isEnabled() is False
    assert analyze_tab.quick_run_combo.isEnabled() is True
    assert analyze_tab.quick_run_btn.isEnabled() is True


def test_quick_run_disabled_when_no_operation_needed(analyze_tab, source, monkeypatch):
    clip = make_test_clip("c1")
    monkeypatch.setattr(
        "ui.tabs.analyze_tab.compute_disabled_operations",
        lambda clips, op_keys: set(op_keys),
    )

    analyze_tab.set_lookups({clip.id: clip}, {source.id: source})
    analyze_tab.add_clips([clip.id])

    assert analyze_tab.quick_run_combo.isEnabled() is False
    assert analyze_tab.quick_run_btn.isEnabled() is False

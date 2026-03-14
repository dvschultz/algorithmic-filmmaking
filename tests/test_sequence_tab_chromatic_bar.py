"""Tests for Chromatics color-bar controls in SequenceTab."""

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


def test_confirm_view_shows_color_bar_option_for_chromatic_flow(qapp):
    from ui.tabs.sequence_tab import SequenceTab

    tab = SequenceTab()

    tab._show_confirm_view("color", [], [])
    assert tab._confirm_chromatic_bar_checkbox.isHidden() is False

    tab._show_confirm_view("duration", [], [])
    assert tab._confirm_chromatic_bar_checkbox.isHidden() is True


def test_should_show_chromatic_bar_only_for_color_algorithm(qapp):
    from ui.tabs.sequence_tab import SequenceTab

    tab = SequenceTab()
    tab._set_state(tab.STATE_TIMELINE)
    tab.algorithm_dropdown.setCurrentText("Chromatics")
    tab.set_chromatic_color_bar_enabled(True, emit_signal=False)
    tab._update_chromatic_bar_controls("color")

    assert tab.should_show_chromatic_color_bar() is True
    assert tab.chromatic_bar_checkbox.isHidden() is False

    tab.algorithm_dropdown.setCurrentText("Hatchet Job")
    tab._update_chromatic_bar_controls("shuffle")

    assert tab.should_show_chromatic_color_bar() is False
    assert tab.chromatic_bar_checkbox.isHidden() is True


def test_card_click_chromatic_flow_still_shows_confirm_when_estimates_empty(qapp, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from models.clip import Clip, Source
    from ui.tabs.sequence_tab import SequenceTab

    monkeypatch.setattr("ui.tabs.sequence_tab.estimate_sequence_cost", lambda algorithm, clips: [])

    tab = SequenceTab()
    source = Source(id="src-1", file_path=Path("/tmp/test.mp4"), fps=24.0)
    clip = Clip(id="clip-1", source_id=source.id, start_frame=0, end_frame=24, dominant_colors=[(255, 0, 0)])
    tab._available_clips = [(clip, source)]
    tab._clips = [clip]
    tab._gui_state = SimpleNamespace(analyze_selected_ids=[clip.id], cut_selected_ids=[])

    tab._on_card_clicked("color")

    assert tab._current_state == tab.STATE_CONFIRM
    assert tab._pending_algorithm == "color"
    assert len(tab._pending_clips) == 1

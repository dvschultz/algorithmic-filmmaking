"""Tests for Chromatics color-bar controls in SequenceTab."""

import os

import pytest
from core.cost_estimates import OperationEstimate

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


def test_card_click_resolves_selected_clips_when_available_cache_is_empty(qapp, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from models.clip import Clip, Source
    from ui.tabs.sequence_tab import SequenceTab

    tab = SequenceTab()
    source = Source(id="src-1", file_path=Path("/tmp/test.mp4"), fps=24.0)
    clip = Clip(id="clip-1", source_id=source.id, start_frame=0, end_frame=24)
    tab.set_available_clips([], all_clips=[clip], sources_by_id={source.id: source})
    tab._gui_state = SimpleNamespace(analyze_selected_ids=[], cut_selected_ids=[clip.id])

    dialog_calls = []
    warnings = []

    monkeypatch.setattr(
        tab,
        "_show_dice_roll_dialog",
        lambda clips: dialog_calls.append(clips),
    )
    monkeypatch.setattr(
        "ui.tabs.sequence_tab.QMessageBox.warning",
        lambda *args: warnings.append(args),
    )

    tab._on_card_clicked("shuffle")

    assert warnings == []
    assert dialog_calls == [[(clip, source)]]


def test_match_cut_card_disabled_when_boundary_embeddings_missing_and_runtime_unavailable(qapp, monkeypatch):
    from pathlib import Path

    from models.clip import Clip, Source
    from ui.tabs.sequence_tab import SequenceTab

    tab = SequenceTab()
    source = Source(id="src-1", file_path=Path("/tmp/test.mp4"), fps=24.0)
    clip = Clip(id="clip-1", source_id=source.id, start_frame=0, end_frame=24)

    monkeypatch.setattr("ui.tabs.sequence_tab.load_settings", lambda: object())
    monkeypatch.setattr("ui.tabs.sequence_tab.check_feature_ready", lambda _name: (False, ["package:torch"]))

    tab.set_available_clips([(clip, source)], all_clips=[clip], sources_by_id={source.id: source})
    tab._update_card_availability()

    assert tab.card_grid._cards["match_cut"].is_enabled() is False
    assert "Install embeddings dependencies" in tab.card_grid._cards["match_cut"].toolTip()


def test_confirm_view_disables_generate_when_sequence_dependencies_missing(qapp, monkeypatch):
    from pathlib import Path

    from models.clip import Clip, Source
    from ui.tabs.sequence_tab import SequenceTab

    tab = SequenceTab()
    source = Source(id="src-1", file_path=Path("/tmp/test.mp4"), fps=24.0)
    clip = Clip(id="clip-1", source_id=source.id, start_frame=0, end_frame=24)
    estimates = [
        OperationEstimate(
            operation="boundary_embeddings",
            label="Boundary Embeddings",
            clips_needing=1,
            clips_total=1,
            tier="local",
            time_seconds=1.5,
            cost_dollars=0.0,
        )
    ]

    monkeypatch.setattr("ui.tabs.sequence_tab.load_settings", lambda: object())
    monkeypatch.setattr("ui.tabs.sequence_tab.check_feature_ready", lambda _name: (False, ["package:torch"]))

    tab._show_confirm_view("match_cut", [(clip, source)], estimates)

    assert tab._confirm_generate_btn.isEnabled() is False
    assert "Boundary Embeddings require local dependencies" in tab._confirm_cost_panel._warning_label.text()

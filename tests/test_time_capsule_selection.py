"""Regression tests for issue #150: Time Capsule uses wrong clips / wrong order.

A. ``sequential`` orders clips chronologically by source timeline, not by the
   clip browser's display order (e.g. sorted by duration).
B. ``GUIState.sequencing_selected_ids()`` follows the most recently set tab
   selection, so a fresh filtered Cut selection beats a stale Analyze one.
"""

from pathlib import Path

import pytest

from core.gui_state import GUIState
from core.project import Project
from core.remix import run_registry_algorithm
from models.clip import Clip, Source

# (id, source, start, end) -- durations deliberately unrelated to timeline order.
_SPECS = [
    ("a1", "sA", 0, 90),
    ("a2", "sA", 90, 100),
    ("a3", "sA", 100, 160),
    ("a4", "sA", 160, 165),
    ("b1", "sB", 0, 40),
    ("b2", "sB", 40, 200),
    ("b3", "sB", 200, 210),
]
_CHRONO = ["a1", "a2", "a3", "a4", "b1", "b2", "b3"]


def _project() -> Project:
    project = Project()
    for sid in ("sA", "sB"):
        project.add_source(Source(id=sid, file_path=Path(f"/tmp/{sid}.mp4"), fps=30.0, duration_seconds=60))
    project.add_clips([Clip(id=cid, source_id=sid, start_frame=s, end_frame=e) for cid, sid, s, e in _SPECS])
    project.mark_clean()
    return project


def _pairs(project: Project, order: list[str]):
    by_id = project.clips_by_id
    return [(by_id[cid], project.sources_by_id[by_id[cid].source_id]) for cid in order]


def _run_ids(pairs) -> list[str]:
    run = run_registry_algorithm("sequential", pairs)
    return [entry.clip_id for entry in run.recipe.realized]


# --- A. chronological ordering ------------------------------------------------


def test_sequential_orders_duration_sorted_inputs_chronologically():
    project = _project()
    by_duration = sorted(_CHRONO, key=lambda cid: project.clips_by_id[cid].end_frame - project.clips_by_id[cid].start_frame)
    assert by_duration != _CHRONO
    result = _run_ids(_pairs(project, by_duration))
    assert result == _CHRONO
    assert sorted(result) == sorted(by_duration)


def test_sequential_is_independent_of_input_shuffle_when_source_order_matches():
    project = _project()
    first = ["a3", "b2", "a1", "b3", "a4", "b1", "a2"]
    second = ["a4", "a2", "b1", "b3", "a1", "b2", "a3"]
    assert _run_ids(_pairs(project, first)) == _run_ids(_pairs(project, second)) == _CHRONO


def test_sequential_groups_sources_by_first_appearance_and_keeps_only_inputs():
    project = _project()
    subset = ["b3", "a3", "b1", "a1"]
    result = _run_ids(_pairs(project, subset))
    assert result == ["b1", "b3", "a1", "a3"]
    assert set(result) == set(subset) and len(result) == len(subset)


# --- B. GUIState selection precedence ------------------------------------------


def test_fresh_cut_selection_beats_stale_analyze_selection():
    state = GUIState()
    state.set_tab_selection("analyze", ["a1", "a2"])
    state.set_tab_selection("cut", ["c1"])
    assert state.sequencing_selected_ids() == ["c1"]
    assert state.cut_selected_ids == ["c1"]
    assert state.analyze_selected_ids == ["a1", "a2"]
    assert state.selected_clip_ids == ["c1"]

    state.set_tab_selection("analyze", ["a3"])
    assert state.sequencing_selected_ids() == ["a3"]
    assert state.selected_clip_ids == ["a3"]


def test_fresh_empty_cut_selection_does_not_fall_back_to_stale_analyze():
    state = GUIState()
    state.set_tab_selection("analyze", ["a1", "a2"])
    state.set_tab_selection("cut", [])
    assert state.sequencing_selected_ids() == []


def test_direct_assignment_falls_back_to_legacy_precedence():
    state = GUIState()
    state.cut_selected_ids = ["c1"]
    assert state.sequencing_selected_ids() == ["c1"]
    state.analyze_selected_ids = ["a1"]
    assert state.sequencing_selected_ids() == ["a1"]


def test_clear_resets_most_recent_selection_marker():
    state = GUIState()
    state.set_tab_selection("cut", ["c1"])
    state.clear()
    assert state.sequencing_selected_ids() == []
    state.analyze_selected_ids = ["a1"]
    state.cut_selected_ids = ["c2"]
    assert state.sequencing_selected_ids() == ["a1"]


# --- UI: Sequence tab card click -----------------------------------------------


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_time_capsule_card_uses_fresh_cut_subset_not_stale_analyze(qapp, monkeypatch):
    from ui.tabs.sequence_tab import SequenceTab

    project = _project()
    tab = SequenceTab()
    tab.set_project(project)
    tab.set_available_clips(_pairs(project, _CHRONO))
    state = GUIState()
    state.set_tab_selection("analyze", list(_CHRONO))
    state.set_tab_selection("cut", ["b2", "a3"])
    tab.set_gui_state(state)

    calls = []
    monkeypatch.setattr(tab, "_apply_algorithm", lambda algorithm, clips, *a, **kw: calls.append((algorithm, clips)))
    tab._on_card_clicked("sequential")

    assert len(calls) == 1
    algorithm, clips = calls[0]
    assert algorithm == "sequential"
    assert sorted(clip.id for clip, _ in clips) == ["a3", "b2"]
    tab.close()

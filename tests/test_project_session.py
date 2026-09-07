"""Session history stays shared, reversible, and honest about saved state."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.test_clip_disabled import _make_project_with_clips


def test_toggle_undo_redo_tracks_saved_state_and_revisions():
    project = _make_project_with_clips()
    session = project.session
    events = []
    project.add_observer(lambda name, data: events.append((name, project.is_dirty)))
    project.toggle_clips_disabled(["c0", "c0", "missing"])
    assert project.clips_by_id["c0"].disabled
    assert session.can_undo
    assert project.is_dirty
    session.undo()
    assert not project.clips_by_id["c0"].disabled
    assert not project.is_dirty
    session.redo()
    assert project.clips_by_id["c0"].disabled
    assert project.mutation_generation == 3
    assert events == [("clips_updated", True), ("clips_updated", False), ("clips_updated", True)]


def test_saved_history_position_and_new_branch():
    project = _make_project_with_clips()
    project.toggle_clips_disabled(["c0"])
    project.mark_clean()
    project.session.undo()
    assert project.is_dirty
    project.session.redo()
    assert not project.is_dirty
    project.session.undo()
    project.toggle_clips_disabled(["c1"])
    assert not project.session.can_redo
    assert project.is_dirty


def test_analysis_does_not_enter_history_or_get_erased_by_undo():
    project = _make_project_with_clips()
    project.toggle_clips_disabled(["c0"])
    project.clips[0].dominant_colors = [(1, 2, 3)]
    project.update_clips([project.clips[0]])
    project.session.undo()
    assert project.is_dirty
    assert project.clips[0].dominant_colors == [(1, 2, 3)]
    assert not project.session.can_undo


def test_missing_and_unchanged_targets_do_not_create_history():
    project = _make_project_with_clips()
    assert project.set_clips_disabled(["missing"], True) == []
    assert project.set_clips_disabled(["c0"], False) == []
    assert not project.session.can_undo
    assert not project.is_dirty


def test_undo_rejects_replaced_targets_atomically():
    from models.clip import Clip

    project = _make_project_with_clips()
    project.toggle_clips_disabled(["c0", "c1"])
    project.remove_clips(["c1"])
    project.add_clips([Clip(id="c1", source_id="s1", start_frame=0, end_frame=1)])
    with pytest.raises(ValueError, match="changed"):
        project.session.undo()
    assert project.clips_by_id["c0"].disabled
    assert not project.clips_by_id["c1"].disabled


def test_clear_resets_history_and_session_identity():
    project = _make_project_with_clips()
    project.toggle_clips_disabled(["c0"])
    old_id = project.session.session_id
    project.clear()
    assert not project.session.can_undo
    assert not project.session.can_redo
    assert project.session.session_id != old_id
    assert not project.is_dirty


def test_session_rejects_foreign_thread_before_mutation():
    project = _make_project_with_clips()
    session = project.session
    with ThreadPoolExecutor() as pool:
        with pytest.raises(RuntimeError, match="owner thread"):
            pool.submit(project.toggle_clips_disabled, ["c0"]).result()
    assert not project.clips[0].disabled
    assert not session.can_undo


def test_agent_toggle_shares_project_history():
    from core.chat_tools import toggle_clip_disabled

    project = _make_project_with_clips()
    project.toggle_clips_disabled(["c0"])
    result = toggle_clip_disabled(project, ["c1", "missing"], disabled=True)
    assert result["updated"] == [{"id": "c1", "disabled": True}]
    assert result["not_found"] == ["missing"]
    project.session.undo()
    assert not project.clips[1].disabled
    assert project.clips[0].disabled
    project.session.undo()
    assert not project.is_dirty


def test_pending_color_result_rejected_after_clear_and_reimport(tmp_path):
    from unittest.mock import patch

    from core.operations.colors import ColorApplication, color_request, compute_colors
    from tests.test_spine_analyze import _build_project

    project = _build_project(tmp_path, 1)
    application = ColorApplication(project, color_request(project))
    with patch("core.analysis.color.extract_dominant_colors", return_value=[(1, 2, 3)]):
        result = compute_colors(application.request)
    old_clips, old_sources = list(project.clips), list(project.sources)
    project.clear()
    for source in old_sources:
        project.add_source(source)
    project.add_clips(old_clips)
    with pytest.raises(ValueError, match="session"):
        application.apply(result)
    assert project.clips[0].dominant_colors is None


def test_legacy_gui_edit_while_dirty_survives_undo():
    from types import SimpleNamespace

    from ui.main_window import MainWindow

    project = _make_project_with_clips()
    project.toggle_clips_disabled(["c0"])
    project.clips[0].notes = "Keep this unsaved note"
    window = SimpleNamespace(project=project, _update_window_title=lambda: None)
    MainWindow._mark_dirty(window)
    project.session.undo()
    assert project.clips[0].notes == "Keep this unsaved note"
    assert project.is_dirty


def test_frame_analysis_update_survives_undo():
    from models.frame import Frame

    project = _make_project_with_clips()
    project.add_frames([Frame(id="f0")])
    project.mark_clean()
    project.toggle_clips_disabled(["c0"])
    project.update_frame("f0", description="Keep this analysis")
    project.session.undo()
    assert project.frames_by_id["f0"].description == "Keep this analysis"
    assert project.is_dirty

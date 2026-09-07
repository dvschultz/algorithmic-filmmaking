"""Insertion/removal preserve timeline identity, timing and shared history."""

from types import SimpleNamespace

import pytest

from core.chat_tools import add_to_sequence, new_project, remove_from_sequence, undo
from models.sequence import Sequence, SequenceClip, Track
from tests.test_clip_disabled import _make_project_with_clips


def test_agent_batch_insertion_is_one_edit_and_undo_restores_saved_state():
    project = _make_project_with_clips()
    events = []
    project.add_observer(
        lambda event, data: events.append((event, data, project.is_dirty))
    )
    assert add_to_sequence(project, ["c0", "c1"])["success"]
    entries = list(project.sequence.tracks[0].clips)
    assert len(entries) == 2
    assert entries[1].start_frame == entries[0].duration_frames
    assert len(events) == 1 and events[0][0] == "sequence_changed"
    project.session.undo()
    assert not project.sequence.get_all_clips()
    assert not project.is_dirty
    project.session.redo()
    assert project.sequence.tracks[0].clips == entries
    assert project.sequence.tracks[0].clips[0] is entries[0]


def test_removal_restores_gaps_trims_transforms_and_order():
    project = _make_project_with_clips()
    entries = [
        SequenceClip(
            source_clip_id=f"c{i}",
            source_id="s1",
            start_frame=i * 100,
            in_point=3,
            out_point=23,
            hflip=True,
        )
        for i in range(3)
    ]
    project.insert_sequence_clips(entries)
    project.mark_clean()
    assert remove_from_sequence(project, [entries[1].id])["success"]
    assert [c.start_frame for c in project.sequence.tracks[0].clips] == [0, 20]
    project.session.undo()
    assert not project.is_dirty
    assert [c.start_frame for c in project.sequence.tracks[0].clips] == [0, 100, 200]
    assert project.sequence.tracks[0].clips[1] is entries[1]
    assert entries[1].in_point == 3 and entries[1].hflip
    project.session.redo()
    assert entries[1] not in project.sequence.tracks[0].clips


def test_noops_leave_history_and_dirty_unchanged():
    project = _make_project_with_clips()
    project.add_to_sequence(["missing"])
    assert project.remove_from_sequence(["missing"]) == []
    assert not project.is_dirty and not project.session.can_undo


def test_clear_all_tracks_is_one_edit_and_restores_original_entries():
    from core.chat_tools import clear_sequence

    project = _make_project_with_clips()
    project.sequence.tracks.append(Track())
    entries = [
        SequenceClip(source_id="s1", source_clip_id=f"c{i}", track_index=i,
                     start_frame=100, in_point=5, out_point=25, hflip=True)
        for i in range(2)
    ]
    project.insert_sequence_clips(entries)
    project.mark_clean()
    events = []
    project.add_observer(lambda event, data: events.append((event, data)))

    assert clear_sequence(project)["clips_removed"] == 2
    assert not project.sequence.get_all_clips()
    assert events == [("sequence_changed", [])]
    project.session.undo()
    assert not project.is_dirty
    for index, entry in enumerate(entries):
        assert project.sequence.tracks[index].clips[0] is entry
        assert (entry.start_frame, entry.in_point, entry.out_point, entry.hflip) == (100, 5, 25, True)
    project.session.redo()
    assert not project.sequence.get_all_clips()


def test_clear_empty_sequence_is_noop():
    project = _make_project_with_clips()
    events = []
    project.add_observer(lambda event, data: events.append(event))
    assert project.clear_sequence() == 0
    assert not project.is_dirty and not project.session.can_undo
    assert events == []


def test_clear_retains_source_media_for_undo():
    project = _make_project_with_clips()
    project.add_to_sequence(["c0"])
    project.clear_sequence()
    assert project.source_in_sequences("s1") == [project.sequence.name]
    project.remove_source("s1")
    project.session.undo()  # Source removal restores media before Clear is undone.
    project.session.undo()
    assert project.sequence.get_all_clips()[0].source_id in project.sources_by_id


def test_clear_undo_rejects_permanently_removed_library_clip_atomically():
    project = _make_project_with_clips()
    project.add_to_sequence(["c0", "c1"])
    project.clear_sequence()
    project.remove_clips(["c1"])
    with pytest.raises(ValueError, match="referenced media was removed"):
        project.session.undo()
    assert not project.sequence.get_all_clips()
    assert project.session.undo_text == "Clear sequence"


def test_history_targets_original_sequence_after_active_sequence_switch():
    project = _make_project_with_clips()
    first = project.sequence
    second = Sequence()
    project.add_sequence(second)
    project.mark_clean()
    project.add_to_sequence(["c0"])
    project.set_active_sequence(1)
    project.session.undo()
    assert not first.get_all_clips() and not second.get_all_clips()
    assert project.sequence is second
    assert not project.is_dirty


def test_multitrack_conflict_is_atomic():
    project = _make_project_with_clips()
    project.sequence.tracks.append(Track())
    entries = [SequenceClip(track_index=i, out_point=20) for i in range(2)]
    project.insert_sequence_clips(entries)
    entries[1].out_point = 10  # unmigrated trim
    with pytest.raises(ValueError, match="Sequence changed"):
        project.session.undo()
    assert project.sequence.tracks[0].clips == [entries[0]]
    assert project.sequence.tracks[1].clips == [entries[1]]


def test_analysis_survives_sequence_undo_and_new_project_clears_history():
    project = _make_project_with_clips()
    project.add_to_sequence(["c0"])
    project.clips[0].dominant_colors = [(1, 2, 3)]
    project.update_clips([project.clips[0]])
    project.session.undo()
    assert project.clips[0].dominant_colors == [(1, 2, 3)]
    assert project.is_dirty
    project.clear()
    assert not project.session.can_redo and not project.session.can_undo


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_timeline_and_agent_share_commands_without_refresh_marking_dirty(qapp):
    from ui.project_adapter import ProjectSignalAdapter
    from ui.session_history import SessionHistoryAdapter
    from ui.timeline.timeline_widget import TimelineWidget

    project = _make_project_with_clips()
    timeline = TimelineWidget()
    timeline.scene.project = project
    timeline.scene.set_sequence(project.sequence)
    adapter = ProjectSignalAdapter(project)
    adapter.sequence_changed.connect(
        lambda _: timeline.load_sequence(
            project.sequence, project.sources_by_id, project.clips
        )
    )
    timeline.sequence_changed.connect(project.mark_dirty)
    history = SessionHistoryAdapter(project.session)
    action = history.createUndoAction(history)
    timeline.add_clip(project.clips[0], project.sources[0], start_frame=50)
    entry = project.sequence.get_all_clips()[0]
    assert entry.id in timeline.scene._clip_items
    assert project.is_dirty
    action.trigger()
    assert not project.is_dirty and not timeline.scene._clip_items
    project.session.redo()
    project.mark_clean()
    timeline.scene.remove_clip(entry.id)
    assert not timeline.scene._clip_items
    assert undo(SimpleNamespace(undo_stack=history))["success"]
    assert not project.is_dirty
    assert project.sequence.get_all_clips()[0].start_frame == 50
    assert entry.id in timeline.scene._clip_items
    project.mark_clean()
    timeline.clear_timeline()
    assert not timeline.scene._clip_items
    assert not project.sequence.get_all_clips()
    action.trigger()
    assert not project.is_dirty
    assert project.sequence.get_all_clips()[0] is entry
    assert entry.id in timeline.scene._clip_items
    timeline.close()


def test_generation_clear_does_not_record_manual_history(qapp):
    from ui.timeline.timeline_widget import TimelineWidget

    project = _make_project_with_clips()
    project.sequence.tracks[0].clips.append(SequenceClip(out_point=20))
    timeline = TimelineWidget()
    timeline.scene.project = project
    timeline.scene.set_sequence(project.sequence)
    timeline.scene.history_enabled = lambda: False
    timeline.clear_timeline()
    assert not project.sequence.get_all_clips()
    assert not project.session.can_undo
    timeline.close()


def test_new_project_can_clear_departing_sequence_view(qapp):
    from ui.tabs.sequence_tab import SequenceTab

    project = _make_project_with_clips()
    project.add_to_sequence(["c0"])
    tab = SequenceTab()
    tab.set_project(project)
    tab.timeline.sequence_changed.connect(project.mark_dirty)
    # MainWindow resets the model before clearing and rebinding the tab.
    project.clear()
    tab.clear()
    tab.set_project(project)
    assert tab.timeline.get_sequence() is project.sequence
    assert not project.session.can_undo
    assert not project.is_dirty
    tab.close()


def test_agent_new_project_rebinds_history(qapp):
    from ui.project_adapter import ProjectSignalAdapter
    from ui.session_history import SessionHistoryAdapter

    project = _make_project_with_clips()
    window = SimpleNamespace(
        project=project,
        undo_stack=SessionHistoryAdapter(project.session),
        _project_adapter=ProjectSignalAdapter(project),
        _clear_project_state=project.clear,
        _update_window_title=lambda: None,
    )
    new_project(main_window=window)
    window.project.add_source(_make_project_with_clips().sources[0])
    window.project.add_clips(_make_project_with_clips().clips)
    window.project.mark_clean()
    window.project.add_to_sequence(["c0"])
    assert undo(window)["success"]
    assert not window.project.sequence.get_all_clips()


def test_agent_load_project_rebinds_history(qapp, tmp_path, monkeypatch):
    from core.chat_tools import load_project
    from core.project import Project
    from ui.project_adapter import ProjectSignalAdapter
    from ui.session_history import SessionHistoryAdapter

    first = _make_project_with_clips()
    second = _make_project_with_clips()
    path = tmp_path / "project.json"
    path.write_text("{}")
    monkeypatch.setattr(Project, "load", lambda *args, **kwargs: second)
    window = SimpleNamespace(
        project=first,
        undo_stack=SessionHistoryAdapter(first.session),
        _project_adapter=ProjectSignalAdapter(first),
        _clear_project_state=first.clear,
        _refresh_ui_from_project=lambda: None,
    )
    assert load_project(str(path), main_window=window)["success"]
    second.add_to_sequence(["c0"])
    assert undo(window)["success"]
    assert not second.sequence.get_all_clips()
    with pytest.raises(RuntimeError, match="closed"):
        first.add_to_sequence(["c0"])


def test_invalid_insert_does_not_partially_modify_tracks():
    project = _make_project_with_clips()
    with pytest.raises(ValueError, match="Invalid sequence track"):
        project.insert_sequence_clips(
            [SequenceClip(out_point=20), SequenceClip(track_index=4, out_point=20)]
        )
    assert not project.sequence.get_all_clips()
    assert not project.is_dirty and not project.session.can_undo


def test_loading_empty_sequence_preserves_departing_sequence_and_history(qapp):
    from unittest.mock import Mock
    from ui.tabs.sequence_tab import SequenceTab
    from ui.timeline.timeline_widget import TimelineWidget

    project = _make_project_with_clips()
    project.add_to_sequence(["c0"])
    first = project.sequence
    entry = first.get_all_clips()[0]
    timeline = TimelineWidget()
    timeline.scene.set_sequence(first)
    project.add_sequence(Sequence())
    project.set_active_sequence(1)
    tab = SimpleNamespace(
        _project=project,
        _sources=project.sources_by_id,
        _clips=project.clips,
        timeline=timeline,
        timeline_preview=Mock(),
        _set_state=Mock(),
        sync_sequence_metadata=Mock(),
        STATE_CARDS=0,
        STATE_TIMELINE=1,
    )
    SequenceTab._load_active_sequence(tab)
    assert first.get_all_clips() == [entry]
    assert timeline.sequence is project.sequence
    project.session.undo()  # Undo creation of the empty sequence.
    assert first.get_all_clips() == [entry]
    project.session.undo()
    assert not first.get_all_clips()
    timeline.close()


def test_frame_batch_is_one_undoable_insertion():
    from models.frame import Frame

    project = _make_project_with_clips()
    project.add_frames([Frame(id="f1"), Frame(id="f2")])
    project.mark_clean()
    project.add_frames_to_sequence(["f1", "missing", "f2"], hold_frames=12)
    entries = project.sequence.get_all_clips()
    assert [(c.frame_id, c.start_frame, c.hold_frames) for c in entries] == [
        ("f1", 0, 12),
        ("f2", 12, 12),
    ]
    project.session.undo()
    assert not project.sequence.get_all_clips() and not project.is_dirty
    project.session.redo()
    assert project.sequence.get_all_clips() == entries


def test_fps_change_stays_dirty_when_insertion_is_undone(qapp):
    from ui.timeline.timeline_widget import TimelineWidget

    project = _make_project_with_clips()
    timeline = TimelineWidget()
    timeline.scene.project = project
    timeline.scene.set_sequence(project.sequence)
    timeline.set_fps(24.0)
    timeline.add_clip(project.clips[0], project.sources[0])
    project.session.undo()
    assert project.sequence.fps == 24.0 and project.is_dirty
    timeline.close()


def test_removal_keeps_requested_id_order_without_duplicate_edits():
    project = _make_project_with_clips()
    project.add_to_sequence(["c0", "c1"])
    first, second = project.sequence.get_all_clips()
    assert project.remove_from_sequence([second.id, first.id, second.id]) == [
        second.id,
        first.id,
    ]
    project.session.undo()
    assert project.sequence.get_all_clips() == [first, second]

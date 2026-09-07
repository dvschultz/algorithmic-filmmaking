"""Sequence lifecycle history preserves selection, identity and media references."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.chat_tools import (
    create_sequence,
    delete_sequence,
    list_sequences,
    rename_sequence,
    update_sequence,
)
from core.project import Project
from models.sequence import Sequence
from tests.test_clip_disabled import _make_project_with_clips


def test_create_activate_undo_redo_are_one_edit():
    project = Project.new()
    original = project.sequence
    result = create_sequence(project, name="Second", fps=24)
    created = project.sequence
    assert result["success"] and created is not original
    assert project.session.undo_text == "Create sequence"
    project.session.undo()
    assert project.sequences == [original] and project.sequence is original
    assert not project.is_dirty
    project.session.redo()
    assert project.sequence is created and created.id == result["sequence_id"]


def test_delete_last_sequence_restores_contents_metadata_and_identity():
    project = _make_project_with_clips()
    project.add_to_sequence(["c0"])
    original = project.sequence
    original.algorithm = "shuffle"
    original.music_path = "/tmp/music.wav"
    entry = original.get_all_clips()[0]
    project.mark_clean()
    assert delete_sequence(project, original.id)["success"]
    fallback = project.sequence
    assert fallback is not original and not fallback.get_all_clips()
    project.session.undo()
    assert project.sequence is original and not project.is_dirty
    assert original.get_all_clips()[0] is entry
    assert original.algorithm == "shuffle" and original.music_path == "/tmp/music.wav"
    project.session.redo()
    assert project.sequence is fallback


def test_rename_nonactive_sequence_keeps_selection_and_tracks_save():
    project = Project.new()
    first = project.sequence
    project.add_sequence(Sequence(name="Second"), activate=True)
    active = project.sequence
    project.mark_clean()
    assert rename_sequence(project, first.id, " Renamed ")["success"]
    assert first.name == "Renamed" and project.sequence is active
    project.session.undo()
    assert first.name != "Renamed" and project.sequence is active
    assert not project.is_dirty
    generation = project.mutation_generation
    project.rename_sequence(0, first.name)
    assert project.mutation_generation == generation
    assert list_sequences(project)["sequences"][1]["active"]


def test_metadata_validation_is_atomic_and_grouped():
    project = Project.new()
    before = project.sequence.name
    assert not update_sequence(project, name="Changed", fps=-1)["success"]
    assert project.sequence.name == before and not project.is_dirty
    assert update_sequence(project, name="Changed", fps=24)["success"]
    project.session.undo()
    assert project.sequence.name == before and project.sequence.fps == 30
    assert not project.is_dirty


def test_delete_before_active_restores_index_and_original_selection():
    project = Project.new()
    first = project.sequence
    project.add_sequence(Sequence(name="Active"), activate=True)
    active = project.sequence
    project.mark_clean()
    project.remove_sequence(0)
    assert project.sequence is active and project.active_sequence_index == 0
    project.session.undo()
    assert project.sequence is active and project.active_sequence_index == 1
    assert project.sequences[0] is first and not project.is_dirty


def test_deleted_sequence_media_remains_protected_until_history_is_cleared():
    project = _make_project_with_clips()
    project.add_to_sequence(["c0"])
    source_id = project.clips[0].source_id
    project.remove_sequence(0)
    assert project.source_in_sequences(source_id)
    with pytest.raises(ValueError, match="undo history"):
        project.remove_source(source_id)
    assert source_id in project.sources_by_id
    project.session.undo()
    assert project.sequence.get_all_clips()[0].source_id == source_id
    project.clear()
    assert not project.session.retained_sequences


def test_replaced_sequence_collection_conflicts_without_partial_mutation():
    project = Project.new()
    project.add_sequence(Sequence())
    unrelated = Sequence()
    project.sequences = [unrelated]
    with pytest.raises(ValueError, match="collection changed"):
        project.session.undo()
    assert project.sequences == [unrelated]


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_desktop_new_sequence_and_agent_rename_share_menu_history(qapp):
    from PySide6.QtWidgets import QComboBox
    from ui.project_adapter import ProjectSignalAdapter
    from ui.session_history import SessionHistoryAdapter
    from ui.tabs.sequence_tab import SequenceTab
    from ui.timeline.timeline_widget import TimelineWidget

    project = Project.new()
    original = project.sequence
    timeline = TimelineWidget()
    timeline.scene.set_sequence(original)
    tab = SimpleNamespace(
        _project=project,
        timeline=timeline,
        sequence_dropdown=QComboBox(),
        cards_sequence_dropdown=QComboBox(),
        _sources={},
        _clips=[],
        timeline_preview=Mock(),
        _set_state=Mock(),
        sync_sequence_metadata=Mock(),
        STATE_CARDS=0,
        STATE_TIMELINE=1,
    )
    for name in [
        "_sync_sequence_dropdown",
        "_load_active_sequence",
        "_persist_current_sequence",
    ]:
        setattr(tab, name, getattr(SequenceTab, name).__get__(tab))
    adapter = ProjectSignalAdapter(project)
    adapter.sequences_changed.connect(lambda _: tab._sync_sequence_dropdown())
    adapter.active_sequence_changed.connect(lambda _: tab._load_active_sequence())
    timeline.sequence_changed.connect(project.mark_dirty)
    history = SessionHistoryAdapter(project.session)
    undo = history.createUndoAction(history)
    SequenceTab._on_new_sequence_clicked(tab)
    created = project.sequence
    assert tab.sequence_dropdown.count() == 2 and timeline.sequence is created
    undo.trigger()
    assert timeline.sequence is original and tab.sequence_dropdown.count() == 1
    assert not project.is_dirty
    project.session.redo()
    assert timeline.sequence is created
    project.mark_clean()
    assert rename_sequence(project, created.id, "Agent rename")["success"]
    assert tab.sequence_dropdown.currentText() == "Agent rename"
    undo.trigger()
    assert tab.sequence_dropdown.currentText() == "Untitled Sequence"
    assert not project.is_dirty
    timeline.close()


def test_undo_deleted_sequence_rejects_permanently_removed_clips_atomically():
    project = _make_project_with_clips()
    project.add_to_sequence(["c0"])
    project.remove_sequence(0)
    fallback = project.sequence
    project.remove_clips(["c0"])
    with pytest.raises(ValueError, match="referenced media was removed"):
        project.session.undo()
    assert project.sequences == [fallback] and project.sequence is fallback

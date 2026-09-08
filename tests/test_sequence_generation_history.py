"""Generation publishes completed output once and never replays computation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.spine.sequences import SequenceDraft
from models.sequence import SequenceClip
from tests.test_clip_disabled import _make_project_with_clips


@pytest.mark.parametrize(
    "populated,replace", [(False, False), (True, False), (True, True)]
)
def test_generation_is_detached_until_one_commit_and_restores_original(
    populated, replace
):
    project = _make_project_with_clips()
    if populated:
        project.add_to_sequence(["c0"])
    original = project.sequence
    before = original.to_dict()
    project.mark_clean()
    generation = project._mutation_generation
    proposal = SequenceDraft.prepare(
        project,
        "color",
        "Chromatics",
        replace_sequence_id=original.id if replace else None,
        show_chromatic_color_bar=True,
    )
    entry = SequenceClip(source_id="s1", source_clip_id="c1", out_point=50, hflip=True)
    proposal.sequence.tracks[0].clips.append(entry)
    assert original.to_dict() == before and not project.is_dirty
    proposal.commit(project)
    assert project._mutation_generation == generation + 1
    assert project.session.undo_text == "Generate sequence"
    assert project.sequence is proposal.sequence
    assert len(project.sequences) == (2 if populated and not replace else 1)
    project.session.undo()
    assert project.sequence is original and original.to_dict() == before
    assert not project.is_dirty
    project.session.redo()
    assert project.sequence.tracks[0].clips[0] is entry
    assert project.sequence.show_chromatic_color_bar


def test_draft_rejects_changed_replacement_and_previous_project_session():
    project = _make_project_with_clips()
    proposal = SequenceDraft.prepare(project, "shuffle", "Draft")
    project.rename_sequence(0, "New name")
    with pytest.raises(ValueError, match="target changed"):
        proposal.commit(project)
    assert project.sequence.name == "New name"
    project.clear()
    with pytest.raises(ValueError, match="previous project session"):
        proposal.commit(project)
    assert not project.is_dirty


def test_invalid_generated_media_leaves_project_and_history_unchanged():
    project = _make_project_with_clips()
    proposal = SequenceDraft.prepare(project, "shuffle", "Draft")
    proposal.sequence.tracks[0].clips.append(
        SequenceClip(source_clip_id="missing", out_point=5)
    )
    with pytest.raises(ValueError, match="referenced media was removed"):
        proposal.commit(project)
    assert project.sequence is proposal.origin
    assert not project.is_dirty and not project.session.can_undo


def test_generated_order_preserves_relative_subclip_ranges():
    from core.spine.sequences import apply_generated_order

    project = _make_project_with_clips()
    clip = project.clips[1]  # source frames 100..200
    sequence = apply_generated_order(
        project,
        [(clip, project.sources[0])],
        "signature_style",
        "Signature Style",
        relative_ranges=[(10, 30)],
    )
    entry = sequence.get_all_clips()[0]
    assert (entry.start_frame, entry.in_point, entry.out_point) == (0, 110, 130)
    project.session.undo()
    project.session.redo()
    assert project.sequence.get_all_clips()[0] is entry


def test_agent_reports_failed_generation_commit(monkeypatch):
    from core.chat_tools import generate_cassette_tape

    project = _make_project_with_clips()
    clip, source = project.clips[0], project.sources[0]
    clip.transcript = [SimpleNamespace(text="hello")]
    monkeypatch.setattr("core.remix.cassette_tape.match_phrases", lambda *args: {"hello": [object()]})
    monkeypatch.setattr("core.remix.cassette_tape.flatten_matches_in_phrase_order", lambda *args: [object()])
    monkeypatch.setattr("core.remix.cassette_tape.build_sequence_data", lambda *args: [(clip, source, 0, 10)])
    tab = Mock()
    tab._apply_cassette_tape_sequence.return_value = False
    window = SimpleNamespace(_gui_state=None, sequence_tab=tab)
    result = generate_cassette_tape(project, window, [{"phrase": "hello"}])
    assert result["success"] is False
    assert not project.sequence.get_all_clips()


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("fail", [False, True])
def test_worker_application_commits_once_or_discards_partial_output(
    qapp, monkeypatch, fail
):
    from ui.tabs.sequence_tab import SequenceTab
    from ui.project_adapter import ProjectSignalAdapter

    project = _make_project_with_clips()
    project.add_to_sequence(["c0"])
    original = project.sequence
    project.mark_clean()
    tab = SequenceTab()
    tab.set_project(project)
    tab.timeline.sequence_changed.connect(project.mark_dirty)
    adapter = ProjectSignalAdapter(project)
    adapter.active_sequence_changed.connect(lambda _: tab._load_active_sequence())
    proposal = SequenceDraft.prepare(
        project, "color", "Chromatics", replace_sequence_id=original.id
    )
    worker = SimpleNamespace(_pending_algorithm="color", _pending_draft=proposal)
    tab._sequence_worker = worker
    errors = Mock()
    monkeypatch.setattr("ui.tabs.sequence_tab.QMessageBox.critical", errors)
    if fail:
        real_add = tab.timeline.add_clip
        calls = []

        def add(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise RuntimeError("Population failed")
            return real_add(*args, **kwargs)

        monkeypatch.setattr(tab.timeline, "add_clip", add)
    tab._on_sequence_ready(
        [(clip, project.sources[0]) for clip in project.clips[:2]], worker
    )
    assert tab.isEnabled() and not tab.timeline.signalsBlocked()
    assert tab.timeline.get_sequence() is project.sequence
    if fail:
        errors.assert_called_once()
        assert project.sequence is original and not project.is_dirty
    else:
        errors.assert_not_called()
        assert len(project.sequence.get_all_clips()) == 2
        project.session.undo()
        assert project.sequence is original and not project.is_dirty
    tab.close()


def test_previous_session_worker_result_is_ignored(qapp, monkeypatch):
    from ui.tabs.sequence_tab import SequenceTab

    project = _make_project_with_clips()
    tab = SequenceTab()
    tab.set_project(project)
    worker = SimpleNamespace(
        _pending_algorithm="color",
        _pending_draft=SequenceDraft.prepare(project, "color", "Draft"),
    )
    tab._sequence_worker = worker
    errors = Mock()
    monkeypatch.setattr("ui.tabs.sequence_tab.QMessageBox.critical", errors)
    project.clear()
    tab._on_sequence_ready([], worker)
    errors.assert_not_called()
    assert not project.is_dirty and not project.session.can_undo
    tab.close()


def test_agent_generation_without_analysis_returns_to_saved_state_on_undo(
    qapp, monkeypatch
):
    from ui.tabs.sequence_tab import SequenceTab

    project = _make_project_with_clips()
    project.mark_clean()
    tab = SequenceTab()
    tab.set_project(project)
    tab._gui_state = SimpleNamespace(analyze_selected_ids=["c0"], cut_selected_ids=[])
    pairs = [(project.clips[0], project.sources[0])]
    monkeypatch.setattr(tab, "_resolve_selected_clips", lambda _: pairs)
    monkeypatch.setattr("ui.tabs.sequence_tab.generate_sequence", lambda **_: pairs)
    tab.clips_data_changed.connect(project.update_clips)
    tab.timeline.sequence_changed.connect(project.mark_dirty)
    assert tab.generate_and_apply("shuffle")["success"]
    project.session.undo()
    assert not project.is_dirty
    tab.close()


def test_worker_error_clears_pending_replace_target(qapp, monkeypatch):
    from ui.tabs.sequence_tab import SequenceTab

    project = _make_project_with_clips()
    project.add_to_sequence(["c0"])
    tab = SequenceTab()
    tab.set_project(project)
    tab._replace_sequence_index = 0
    worker = SimpleNamespace(_pending_draft=tab._prepare_sequence_draft("shuffle"))
    tab._sequence_worker = worker
    monkeypatch.setattr("ui.tabs.sequence_tab.QMessageBox.critical", Mock())
    tab._on_sequence_error("Failed", worker)
    assert tab._replace_sequence_index is None
    assert not tab._prepare_sequence_draft("shuffle").replace
    tab.close()


def test_free_association_rationales_are_present_at_commit(qapp, monkeypatch):
    from ui.tabs.sequence_tab import SequenceTab

    project = _make_project_with_clips()
    tab = SequenceTab()
    tab.set_project(project)
    monkeypatch.setattr(tab.video_player, "load_video", Mock())
    events = []
    project.add_observer(
        lambda event, _: events.append(
            [clip.rationale for clip in project.sequence.get_all_clips()]
        )
        if event == "sequences_changed"
        else None
    )
    source = project.sources[0]
    assert tab._apply_free_association_sequence(
        [
            (project.clips[0], source, None),
            (project.clips[1], source, "Visual link"),
        ]
    )
    assert events == [[None, "Visual link"]]
    project.session.undo()
    project.session.redo()
    assert project.sequence.get_all_clips()[1].rationale == "Visual link"
    tab.close()


@pytest.mark.parametrize("cancel", [False, True])
def test_real_worker_delivers_on_owner_thread_and_releases_draft(
    qapp, monkeypatch, cancel
):
    from threading import Event
    from ui.tabs.sequence_tab import SequenceTab

    project = _make_project_with_clips()
    tab = SequenceTab()
    tab.set_project(project)
    gate = Event()
    pairs = [(project.clips[0], project.sources[0])]

    def compute(**kwargs):
        assert gate.wait(2)
        return pairs

    monkeypatch.setattr("core.remix.generate_sequence", compute)
    monkeypatch.setattr("ui.tabs.sequence_tab.estimate_sequence_cost", lambda *args, **kwargs: [])
    errors = Mock()
    monkeypatch.setattr("ui.tabs.sequence_tab.QMessageBox.critical", errors)
    tab._apply_algorithm("shuffle", pairs)
    worker = tab._sequence_worker
    if cancel:
        worker.cancel()
    gate.set()
    assert worker.wait(2000)
    qapp.processEvents()
    errors.assert_not_called()
    assert project.session.can_undo is (not cancel)
    assert worker._pending_draft is None
    assert not worker._pending_inputs
    tab.close()

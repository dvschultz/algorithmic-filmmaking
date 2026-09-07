"""Source removal retains model references without rerunning import/analysis."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.chat_tools import remove_source
from models.clip import Source
from models.frame import Frame
from models.sequence import Sequence, SequenceClip, Track
from tests.test_clip_disabled import _make_project_with_clips


def test_source_removal_restores_library_and_all_sequences_in_one_undo():
    project = _make_project_with_clips()
    source = project.sources[0]
    clips = list(project.clips)
    frame = Frame(id="frame", source_id=source.id, clip_id="c0")
    project.add_frames([frame])
    project.add_to_sequence(["c0"])
    first = project.sequence
    first.tracks.append(Track())
    entry = SequenceClip(
        frame_id=frame.id, hold_frames=45, track_index=1, start_frame=100
    )
    first.tracks[1].clips.append(entry)
    second = Sequence()
    project.add_sequence(second, activate=True)
    project.add_to_sequence(["c1"])
    before = [list(track.clips) for seq in project.sequences for track in seq.tracks]
    project.mark_clean()
    generation = project._mutation_generation
    events = []
    project.add_observer(
        lambda event, data: events.append((event, data, project.is_dirty))
    )

    result = remove_source(None, project, source.id)
    assert result["success"] and result["removed_clips_count"] == 3
    assert not project.sources and not project.clips and not project.frames
    assert all(not seq.get_all_clips() for seq in project.sequences)
    assert project._mutation_generation == generation + 1
    assert events[0] == ("sources_changed", [source.id], True)
    assert events[1] == ("source_removed", source, True)
    project.session.undo()
    assert not project.is_dirty
    assert project.sources[0] is source and project.frames[0] is frame
    assert all(a is b for a, b in zip(project.clips, clips))
    after = [list(track.clips) for seq in project.sequences for track in seq.tracks]
    assert before == after and first.tracks[1].clips[0] is entry
    assert project.sequence is second
    project.session.redo()
    assert not project.sources and not project.frames
    assert all(not seq.get_all_clips() for seq in project.sequences)


def test_batch_removal_is_one_edit_and_unknown_sources_are_noops():
    project = _make_project_with_clips()
    other = Source(id="other", file_path=Path("/tmp/other.mp4"))
    project.add_source(other)
    assert project.remove_source("missing") is None
    assert not project.session.can_undo
    project.mark_clean()
    assert len(project.remove_sources(["s1", "other", "s1"])) == 2
    project.session.undo()
    assert len(project.sources) == 2 and not project.is_dirty


def test_source_undo_conflict_does_not_partially_restore_library():
    project = _make_project_with_clips()
    project.add_to_sequence(["c0"])
    project.remove_source("s1")
    project.sequence.tracks[0].clips.append(SequenceClip(out_point=20))
    with pytest.raises(ValueError, match="Sequence changed"):
        project.session.undo()
    assert not project.sources and not project.clips
    assert project.session.undo_text == "Remove source"


def test_source_redo_rejects_new_frame_references_before_mutation():
    project = _make_project_with_clips()
    frame = Frame(id="frame", source_id="s1")
    project.add_frames([frame])
    project.remove_source("s1")
    project.session.undo()
    # Simulate a legacy writer that bypasses history and leaves redo available.
    project.sequence.tracks.append(Track(clips=[SequenceClip(frame_id=frame.id)]))
    with pytest.raises(ValueError, match="new sequence edit"):
        project.session.redo()
    assert project.frames[0] is frame and project.sources[0].id == "s1"


def test_source_restoration_signal_refreshes_all_library_views():
    from ui.main_window import MainWindow
    from ui.project_adapter import ProjectSignalAdapter

    project = _make_project_with_clips()
    original = project.sources[0]
    ui = SimpleNamespace(
        project=project,
        current_source=original,
        collect_tab=Mock(),
        cut_tab=Mock(),
        analyze_tab=Mock(),
        frames_tab=Mock(),
        _stop_playback=Mock(),
        _auto_include_analyzed_clips=Mock(),
        _refresh_timeline_from_project=Mock(),
        _update_window_title=Mock(),
    )
    ui.analyze_tab.get_clip_ids.return_value = {"c0"}
    adapter = ProjectSignalAdapter(project)
    adapter.sources_changed.connect(lambda ids: MainWindow._on_sources_changed(ui, ids))
    project.remove_source("s1")
    ui.collect_tab.remove_source.assert_called_once_with("s1")
    ui.cut_tab.set_clip_source_pairs.assert_called_with([])
    assert ui.current_source is None
    project.session.undo()
    ui.collect_tab.add_source.assert_called_once_with(original)
    assert len(ui.cut_tab.set_clip_source_pairs.call_args.args[0]) == 3
    ui.frames_tab.update_frame_browser.assert_called()
    assert ui.current_source is original
    ui.analyze_tab.add_clips.assert_called_once_with(["c0"])


def test_source_edits_require_owner_thread():
    from concurrent.futures import ThreadPoolExecutor

    project = _make_project_with_clips()
    with ThreadPoolExecutor() as pool:
        with pytest.raises(RuntimeError, match="owner thread"):
            pool.submit(project.remove_source, "s1").result()
    assert len(project.sources) == 1 and not project.session.can_undo


@pytest.mark.asyncio
async def test_mcp_source_removal_clears_every_sequence_and_preserves_response(
    tmp_path,
):
    import json
    from core.project import Project
    from scene_ripper_mcp.tools.project import remove_source as mcp_remove_source

    project = _make_project_with_clips()
    video = tmp_path / "video.mp4"
    video.touch()
    project.sources[0].file_path = video
    project.add_to_sequence(["c0"])
    project.add_sequence(Sequence(), activate=True)
    project.add_to_sequence(["c1"])
    path = tmp_path / "project.sceneripper"
    project.save(path)
    response = json.loads(await mcp_remove_source(str(path), "s1"))
    assert response == {
        "success": True,
        "removed_source": "video.mp4",
        "removed_clips": 3,
        "remaining_sources": 0,
        "remaining_clips": 0,
    }
    loaded = Project.load(path)
    assert all(not sequence.get_all_clips() for sequence in loaded.sequences)
    assert video.exists()

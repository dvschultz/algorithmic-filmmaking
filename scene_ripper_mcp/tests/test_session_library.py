"""Legacy insertion and library edits participate in one retained history."""

import json
from types import SimpleNamespace

import pytest
import pytest_asyncio

from core.project import Project
from models.clip import Source, Clip
from scene_ripper_mcp.project_sessions import SessionRuntime
from scene_ripper_mcp.tools import clips, sequence, sessions


@pytest_asyncio.fixture
async def ctx():
    runtime = SessionRuntime()
    yield SimpleNamespace(
        request_context=SimpleNamespace(lifespan_context={"project_sessions": runtime})
    )
    await runtime.shutdown()


@pytest.fixture
def path(tmp_path):
    path = tmp_path / "library.sceneripper"
    source = Source(
        id="s", file_path=tmp_path / "offline.mp4", fps=30, duration_seconds=30
    )
    project = Project(
        sources=[source],
        clips=[Clip(id="c", source_id="s", start_frame=100, end_frame=140)],
    )
    assert project.save(path)
    return path


@pytest.mark.asyncio
async def test_insertion_tracks_offsets_and_metadata_share_undo(path, ctx):
    sid = json.loads(await sessions.open_project_session(str(path), ctx))["session_id"]
    result = json.loads(
        await sequence.add_to_sequence(str(path), ["c"], track_index=2, ctx=ctx)
    )
    assert result["success"], result
    loaded = Project.load(path)
    assert len(loaded.sequence.tracks) == 3
    assert [track.name for track in loaded.sequence.tracks[1:]] == [
        "Video 2",
        "Video 3",
    ]
    entry = loaded.sequence.tracks[2].clips[0]
    assert (entry.in_point, entry.out_point) == (100, 140)
    assert json.loads(await clips.add_clip_tags(str(path), "c", [" one ", "two"], ctx))[
        "all_tags"
    ] == ["one", "two"]
    assert json.loads(await clips.remove_clip_tags(str(path), "c", ["one"], ctx))[
        "all_tags"
    ] == ["two"]
    assert (
        json.loads(await clips.add_clip_note(str(path), "c", " note ", ctx))["note"]
        == "note"
    )
    assert json.loads(await sessions.undo_project_session(sid, ctx))["success"]
    assert Project.load(path).clips[0].notes == ""
    assert json.loads(await sessions.undo_project_session(sid, ctx))["success"]
    assert Project.load(path).clips[0].tags == ["one", "two"]
    assert json.loads(await sessions.undo_project_session(sid, ctx))["success"]
    assert Project.load(path).clips[0].tags == []
    assert json.loads(await sessions.undo_project_session(sid, ctx))["success"]
    assert len(Project.load(path).sequence.tracks) == 1
    assert json.loads(await sessions.redo_project_session(sid, ctx))["success"]
    loaded = Project.load(path)
    assert loaded.sequence.tracks[2].clips[0].id == entry.id


@pytest.mark.asyncio
@pytest.mark.parametrize("track_index", [-1, 256])
async def test_invalid_track_does_not_create_history_or_write(path, ctx, track_index):
    sid = json.loads(await sessions.open_project_session(str(path), ctx))["session_id"]
    before = path.read_bytes()
    result = json.loads(
        await sequence.add_to_sequence(
            str(path), ["c"], track_index=track_index, ctx=ctx
        )
    )
    assert not result["success"]
    assert path.read_bytes() == before
    assert not json.loads(await sessions.get_project_session(sid, ctx))["can_undo"]


@pytest.mark.asyncio
async def test_standalone_insertion_preserves_source_offset(path):
    result = json.loads(await sequence.add_to_sequence(str(path), ["c"], ctx=None))
    assert result["success"]
    entry = Project.load(path).sequence.get_all_clips()[0]
    assert (entry.in_point, entry.out_point) == (100, 140)


@pytest.mark.asyncio
async def test_failed_insertion_save_discards_created_tracks(path, ctx, monkeypatch):
    sid = json.loads(await sessions.open_project_session(str(path), ctx))["session_id"]
    original = Project.save
    monkeypatch.setattr(Project, "save", lambda *a, **k: False)
    result = json.loads(
        await sequence.add_to_sequence(str(path), ["c"], track_index=2, ctx=ctx)
    )
    assert not result["success"]
    monkeypatch.setattr(Project, "save", original)
    assert len(Project.load(path).sequence.tracks) == 1
    state = json.loads(await sessions.get_project_session(sid, ctx))
    assert not state["can_undo"] and state["history_reset"]


@pytest.mark.asyncio
async def test_source_removal_and_metadata_restore_with_undo(path, ctx):
    from scene_ripper_mcp.tools.project import remove_source

    sid = json.loads(await sessions.open_project_session(str(path), ctx))["session_id"]
    await sequence.add_to_sequence(str(path), ["c"], ctx=ctx)
    assert json.loads(
        await sessions.update_session_clip(sid, "c", {"notes": "kept"}, ctx)
    )["success"]
    assert json.loads(await sessions.set_session_clips_disabled(sid, ["c"], True, ctx))[
        "success"
    ]
    assert Project.load(path).clips[0].disabled
    result = json.loads(await remove_source(str(path), "s", ctx))
    assert result["success"] and result["removed_clips"] == 1
    assert not Project.load(path).sources
    assert json.loads(await sessions.undo_project_session(sid, ctx))["success"]
    loaded = Project.load(path)
    assert loaded.clips[0].disabled and loaded.clips[0].notes == "kept"
    assert len(loaded.sequence.get_all_clips()) == 1
    assert json.loads(await sessions.undo_project_session(sid, ctx))["success"]
    assert not Project.load(path).clips[0].disabled


@pytest.mark.asyncio
async def test_transcript_json_edit_is_saved_and_undoable(path, ctx):
    sid = json.loads(await sessions.open_project_session(str(path), ctx))["session_id"]
    segment = {
        "start_time": 0.0,
        "end_time": 1.0,
        "text": "Hello",
        "words": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
    }
    result = json.loads(
        await sessions.update_session_clip(sid, "c", {"transcript": [segment]}, ctx)
    )
    assert result["success"], result
    loaded = Project.load(path).clips[0].transcript
    assert loaded[0].text == "Hello" and loaded[0].words[0].text == "Hello"
    assert json.loads(await sessions.undo_project_session(sid, ctx))["success"]
    assert not Project.load(path).clips[0].transcript


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "segment",
    [
        {"start_time": 2, "end_time": 1, "text": "bad"},
        {
            "start_time": 0,
            "end_time": 1,
            "text": "bad",
            "words": [{"start": 0, "end": 1, "text": False}],
        },
    ],
)
async def test_invalid_transcript_does_not_save_or_replace_history(path, ctx, segment):
    sid = json.loads(await sessions.open_project_session(str(path), ctx))["session_id"]
    await clips.add_clip_note(str(path), "c", "saved", ctx)
    expected = path.read_bytes()
    result = json.loads(
        await sessions.update_session_clip(sid, "c", {"transcript": [segment]}, ctx)
    )
    assert not result["success"]
    assert path.read_bytes() == expected
    assert json.loads(await sessions.undo_project_session(sid, ctx))["success"]


def test_track_creation_redo_validates_removed_track_before_publish(path):
    from models.sequence import SequenceClip

    project = Project.load(path)
    entry = SequenceClip(
        source_clip_id="c", source_id="s", track_index=2, in_point=100, out_point=140
    )
    project.insert_sequence_clips([entry], create_tracks=True)
    gap = project.sequence.tracks[1]
    project.session.undo()
    gap.clips.append(SequenceClip(in_point=0, out_point=1))
    with pytest.raises(ValueError, match="Sequence changed"):
        project.session.redo()
    assert len(project.sequence.tracks) == 1
    assert not project.sequence.get_all_clips()


def test_legacy_mtime_conflict_keeps_structured_error_fields(path):
    from core.spine.project_io import ProjectModifiedExternally, project_error

    result = project_error(ProjectModifiedExternally(path, 10.0, 20.0))
    assert result == {
        "code": "project_modified_externally",
        "path": str(path),
        "expected_mtime": 10.0,
        "current_mtime": 20.0,
    }

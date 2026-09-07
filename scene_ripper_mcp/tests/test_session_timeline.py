"""Timeline edits use stable sequence IDs and survive save/undo across calls."""

import json

import pytest
import pytest_asyncio
from types import SimpleNamespace
from scene_ripper_mcp.project_sessions import SessionRuntime

from core.project import Project
from models.clip import Source, Clip
from models.sequence import Sequence, Track
from scene_ripper_mcp.tests.test_project_sessions import opened
from scene_ripper_mcp.tools import sessions as tools


@pytest_asyncio.fixture
async def context():
    runtime = SessionRuntime()
    ctx = SimpleNamespace(
        request_context=SimpleNamespace(lifespan_context={"project_sessions": runtime})
    )
    yield ctx
    await runtime.shutdown()


@pytest.fixture
def path(tmp_path):
    path = tmp_path / "timeline.sceneripper"
    source = Source(
        id="source", file_path=tmp_path / "offline.mp4", fps=30, duration_seconds=30
    )
    clips = [
        Clip(
            id=f"clip{i}",
            source_id=source.id,
            start_frame=100 + i * 100,
            end_frame=140 + i * 100,
        )
        for i in range(3)
    ]
    project = Project(
        sources=[source],
        clips=clips,
        sequences=[
            Sequence(name="Active"),
            Sequence(name="Target", tracks=[Track(), Track()]),
        ],
    )
    assert project.save(path)
    return path


async def target(path, context):
    session_id = await opened(path, context)
    state = json.loads(await tools.get_project_session(session_id, context))
    return session_id, state["sequences"][1]["id"]


@pytest.mark.asyncio
async def test_insert_trim_reorder_remove_and_undo_on_inactive_track(path, context):
    sid, sequence_id = await target(path, context)
    inserted = json.loads(
        await tools.insert_session_clips(
            sid, sequence_id, ["clip0", "clip1"], context, track_index=1, start_frame=20
        )
    )
    assert inserted["success"], inserted
    ids = inserted["added"]
    project = Project.load(path)
    assert not project.sequence.get_all_clips()
    entries = project.sequences[1].tracks[1].clips
    assert [(c.start_frame, c.in_point, c.out_point) for c in entries] == [
        (20, 100, 140),
        (60, 200, 240),
    ]
    assert json.loads(
        await tools.edit_session_timeline_clip(
            sid,
            sequence_id,
            ids[0],
            {"in_point": 110, "out_point": 130, "hflip": True},
            context,
        )
    )["success"]
    assert json.loads(
        await tools.reorder_session_clips(
            sid, sequence_id, ids[::-1], context, track_index=1
        )
    )["success"]
    project = Project.load(path)
    assert [(c.id, c.start_frame) for c in project.sequences[1].tracks[1].clips] == [
        (ids[1], 0),
        (ids[0], 40),
    ]
    assert json.loads(await tools.undo_project_session(sid, context))["success"]
    assert [c.start_frame for c in Project.load(path).sequences[1].tracks[1].clips] == [
        20,
        60,
    ]
    assert json.loads(
        await tools.remove_session_clips(sid, sequence_id, [ids[0]], context)
    )["success"]
    assert [c.start_frame for c in Project.load(path).sequences[1].tracks[1].clips] == [
        60
    ]
    assert json.loads(await tools.undo_project_session(sid, context))["success"]
    restored = Project.load(path).sequences[1].tracks[1].clips[0]
    assert restored.id == ids[0] and restored.in_point == 110 and restored.hflip
    assert json.loads(await tools.clear_session_timeline(sid, sequence_id, context))[
        "success"
    ]
    assert not Project.load(path).sequences[1].get_all_clips()
    assert json.loads(await tools.undo_project_session(sid, context))["success"]
    assert len(Project.load(path).sequences[1].tracks[1].clips) == 2


@pytest.mark.asyncio
async def test_invalid_batch_is_atomic_and_keeps_history(path, context):
    sid, sequence_id = await target(path, context)
    inserted = json.loads(
        await tools.insert_session_clips(sid, sequence_id, ["clip0"], context)
    )
    assert inserted["success"]
    expected = path.read_bytes()
    result = json.loads(
        await tools.insert_session_clips(
            sid, sequence_id, ["clip1", "missing"], context
        )
    )
    assert not result["success"]
    assert path.read_bytes() == expected
    assert json.loads(await tools.get_project_session(sid, context))["can_undo"]
    assert json.loads(await tools.undo_project_session(sid, context))["success"]
    assert not Project.load(path).sequences[1].get_all_clips()


@pytest.mark.asyncio
async def test_query_ids_and_append_respect_overlapping_clip_ends(path, context):
    sid, sequence_id = await target(path, context)
    await tools.insert_session_clips(
        sid, sequence_id, ["clip0"], context, start_frame=100
    )
    await tools.insert_session_clips(
        sid, sequence_id, ["clip1"], context, start_frame=0
    )
    result = json.loads(
        await tools.insert_session_clips(sid, sequence_id, ["clip2"], context)
    )
    assert result["success"]
    state = json.loads(await tools.get_session_timeline(sid, sequence_id, context))
    assert state["sequence"]["tracks"][0]["clips"][-1]["start_frame"] == 140


@pytest.mark.asyncio
@pytest.mark.parametrize("track_index", [-1, 2])
async def test_invalid_track_does_not_write(path, context, track_index):
    sid, sequence_id = await target(path, context)
    expected = path.read_bytes()
    result = json.loads(
        await tools.insert_session_clips(
            sid, sequence_id, ["clip0"], context, track_index=track_index
        )
    )
    assert not result["success"]
    assert path.read_bytes() == expected


@pytest.mark.asyncio
async def test_path_tools_join_retained_history_without_replaying_shuffle(
    path, context, monkeypatch
):
    from scene_ripper_mcp.tools import sequence as legacy
    import random

    sid = await opened(path, context)
    sequence_id = Project.load(path).sequence.id
    inserted = json.loads(
        await tools.insert_session_clips(sid, sequence_id, ["clip0", "clip1"], context)
    )
    ids = inserted["added"]
    shuffled = []

    def reverse_once(entries):
        shuffled.append(True)
        entries.reverse()

    monkeypatch.setattr(random, "shuffle", reverse_once)
    result = json.loads(
        await legacy.shuffle_sequence(str(path), method="random", ctx=context)
    )
    assert result["success"] and result["new_order"] == ids[::-1]
    assert json.loads(await tools.undo_project_session(sid, context))["success"]
    assert json.loads(await tools.redo_project_session(sid, context))["success"]
    assert shuffled == [True]
    result = json.loads(await legacy.reorder_sequence(str(path), ids, ctx=context))
    assert result["success"] and result["new_order"] == ids
    assert (
        json.loads(await legacy.remove_from_sequence(str(path), [ids[0]], ctx=context))[
            "clips_removed"
        ]
        == 1
    )
    assert json.loads(await tools.undo_project_session(sid, context))["success"]
    assert (
        json.loads(await legacy.clear_sequence(str(path), ctx=context))["clips_removed"]
        == 2
    )
    assert json.loads(await tools.undo_project_session(sid, context))["success"]
    assert len(Project.load(path).sequence.get_all_clips()) == 2


@pytest.mark.asyncio
async def test_legacy_reorder_preserves_unknown_id_compatibility(path, context):
    from scene_ripper_mcp.tools.sequence import reorder_sequence

    sid = await opened(path, context)
    sequence_id = Project.load(path).sequence.id
    result = json.loads(
        await tools.insert_session_clips(sid, sequence_id, ["clip0", "clip1"], context)
    )
    ids = result["added"]
    result = json.loads(
        await reorder_sequence(str(path), ["unknown", ids[1]], ctx=context)
    )
    assert result["success"]
    assert result["new_order"] == ids[::-1]


@pytest.mark.asyncio
async def test_invalid_trim_and_duplicate_reorder_preserve_saved_history(path, context):
    sid, sequence_id = await target(path, context)
    inserted = json.loads(
        await tools.insert_session_clips(sid, sequence_id, ["clip0"], context)
    )
    clip_id = inserted["added"][0]
    expected = path.read_bytes()
    for changes in ({"in_point": 5000}, {"id": "replacement"}, {"hflip": "yes"}):
        result = json.loads(
            await tools.edit_session_timeline_clip(
                sid, sequence_id, clip_id, changes, context
            )
        )
        assert not result["success"]
        assert path.read_bytes() == expected
    result = json.loads(
        await tools.reorder_session_clips(sid, sequence_id, [clip_id, clip_id], context)
    )
    assert not result["success"]
    assert path.read_bytes() == expected
    assert json.loads(await tools.undo_project_session(sid, context))["success"]
    assert not Project.load(path).sequences[1].get_all_clips()

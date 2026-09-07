"""Retained MCP sessions save edits, preserve history, and reject stale undo."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio

from core.project import Project
from core.project_lock import project_writer
from core.spine.project_sessions import ProjectSessions
from scene_ripper_mcp.project_sessions import SessionRuntime
from scene_ripper_mcp.tools import sessions as tools


@pytest.fixture
def path(tmp_path):
    path = tmp_path / "session.sceneripper"
    assert Project.new().save(path)
    return path


@pytest_asyncio.fixture
async def context():
    runtime = SessionRuntime()
    context = SimpleNamespace(
        request_context=SimpleNamespace(lifespan_context={"project_sessions": runtime})
    )
    yield context
    await runtime.shutdown()


async def opened(path, context):
    result = json.loads(await tools.open_project_session(str(path), context))
    assert result["success"], result
    return result["session_id"]


@pytest.mark.asyncio
async def test_edits_undo_redo_and_reopen_share_history(path, context):
    session_id = await opened(path, context)
    result = json.loads(await tools.create_session_sequence(session_id, "New", context))
    assert result["success"] and result["session"]["can_undo"]
    sequence_id = result["sequence_id"]
    assert len(Project.load(path).sequences) == 2
    assert await opened(path, context) == session_id
    assert json.loads(
        await tools.rename_session_sequence(session_id, sequence_id, "Renamed", context)
    )["success"]
    assert Project.load(path).sequences[-1].name == "Renamed"
    assert json.loads(await tools.undo_project_session(session_id, context))["success"]
    assert Project.load(path).sequences[-1].name == "New"
    assert json.loads(await tools.redo_project_session(session_id, context))["success"]
    assert Project.load(path).sequences[-1].name == "Renamed"
    assert json.loads(
        await tools.delete_session_sequence(session_id, sequence_id, context)
    )["success"]
    assert len(Project.load(path).sequences) == 1
    assert json.loads(await tools.undo_project_session(session_id, context))["success"]
    assert Project.load(path).sequences[-1].id == sequence_id


@pytest.mark.asyncio
async def test_external_save_resets_history_before_undo(path, context):
    session_id = await opened(path, context)
    await tools.create_session_sequence(session_id, "New", context)
    external = Project.load(path)
    external.metadata.name = "External"
    assert external.save(path)
    expected = path.read_bytes()
    result = json.loads(await tools.undo_project_session(session_id, context))
    assert not result["success"] and result["history_reset"]
    assert path.read_bytes() == expected
    state = json.loads(await tools.get_project_session(session_id, context))
    assert not state["can_undo"] and not state["can_redo"]
    result = json.loads(
        await tools.create_session_sequence(session_id, "After reload", context)
    )
    assert result["success"]
    assert Project.load(path).metadata.name == "External"


@pytest.mark.asyncio
async def test_busy_operation_retains_history_without_edit(path, context):
    session_id = await opened(path, context)
    await tools.create_session_sequence(session_id, "New", context)
    with project_writer(path):
        result = json.loads(await tools.undo_project_session(session_id, context))
    assert not result["success"] and result["error"]["code"] == "project_busy"
    assert json.loads(await tools.get_project_session(session_id, context))["can_undo"]


@pytest.mark.asyncio
async def test_concurrent_calls_have_one_owner_and_durable_history(path, context):
    session_id = await opened(path, context)
    results = await asyncio.gather(
        *[
            tools.create_session_sequence(session_id, f"Sequence {i}", context)
            for i in range(6)
        ]
    )
    assert all(json.loads(result)["success"] for result in results)
    assert len(Project.load(path).sequences) == 7
    for _ in results:
        assert json.loads(await tools.undo_project_session(session_id, context))[
            "success"
        ]
    assert len(Project.load(path).sequences) == 1


@pytest.mark.asyncio
async def test_close_discards_history_and_reopen_returns_new_id(path, context):
    session_id = await opened(path, context)
    await tools.create_session_sequence(session_id, "New", context)
    assert json.loads(await tools.close_project_session(session_id, context))["success"]
    assert not json.loads(await tools.undo_project_session(session_id, context))[
        "success"
    ]
    new_id = await opened(path, context)
    assert new_id != session_id
    assert not json.loads(await tools.get_project_session(new_id, context))["can_undo"]
    assert len(Project.load(path).sequences) == 2


@pytest.mark.asyncio
async def test_failed_save_discards_unpublished_edit(path, context, monkeypatch):
    session_id = await opened(path, context)
    original = Project.save
    monkeypatch.setattr(Project, "save", lambda *a, **k: False)
    result = json.loads(
        await tools.create_session_sequence(session_id, "Lost", context)
    )
    assert not result["success"]
    monkeypatch.setattr(Project, "save", original)
    state = json.loads(await tools.get_project_session(session_id, context))
    assert state["history_reset"] and not state["can_undo"]
    assert len(state["sequences"]) == len(Project.load(path).sequences) == 1


@pytest.mark.asyncio
async def test_shutdown_rejects_calls_and_closes_models(path):
    runtime = SessionRuntime()
    state = await runtime.call(ProjectSessions.open, path)
    await runtime.shutdown()
    await runtime.shutdown()
    with pytest.raises(RuntimeError, match="closed"):
        await runtime.call(ProjectSessions.inspect, state["session_id"])
    with project_writer(path):
        pass


@pytest.mark.asyncio
async def test_tool_registration_exposes_session_arguments():
    from scene_ripper_mcp.server import mcp

    registered = {tool.name: tool for tool in await mcp.list_tools()}
    for name in (
        "open_project_session",
        "get_project_session",
        "close_project_session",
        "create_session_sequence",
        "rename_session_sequence",
        "delete_session_sequence",
        "undo_project_session",
        "redo_project_session",
        "get_session_timeline",
        "insert_session_clips",
        "remove_session_clips",
        "reorder_session_clips",
        "edit_session_timeline_clip",
        "clear_session_timeline",
        "set_session_clips_disabled",
        "update_session_clip",
    ):
        assert name in registered
        properties = registered[name].inputSchema["properties"]
        assert "ctx" not in properties
        assert (
            "project_path" if name == "open_project_session" else "session_id"
        ) in properties


@pytest.mark.asyncio
async def test_newer_schema_is_inspectable_but_not_editable(path, context):
    data = json.loads(path.read_text())
    data["version"] = "99.0"
    path.write_text(json.dumps(data))
    expected = path.read_bytes()
    session_id = await opened(path, context)
    assert json.loads(await tools.get_project_session(session_id, context))["read_only"]
    result = json.loads(await tools.create_session_sequence(session_id, "No", context))
    assert not result["success"]
    assert path.read_bytes() == expected


@pytest.mark.asyncio
async def test_alias_open_reuses_session_history(path, context, tmp_path):
    session_id = await opened(path, context)
    alias = tmp_path / "alias.sceneripper"
    alias.symlink_to(path)
    assert await opened(alias, context) == session_id


@pytest.mark.asyncio
async def test_hardlink_open_reuses_session(path, context, tmp_path):
    session_id = await opened(path, context)
    alias = tmp_path / "hardlink.sceneripper"
    alias.hardlink_to(path)
    assert await opened(alias, context) == session_id


@pytest.mark.asyncio
async def test_cancelled_running_request_finishes_before_shutdown(path):
    from threading import Event
    from core.spine.sequences import create_sequence

    runtime = SessionRuntime()
    started = Event()
    release = Event()

    def operation(sessions):
        session_id = sessions.open(path)["session_id"]
        started.set()
        if not release.wait(5):
            raise RuntimeError("test did not release edit")
        return sessions.edit(
            session_id, lambda project: create_sequence(project, "Saved")
        )

    task = asyncio.create_task(runtime.call(operation))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await runtime.shutdown()
    assert Project.load(path).sequences[-1].name == "Saved"


@pytest.mark.asyncio
async def test_session_rejects_path_retargeted_after_open(path, context, tmp_path):
    session_id = await opened(path, context)
    target = tmp_path / "other.sceneripper"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    expected = target.read_bytes()
    result = json.loads(
        await tools.create_session_sequence(session_id, "Wrong file", context)
    )
    assert not result["success"]
    assert target.read_bytes() == expected

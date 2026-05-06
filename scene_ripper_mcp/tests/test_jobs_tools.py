"""Tests for the generic job-management MCP tools.

Covers ``get_job_status``, ``get_job_result``, ``cancel_job``, ``list_jobs``,
and ``purge_old_jobs``. The store/runtime are tested separately in
``test_jobs_store.py`` / ``test_jobs_runtime.py``; these tests focus on
the MCP-tool wrappers and their JSON contract (R28 information-disclosure
discipline, R26 snake_case fields).
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from scene_ripper_mcp.jobs import JobRuntime, JobStore
from scene_ripper_mcp.jobs.store import (
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
)
from scene_ripper_mcp.tools.jobs import (
    cancel_job,
    get_job_result,
    get_job_status,
    list_jobs,
    purge_old_jobs,
)


@pytest.fixture
def lifespan_ctx(tmp_path):
    """Build a Context-shaped object with the lifespan dict the tools expect."""
    store = JobStore(tmp_path / "jobs.db")
    runtime = JobRuntime(store, max_workers=4)
    request_context = SimpleNamespace(
        lifespan_context={
            "job_store": store,
            "job_runtime": runtime,
        }
    )
    ctx = AsyncMock()
    ctx.request_context = request_context
    yield ctx, store, runtime
    runtime.shutdown(wait=True)


def _wait_for_status(store, task_id, expected, timeout=5.0):
    deadline = time.monotonic() + timeout
    targets = {expected} if isinstance(expected, str) else expected
    while time.monotonic() < deadline:
        status = store.get(task_id).status
        if status in targets:
            return status
        time.sleep(0.02)
    raise AssertionError(f"timeout waiting for {targets}; last={status!r}")


@pytest.mark.asyncio
async def test_get_job_status_unknown_id(lifespan_ctx):
    ctx, _, _ = lifespan_ctx
    out = json.loads(await get_job_status(task_id="nope", ctx=ctx))
    assert out["success"] is False
    assert out["error"]["code"] == "job_not_found"


@pytest.mark.asyncio
async def test_get_job_status_excludes_payload(lifespan_ctx):
    ctx, store, _ = lifespan_ctx
    row = store.insert(
        kind="x",
        args={"secret": "do-not-leak"},
        project_path="/tmp/p.sceneripper",
    )
    out = json.loads(await get_job_status(task_id=row.id, ctx=ctx))
    assert out["success"] is True
    assert out["task_id"] == row.id
    # Sensitive payload columns must NOT appear in get_job_status (R28).
    assert "args_json" not in out
    assert "result_json" not in out
    assert "error" not in out


@pytest.mark.asyncio
async def test_get_job_result_not_terminal(lifespan_ctx):
    ctx, store, _ = lifespan_ctx
    row = store.insert(kind="x", args={}, status=STATUS_RUNNING)
    out = json.loads(await get_job_result(task_id=row.id, ctx=ctx))
    assert out["success"] is False
    assert out["error"]["code"] == "not_terminal"
    assert out["error"]["status"] == STATUS_RUNNING


@pytest.mark.asyncio
async def test_get_job_result_completed_returns_payload(lifespan_ctx):
    ctx, store, runtime = lifespan_ctx

    def run(progress, cancel):
        return {"value": 99}

    submit = runtime.submit(kind="t", args={}, run=run)
    _wait_for_status(store, submit["task_id"], STATUS_COMPLETED)

    out = json.loads(await get_job_result(task_id=submit["task_id"], ctx=ctx))
    assert out["success"] is True
    assert out["result"] == {"value": 99}
    assert out["status"] == STATUS_COMPLETED


@pytest.mark.asyncio
async def test_get_job_result_failed_surfaces_sanitized_error(lifespan_ctx):
    ctx, store, runtime = lifespan_ctx

    def run(progress, cancel):
        raise RuntimeError("kaboom")

    submit = runtime.submit(kind="t", args={}, run=run)
    _wait_for_status(store, submit["task_id"], STATUS_FAILED)

    out = json.loads(await get_job_result(task_id=submit["task_id"], ctx=ctx))
    assert out["success"] is False
    assert out["status"] == STATUS_FAILED
    assert out["error"]["code"] == "job_failed"
    assert "kaboom" in out["error"]["message"]
    # Sanitized — no absolute paths.
    assert "/Users/" not in out["error"]["message"]


@pytest.mark.asyncio
async def test_cancel_job_running(lifespan_ctx):
    ctx, store, runtime = lifespan_ctx
    started = threading.Event()

    def run(progress, cancel):
        started.set()
        for _ in range(200):
            if cancel.is_set():
                return {}
            time.sleep(0.01)
        return {}

    submit = runtime.submit(kind="t", args={}, run=run)
    started.wait(timeout=2.0)

    out = json.loads(await cancel_job(task_id=submit["task_id"], ctx=ctx))
    assert out["success"] is True
    assert out["ok"] is True

    _wait_for_status(store, submit["task_id"], STATUS_CANCELLED)


@pytest.mark.asyncio
async def test_cancel_job_already_terminal(lifespan_ctx):
    ctx, store, _ = lifespan_ctx
    row = store.insert(kind="x", args={}, status=STATUS_COMPLETED)
    out = json.loads(await cancel_job(task_id=row.id, ctx=ctx))
    assert out["success"] is False
    assert out["error"]["code"] == "already_terminal"


@pytest.mark.asyncio
async def test_cancel_job_unknown(lifespan_ctx):
    ctx, _, _ = lifespan_ctx
    out = json.loads(await cancel_job(task_id="not-a-real-id", ctx=ctx))
    assert out["success"] is False
    assert out["error"]["code"] == "job_not_found"


@pytest.mark.asyncio
async def test_list_jobs_returns_safe_projection(lifespan_ctx):
    ctx, store, _ = lifespan_ctx
    a = store.insert(kind="x", args={"a": 1}, status=STATUS_QUEUED)
    b = store.insert(kind="y", args={"b": 2}, status=STATUS_COMPLETED)
    out = json.loads(await list_jobs(ctx=ctx))
    assert out["success"] is True
    assert out["count"] == 2
    ids = {j["task_id"] for j in out["jobs"]}
    assert ids == {a.id, b.id}
    # Safe projection — no payload columns.
    for j in out["jobs"]:
        assert "args_json" not in j
        assert "result_json" not in j
        assert "error" not in j


@pytest.mark.asyncio
async def test_list_jobs_status_filter(lifespan_ctx):
    ctx, store, _ = lifespan_ctx
    queued = store.insert(kind="x", args={}, status=STATUS_QUEUED)
    completed = store.insert(kind="x", args={}, status=STATUS_COMPLETED)

    out = json.loads(
        await list_jobs(status_filter=[STATUS_COMPLETED], ctx=ctx)
    )
    assert out["count"] == 1
    assert out["jobs"][0]["task_id"] == completed.id


@pytest.mark.asyncio
async def test_purge_old_jobs_explicit_only(lifespan_ctx):
    ctx, store, _ = lifespan_ctx
    # Old completed row.
    old = store.insert(kind="x", args={}, status=STATUS_COMPLETED)
    store.update_status(old.id, STATUS_COMPLETED, terminal=True)
    long_ago = time.time() - 90 * 86400
    import sqlite3

    with sqlite3.connect(str(store.db_path)) as conn:
        conn.execute(
            "UPDATE jobs SET finished_at = ? WHERE id = ?",
            (long_ago, old.id),
        )
        conn.commit()

    out = json.loads(await purge_old_jobs(days=30, ctx=ctx))
    assert out["success"] is True
    assert out["deleted_count"] == 1


@pytest.mark.asyncio
async def test_purge_old_jobs_negative_days_rejected(lifespan_ctx):
    ctx, _, _ = lifespan_ctx
    out = json.loads(await purge_old_jobs(days=-5, ctx=ctx))
    assert out["success"] is False
    assert out["error"]["code"] == "invalid_days"

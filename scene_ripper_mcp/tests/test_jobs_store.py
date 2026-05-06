"""Tests for the SQLite-backed jobs store."""

from __future__ import annotations

import os
import stat
import sqlite3
import sys
import time

import pytest

from scene_ripper_mcp.jobs.store import (
    JobNotFoundError,
    JobStore,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_CRASHED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    sanitize_traceback,
)


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "jobs.db"
    return JobStore(db)


def test_store_creates_db_with_mode_0o600(tmp_path):
    db = tmp_path / "jobs.db"
    JobStore(db)
    assert db.exists()
    if sys.platform != "win32":
        # On POSIX, the DB file should be readable/writable only by the owner.
        mode = stat.S_IMODE(os.stat(db).st_mode)
        assert mode == 0o600


def test_insert_and_get_round_trip(store):
    row = store.insert(
        kind="analyze_colors",
        args={"num_colors": 5},
        project_path="/tmp/proj.sceneripper",
        project_mtime_at_start=12345.0,
    )
    assert row.id
    assert row.status == STATUS_QUEUED
    assert row.args == {"num_colors": 5}

    fetched = store.get(row.id)
    assert fetched.id == row.id
    assert fetched.kind == "analyze_colors"
    assert fetched.args == {"num_colors": 5}


def test_get_unknown_id_raises(store):
    with pytest.raises(JobNotFoundError):
        store.get("not-a-real-id")


def test_update_status_terminal_sets_finished_at(store):
    row = store.insert(kind="x", args={})
    before = time.time()
    store.update_status(
        row.id,
        STATUS_COMPLETED,
        progress=1.0,
        result={"ok": True},
        terminal=True,
    )
    fetched = store.get(row.id)
    assert fetched.status == STATUS_COMPLETED
    assert fetched.progress == 1.0
    assert fetched.result == {"ok": True}
    assert fetched.finished_at is not None
    assert fetched.finished_at >= before


def test_idempotency_unique_constraint(store):
    store.insert(
        kind="batch",
        args={},
        project_path="/tmp/p.sceneripper",
        idempotency_key="abc",
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.insert(
            kind="batch",
            args={},
            project_path="/tmp/p.sceneripper",
            idempotency_key="abc",
        )


def test_idempotency_scoped_per_project(store):
    """Same key against different projects does not collide (R21)."""
    store.insert(
        kind="batch",
        args={},
        project_path="/tmp/A.sceneripper",
        idempotency_key="abc",
    )
    # Different project_path — should not collide.
    store.insert(
        kind="batch",
        args={},
        project_path="/tmp/B.sceneripper",
        idempotency_key="abc",
    )


def test_idempotency_null_key_allows_duplicates(store):
    """Rows with NULL idempotency_key skip the unique constraint."""
    store.insert(kind="x", args={}, project_path="/tmp/p.sceneripper")
    # No collision because the partial unique index excludes NULL keys.
    store.insert(kind="x", args={}, project_path="/tmp/p.sceneripper")


def test_find_by_idempotency_returns_match(store):
    inserted = store.insert(
        kind="batch",
        args={},
        project_path="/tmp/p.sceneripper",
        idempotency_key="xyz",
    )
    found = store.find_by_idempotency(
        kind="batch", project_path="/tmp/p.sceneripper", idempotency_key="xyz"
    )
    assert found is not None
    assert found.id == inserted.id

    missing = store.find_by_idempotency(
        kind="batch", project_path="/tmp/p.sceneripper", idempotency_key="nope"
    )
    assert missing is None


def test_mark_running_jobs_as_crashed(store):
    """Boot sweep flips running/cancelling rows to crashed (R18)."""
    a = store.insert(kind="x", args={}, status=STATUS_RUNNING)
    b = store.insert(kind="x", args={}, status=STATUS_QUEUED)
    c = store.insert(kind="x", args={}, status=STATUS_COMPLETED)
    swept = store.mark_running_jobs_as_crashed()
    assert swept == 1
    assert store.get(a.id).status == STATUS_CRASHED
    assert store.get(a.id).error == "server restarted while in flight"
    assert store.get(a.id).finished_at is not None
    assert store.get(b.id).status == STATUS_QUEUED  # untouched
    assert store.get(c.id).status == STATUS_COMPLETED  # untouched


def test_purge_old_jobs_keeps_running_and_recent(store):
    # Old completed row.
    old = store.insert(kind="x", args={}, status=STATUS_COMPLETED)
    store.update_status(old.id, STATUS_COMPLETED, terminal=True)
    # Backdate finished_at far into the past.
    long_ago = time.time() - 90 * 86400
    with sqlite3.connect(str(store.db_path)) as conn:
        conn.execute(
            "UPDATE jobs SET finished_at = ? WHERE id = ?",
            (long_ago, old.id),
        )
        conn.commit()

    # Recent failed row.
    recent = store.insert(kind="x", args={}, status=STATUS_FAILED)
    store.update_status(recent.id, STATUS_FAILED, terminal=True)

    # Running row — must not be touched.
    running = store.insert(kind="x", args={}, status=STATUS_RUNNING)

    deleted = store.purge_old_jobs(days=30)
    assert deleted == 1
    with pytest.raises(JobNotFoundError):
        store.get(old.id)
    assert store.get(recent.id).status == STATUS_FAILED
    assert store.get(running.id).status == STATUS_RUNNING


def test_list_filter_by_status(store):
    a = store.insert(kind="x", args={}, status=STATUS_QUEUED)
    b = store.insert(kind="x", args={}, status=STATUS_RUNNING)
    c = store.insert(kind="x", args={}, status=STATUS_COMPLETED)

    queued_or_running = store.list(
        status_filter=[STATUS_QUEUED, STATUS_RUNNING]
    )
    ids = {row.id for row in queued_or_running}
    assert a.id in ids
    assert b.id in ids
    assert c.id not in ids


def test_safe_projection_excludes_payload(store):
    row = store.insert(
        kind="x",
        args={"secret": "do not expose"},
        project_path="/tmp/p.sceneripper",
    )
    store.update_status(
        row.id,
        STATUS_FAILED,
        error="long traceback that should not surface in list_jobs",
        terminal=True,
    )
    fetched = store.get(row.id)
    safe = fetched.to_safe_projection()
    assert "args_json" not in safe
    assert "result_json" not in safe
    assert "error" not in safe
    assert safe["task_id"] == row.id
    assert safe["status"] == STATUS_FAILED


def test_sanitize_traceback_strips_paths_and_caps_length():
    try:
        # Generate a realistic traceback.
        def deeply_nested():
            raise ValueError("boom")

        deeply_nested()
    except ValueError as exc:
        sanitized = sanitize_traceback(exc)
    # No absolute paths.
    assert "/Users/" not in sanitized
    assert "/home/" not in sanitized
    # Includes the type and message.
    assert "ValueError: boom" in sanitized
    # Bounded.
    assert len(sanitized.encode("utf-8")) <= 4096

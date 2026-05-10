"""Tests for the in-memory job runtime + cancellation protocol."""

from __future__ import annotations

import threading
import time

import pytest

from scene_ripper_mcp.jobs.runtime import (
    InvalidIdempotencyKeyError,
    JobRuntime,
    DEFAULT_POLL_INTERVAL_SECONDS,
)
from scene_ripper_mcp.jobs.store import (
    JobStore,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
)


@pytest.fixture
def runtime(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    rt = JobRuntime(store, max_workers=4)
    yield rt
    rt.shutdown(wait=True)


def _wait_for_status(
    runtime: JobRuntime,
    task_id: str,
    expected: set[str] | str,
    timeout: float = 5.0,
) -> str:
    """Poll the store until the row reaches one of ``expected`` or timeout."""
    deadline = time.monotonic() + timeout
    targets = {expected} if isinstance(expected, str) else expected
    while time.monotonic() < deadline:
        status = runtime.store.get(task_id).status
        if status in targets:
            return status
        time.sleep(0.02)
    raise AssertionError(
        f"timeout waiting for {targets}; last status={status!r}"
    )


def test_submit_runs_to_completion(runtime):
    def run(progress, cancel):
        progress(0.5, "halfway")
        return {"value": 42}

    result = runtime.submit(kind="test", args={}, run=run)
    assert result["status"] == STATUS_QUEUED
    assert result["poll_interval"] == DEFAULT_POLL_INTERVAL_SECONDS

    final = _wait_for_status(runtime, result["task_id"], STATUS_COMPLETED)
    assert final == STATUS_COMPLETED
    row = runtime.store.get(result["task_id"])
    assert row.result == {"value": 42}
    assert row.progress == 1.0


def test_submit_records_failed_with_sanitized_traceback(runtime):
    def run(progress, cancel):
        raise ValueError("boom")

    result = runtime.submit(kind="test", args={}, run=run)
    _wait_for_status(runtime, result["task_id"], STATUS_FAILED)
    row = runtime.store.get(result["task_id"])
    assert row.status == STATUS_FAILED
    assert row.error is not None
    assert "ValueError: boom" in row.error
    # Sanitization: no absolute paths.
    assert "/Users/" not in row.error


def test_cancellation_observes_event(runtime):
    started = threading.Event()
    saw_cancel = threading.Event()

    def run(progress, cancel):
        started.set()
        for _ in range(200):
            if cancel.is_set():
                saw_cancel.set()
                return {}
            time.sleep(0.01)
        return {}

    result = runtime.submit(kind="test", args={}, run=run)
    started.wait(timeout=2.0)
    runtime.cancel(result["task_id"])
    _wait_for_status(runtime, result["task_id"], STATUS_CANCELLED)
    assert saw_cancel.is_set()


def test_cancel_unknown_id_returns_false(runtime):
    assert runtime.cancel("not-a-real-id") is False


def test_idempotency_key_too_long(runtime):
    def run(progress, cancel):
        return {}

    with pytest.raises(InvalidIdempotencyKeyError):
        runtime.submit(
            kind="test",
            args={},
            run=run,
            idempotency_key="x" * 256,
        )


def test_idempotency_completed_returns_existing_task_id(runtime):
    def run(progress, cancel):
        return {"once": True}

    first = runtime.submit(
        kind="test", args={}, run=run, idempotency_key="abc"
    )
    _wait_for_status(runtime, first["task_id"], STATUS_COMPLETED)

    # Second submit with the same key on a completed row returns the same id.
    second = runtime.submit(
        kind="test", args={}, run=run, idempotency_key="abc"
    )
    assert second["task_id"] == first["task_id"]
    assert second["status"] == STATUS_COMPLETED


def test_idempotency_failed_replaces_row(runtime):
    def run_fail(progress, cancel):
        raise RuntimeError("nope")

    first = runtime.submit(
        kind="test", args={}, run=run_fail, idempotency_key="abc"
    )
    _wait_for_status(runtime, first["task_id"], STATUS_FAILED)

    def run_ok(progress, cancel):
        return {"ok": True}

    second = runtime.submit(
        kind="test", args={}, run=run_ok, idempotency_key="abc"
    )
    # Different task_id — the failed row was deleted, a new row spawned.
    assert second["task_id"] != first["task_id"]
    _wait_for_status(runtime, second["task_id"], STATUS_COMPLETED)


def test_idempotency_scoped_per_project(runtime):
    def run(progress, cancel):
        return {}

    a = runtime.submit(
        kind="test",
        args={},
        run=run,
        project_path="/tmp/A.sceneripper",
        idempotency_key="same",
    )
    b = runtime.submit(
        kind="test",
        args={},
        run=run,
        project_path="/tmp/B.sceneripper",
        idempotency_key="same",
    )
    # Different project — different task ids despite shared key (R21).
    assert a["task_id"] != b["task_id"]


def test_per_project_mutex_serializes_same_project(runtime, tmp_path):
    """Same-project jobs run serially; different-project jobs in parallel."""
    project_a = tmp_path / "A.sceneripper"
    project_a.write_text("{}")

    started = []
    finished = []
    started_lock = threading.Lock()

    def run(progress, cancel):
        with started_lock:
            started.append(time.monotonic())
        time.sleep(0.2)
        with started_lock:
            finished.append(time.monotonic())
        return {}

    r1 = runtime.submit(
        kind="t", args={}, run=run, project_path=str(project_a)
    )
    r2 = runtime.submit(
        kind="t", args={}, run=run, project_path=str(project_a)
    )

    _wait_for_status(runtime, r1["task_id"], STATUS_COMPLETED, timeout=5.0)
    _wait_for_status(runtime, r2["task_id"], STATUS_COMPLETED, timeout=5.0)

    assert len(started) == 2
    assert len(finished) == 2
    # Second job started after the first finished (serial under mutex).
    assert started[1] >= finished[0] - 0.05  # tiny slack


def test_canonical_path_aliasing(runtime, tmp_path):
    """Tilde and resolved paths point to the same lock (R17)."""
    project = tmp_path / "p.sceneripper"
    project.write_text("{}")

    started = []
    finished = []
    started_lock = threading.Lock()

    def run(progress, cancel):
        with started_lock:
            started.append(time.monotonic())
        time.sleep(0.2)
        with started_lock:
            finished.append(time.monotonic())
        return {}

    # Both paths resolve to the same canonical absolute path.
    r1 = runtime.submit(
        kind="t", args={}, run=run, project_path=str(project)
    )
    r2 = runtime.submit(
        kind="t", args={}, run=run, project_path=str(project.resolve())
    )

    _wait_for_status(runtime, r1["task_id"], STATUS_COMPLETED, timeout=5.0)
    _wait_for_status(runtime, r2["task_id"], STATUS_COMPLETED, timeout=5.0)

    # If the lock were keyed on the raw string, they could overlap. The
    # canonical-path rule should serialise them.
    assert started[1] >= finished[0] - 0.05


def test_progress_callback_writes_after_debounce(runtime):
    started = threading.Event()

    def run(progress, cancel):
        started.set()
        # Generate >1 progress writes spaced past the debounce window.
        progress(0.1, "step 1")
        time.sleep(2.1)
        progress(0.5, "step 2")
        return {}

    result = runtime.submit(kind="test", args={}, run=run)
    started.wait(timeout=2.0)
    _wait_for_status(runtime, result["task_id"], STATUS_COMPLETED, timeout=10.0)
    row = runtime.store.get(result["task_id"])
    assert row.progress == 1.0
    # Final status_message should be the last one set ("step 2") or
    # "completed", whichever the worker flushed.
    assert row.status_message in {"step 2", "completed"}

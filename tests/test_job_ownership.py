"""Restart recovery distinguishes abandoned jobs from live runtime owners."""

import threading
import json
import subprocess
import sys
import time

import pytest

from core.jobs import JobRuntime, JobStore


def test_boot_sweep_preserves_live_owner_and_queued_work(tmp_path):
    path = tmp_path / "jobs.db"
    store = JobStore(path)
    runtime = JobRuntime(store, max_workers=1)
    entered, release = threading.Event(), threading.Event()

    def run(progress, cancel):
        entered.set()
        assert release.wait(5)
        return {}

    try:
        task = runtime.submit(kind="running", args={}, run=run)
        assert entered.wait(5)
        queued = runtime.submit(kind="queued", args={}, run=lambda p, c: {})
        assert JobStore(path).mark_running_jobs_as_crashed() == 0
        assert store.get(task["task_id"]).status == "running"
        assert store.get(queued["task_id"]).status == "queued"
    finally:
        release.set()
        runtime.shutdown()


def test_dead_process_recovers_running_and_queued_jobs(tmp_path):
    path = tmp_path / "jobs.db"
    ready = tmp_path / "ready.json"
    code = """
import json, sys, threading
from pathlib import Path
from core.jobs import JobRuntime, JobStore
runtime = JobRuntime(JobStore(Path(sys.argv[1])), max_workers=1)
entered = threading.Event()
def block(progress, cancel):
    entered.set()
    threading.Event().wait()
    return {}
running = runtime.submit(kind="running", args={}, run=block)
assert entered.wait(5)
queued = runtime.submit(kind="queued", args={}, run=lambda p, c: {})
ready = Path(sys.argv[2])
temporary = ready.with_suffix(".tmp")
temporary.write_text(json.dumps([running["task_id"], queued["task_id"]]))
temporary.replace(ready)
threading.Event().wait()
"""
    process = subprocess.Popen([sys.executable, "-c", code, str(path), str(ready)])
    try:
        deadline = time.monotonic() + 10
        while (
            not ready.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert ready.exists(), "child runtime did not start"
        store = JobStore(path)
        assert store.mark_running_jobs_as_crashed() == 0
    finally:
        process.kill()
        process.wait(timeout=5)
    ids = json.loads(ready.read_text())
    assert store.mark_running_jobs_as_crashed() == 2
    assert [store.get(task).status for task in ids] == ["crashed", "crashed"]
    assert store.mark_running_jobs_as_crashed() == 0


def test_closed_runtime_cannot_publish_unowned_queued_jobs(tmp_path):
    runtime = JobRuntime(JobStore(tmp_path / "jobs.db"))
    runtime.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        runtime.submit(kind="test", args={}, run=lambda p, c: {})
    assert runtime.store.list() == []


def test_shutdown_cannot_release_owner_during_submission(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs.db")
    runtime = JobRuntime(store)
    inserted, release_insert, release_job = (threading.Event() for _ in range(3))
    shutdown_entered, shutdown_returned = threading.Event(), threading.Event()
    original = store.insert

    def insert(**kwargs):
        row = original(**kwargs)
        inserted.set()
        assert release_insert.wait(5)
        return row

    monkeypatch.setattr(store, "insert", insert)
    errors = []

    def run(progress, cancel):
        assert release_job.wait(5)
        return {}

    def submit():
        try:
            runtime.submit(kind="test", args={}, run=run)
        except BaseException as exc:
            errors.append(exc)

    def shutdown():
        shutdown_entered.set()
        runtime.shutdown(wait=False)
        shutdown_returned.set()

    submit_thread = threading.Thread(target=submit)
    shutdown_thread = threading.Thread(target=shutdown)
    submit_thread.start()
    try:
        assert inserted.wait(5)
        shutdown_thread.start()
        assert shutdown_entered.wait(5)
        assert JobStore(store.db_path).mark_running_jobs_as_crashed() == 0
        assert not shutdown_returned.is_set()
        release_insert.set()
        submit_thread.join(5)
        shutdown_thread.join(5)
        assert not errors
        assert shutdown_returned.is_set()
        assert JobStore(store.db_path).mark_running_jobs_as_crashed() == 0
    finally:
        release_insert.set()
        release_job.set()
        submit_thread.join(5)
        if shutdown_thread.ident is not None:
            shutdown_thread.join(5)
        runtime.shutdown()
    assert store.list()[0].status == "completed"


def test_nonblocking_shutdown_retains_ownership_until_work_settles(tmp_path):
    path = tmp_path / "jobs.db"
    runtime = JobRuntime(JobStore(path))
    entered, release = threading.Event(), threading.Event()

    def run(progress, cancel):
        entered.set()
        assert release.wait(5)
        return {}

    try:
        task = runtime.submit(kind="running", args={}, run=run)
        assert entered.wait(5)
        runtime.shutdown(wait=False)
        assert JobStore(path).mark_running_jobs_as_crashed() == 0
        assert runtime.store.get(task["task_id"]).status == "running"
    finally:
        release.set()
        runtime.shutdown()
    # The worker that settles last must release the lease, not leave an
    # apparently live owner behind until the entire process exits.
    assert runtime._owner_lease is None
    from core.jobs.ownership import acquire_owner

    lease = acquire_owner(runtime._owner_id)
    lease.close()

"""Shared lifecycle preserves terminal decisions across transport adapters."""

from concurrent.futures import Future

import pytest

from core.jobs import JobRuntime, JobStore
from core.jobs.store import STATUS_COMPLETED, STATUS_RUNNING, STATUS_CANCELLED


def test_terminal_result_cannot_be_overwritten(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    row = store.insert(kind="test", args={})
    store.update_status(row.id, STATUS_COMPLETED, result={"kept": True}, terminal=True)
    before = store.get(row.id)
    store.update_status(row.id, STATUS_RUNNING, progress=0.1)
    store.update_status(row.id, STATUS_CANCELLED, terminal=True)
    after = store.get(row.id)
    assert after.status == STATUS_COMPLETED
    assert after.result == {"kept": True}
    assert after.finished_at == before.finished_at


def test_handle_exists_before_executor_can_start(tmp_path):
    runtime = JobRuntime(JobStore(tmp_path / "jobs.db"))
    runtime._executor.shutdown()

    class ImmediateExecutor:
        def submit(self, func, *args):
            future = Future()
            try:
                func(*args)
                future.set_result(None)
            except BaseException as exc:
                future.set_exception(exc)
            return future

        def shutdown(self, **kwargs):
            pass

    runtime._executor = ImmediateExecutor()
    result = runtime.submit(kind="fast", args={}, run=lambda p, c: {"value": 1})
    row = runtime.store.get(result["task_id"])
    assert row.status == STATUS_COMPLETED
    assert not runtime.is_handle_live(row.id)


def test_queued_cancellation_does_not_invoke_runner(tmp_path):
    import threading

    entered, release = threading.Event(), threading.Event()
    runtime = JobRuntime(JobStore(tmp_path / "jobs.db"), max_workers=1)

    def blocked(progress, cancel):
        entered.set()
        assert release.wait(5)
        return {}

    runtime.submit(kind="blocker", args={}, run=blocked)
    assert entered.wait(5)
    called = []
    queued = runtime.submit(
        kind="queued", args={}, run=lambda p, c: called.append(True) or {}
    )
    runtime.cancel(queued["task_id"])
    release.set()
    runtime.shutdown()
    assert not called
    assert runtime.store.get(queued["task_id"]).status == STATUS_CANCELLED


def test_legacy_migration_rows_and_import_aliases_survive(tmp_path):
    import sqlite3
    from pathlib import Path
    from scene_ripper_mcp.jobs import (
        JobRuntime as LegacyRuntime,
        JobStore as LegacyStore,
    )

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            Path("scene_ripper_mcp/jobs/migrations/0001_init.sql").read_text()
        )
        conn.execute(
            "INSERT INTO jobs (id,kind,status,args_json,created_at,updated_at) VALUES ('old','test','completed','{}',1,1)"
        )
    assert JobStore(path).get("old").status == STATUS_COMPLETED
    assert LegacyRuntime is JobRuntime and LegacyStore is JobStore


def test_core_jobs_import_without_mcp_or_qt(tmp_path):
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import core.jobs
assert not any(n == 'mcp' or n.startswith('mcp.') or n.startswith('scene_ripper_mcp') or n.startswith('PySide6') for n in sys.modules)
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_qt_adapter_emits_terminal_once_on_owner_thread(tmp_path):
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication
    from ui.workers.job_adapter import JobAdapter

    _app = QApplication.instance() or QApplication([])
    runtime = JobRuntime(JobStore(tmp_path / "qt.db"))
    adapter = JobAdapter(runtime)
    completed, settled = [], []
    owner = QThread.currentThread()
    adapter.completed.connect(
        lambda task_id, result: completed.append(
            (task_id, result, QThread.currentThread())
        )
    )
    adapter.settled.connect(settled.append)
    result = adapter.start(kind="qt", args={}, run=lambda p, c: {"value": 4})
    runtime.shutdown()
    adapter._poll()
    adapter._poll()
    assert completed == [(result["task_id"], {"value": 4}, owner)]
    assert settled == [result["task_id"]]


def test_rejected_submission_does_not_leave_queued_row(tmp_path):
    runtime = JobRuntime(JobStore(tmp_path / "closed.db"))
    runtime.shutdown()
    with pytest.raises(RuntimeError):
        runtime.submit(kind="closed", args={}, run=lambda p, c: {})
    assert runtime.store.list()[0].status == "failed"


def test_qt_adapter_settles_if_completed_history_was_purged(tmp_path):
    from PySide6.QtWidgets import QApplication
    from ui.workers.job_adapter import JobAdapter

    _app = QApplication.instance() or QApplication([])
    runtime = JobRuntime(JobStore(tmp_path / "purged.db"))
    adapter = JobAdapter(runtime)
    failed = []
    adapter.failed.connect(lambda task_id, message: failed.append(task_id))
    result = adapter.start(kind="purged", args={}, run=lambda p, c: {})
    runtime.shutdown()
    runtime.store.delete(result["task_id"])
    adapter._poll()
    adapter._poll()
    assert failed == [result["task_id"]]


def test_failed_work_result_is_not_reported_completed(tmp_path):
    runtime = JobRuntime(JobStore(tmp_path / "error.db"))
    result = runtime.submit(
        kind="error",
        args={},
        run=lambda p, c: {"success": False, "error": "cannot process"},
    )
    runtime.shutdown()
    row = runtime.store.get(result["task_id"])
    assert row.status == "failed"
    assert row.result == {"success": False, "error": "cannot process"}

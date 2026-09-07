"""Unsaved-project work has no restart-safe history or idempotency cache."""

import pytest

from core.jobs import JobNotFoundError, JobRuntime


def test_session_job_is_labelled_and_not_reused_after_restart():
    runtime = JobRuntime.for_session()
    result = runtime.submit(
        kind="test", args={}, run=lambda p, c: {"value": 1}, idempotency_key="same"
    )
    runtime.close_session()
    assert result["persistence"] == "session_only"
    restarted = JobRuntime.for_session()
    try:
        with pytest.raises(JobNotFoundError):
            restarted.store.get(result["task_id"])
        fresh = restarted.submit(
            kind="test", args={}, run=lambda p, c: {}, idempotency_key="same"
        )
        assert fresh["task_id"] != result["task_id"]
    finally:
        restarted.close_session()


def test_session_store_projection_never_claims_persisted_project():
    runtime = JobRuntime.for_session()
    try:
        result = runtime.submit(kind="test", args={}, run=lambda p, c: {})
        runtime._executor.shutdown(wait=True)
        row = runtime.store.get(result["task_id"])
        assert row.status == "completed"
        assert row.to_safe_projection()["persistence"] == "session_only"
    finally:
        runtime.close_session()


def test_session_runtime_rejects_saved_project_submission(tmp_path):
    runtime = JobRuntime.for_session()
    try:
        with pytest.raises(ValueError, match="session-only"):
            runtime.submit(
                kind="test",
                args={},
                run=lambda p, c: {},
                project_path=tmp_path / "saved.sceneripper",
            )
    finally:
        runtime.close_session()


def test_session_store_cannot_acknowledge_durable_project_commit(tmp_path):
    from core.jobs.commits import ResultSpec, commit_result
    from core.project import Project

    path = tmp_path / "saved.sceneripper"
    assert Project.new().save(path)
    runtime = JobRuntime.for_session()
    try:
        with pytest.raises(ValueError, match="durable"):
            commit_result(
                runtime.store,
                ResultSpec.build(
                    path, kind="test", version=1, target_id="x", arguments={}, inputs={}
                ),
                compute=lambda: {},
                validate_input=lambda p: True,
                apply=lambda p, r: None,
                is_applied=lambda p, r: True,
            )
    finally:
        runtime.close_session()


def test_qt_adapter_announces_session_only_before_terminal():
    from PySide6.QtWidgets import QApplication
    from ui.workers.job_adapter import JobAdapter

    _app = QApplication.instance() or QApplication([])
    runtime = JobRuntime.for_session()
    adapter = JobAdapter(runtime)
    events = []
    adapter.started.connect(
        lambda task, persistence: events.append((task, persistence))
    )
    adapter.completed.connect(lambda task, result: events.append((task, "completed")))
    try:
        result = adapter.start(kind="test", args={}, run=lambda p, c: {})
        runtime.shutdown()
        adapter._poll()
        adapter._poll()
        assert events == [
            (result["task_id"], "session_only"),
            (result["task_id"], "completed"),
        ]
    finally:
        runtime.close_session()


def test_session_cancellation_and_cache_are_local_to_runtime():
    import threading

    runtime = JobRuntime.for_session(max_workers=1)
    entered, release = threading.Event(), threading.Event()

    def block(progress, cancel):
        entered.set()
        assert release.wait(5)
        return {}

    try:
        runtime.submit(kind="block", args={}, run=block)
        assert entered.wait(5)
        called = []
        task = runtime.submit(
            kind="queued",
            args={},
            idempotency_key="key",
            run=lambda p, c: called.append(True) or {},
        )
        cached = runtime.submit(
            kind="queued", args={}, idempotency_key="key", run=lambda p, c: {}
        )
        assert cached["task_id"] == task["task_id"]
        assert cached["persistence"] == "session_only"
        assert runtime.cancel(task["task_id"])
        release.set()
        runtime.shutdown()
        assert not called
        assert runtime.store.get(task["task_id"]).status == "cancelled"
    finally:
        release.set()
        runtime.close_session()


def test_session_store_has_no_database_file_and_close_is_final():
    runtime = JobRuntime.for_session()
    with runtime.store._connect() as connection:
        assert connection.execute("PRAGMA database_list").fetchone()[2] == ""
        assert connection.execute("PRAGMA temp_store").fetchone()[0] == 2
    task = runtime.submit(
        kind="test", args={}, run=lambda p, c: {}, idempotency_key="completed"
    )
    runtime._executor.shutdown(wait=True)
    cached = runtime.submit(
        kind="test", args={}, run=lambda p, c: {}, idempotency_key="completed"
    )
    assert cached["task_id"] == task["task_id"]
    assert cached["persistence"] == "session_only"
    runtime.close_session()
    runtime.close_session()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.store.list()

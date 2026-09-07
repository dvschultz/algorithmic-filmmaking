"""GUI computation history persists without borrowing the editor's writer."""

from dataclasses import replace
from hashlib import sha256
import json
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from core.jobs import JobRuntime, JobStore
from core.jobs.spec import OperationSpec, encode_object
from core.project_lock import ProjectWriter
from ui.workers.job_adapter import (
    gui_job_operation,
    gui_job_runtime,
    close_gui_job_runtime,
)


def operation(path):
    base = OperationSpec.build(
        kind="transcribe",
        version=1,
        arguments={},
        inputs={},
        persistence="session_only",
        session_id="editor",
        input_revision="1",
    )
    return gui_job_operation(base, path)


def test_default_operation_keeps_legacy_json_identity():
    legacy = dict(
        kind="test",
        version=1,
        arguments={},
        inputs={},
        persistence="job_history",
        cancellable=True,
        session_id=None,
        input_revision=None,
    )
    encoded = encode_object(legacy)
    spec = OperationSpec.from_json(encoded)
    assert spec.publication == "worker"
    assert spec.to_json() == encoded
    assert spec.operation_id == sha256(encoded.encode()).hexdigest()
    assert "publication" not in spec.safe_projection()


@pytest.mark.parametrize("state", ["completed", "cancelled", "failed"])
def test_history_survives_runtime_close_with_editor_writer_held(
    tmp_path, monkeypatch, state
):
    path = tmp_path / "project.json"
    path.write_text("{}")
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    spec = operation(path)
    writer = ProjectWriter(path).acquire()
    runtime = gui_job_runtime(spec)

    def compute(progress, cancel):
        if state == "cancelled":
            cancel.set()
        if state == "failed":
            return {"success": False, "error": "preflight failed"}
        return {"outcomes": []}

    try:
        task = runtime.submit(
            kind=spec.kind,
            args=spec.arguments,
            operation=spec,
            project_path=path,
            run=compute,
        )["task_id"]
        close_gui_job_runtime(runtime)
        history = JobStore(tmp_path / "jobs.db")
        row = history.get(task)
        assert row.status == state
        assert row.project_path == str(path)
        assert row.kind == "gui_transcribe"
        assert row.result["publication"] == "explicit_project_save"
        assert json.loads(row.operation_json)["publication"] == "owner_thread"
        assert history.list(project_filter=str(path))[0].id == task
        assert path.read_text() == "{}"
    finally:
        runtime.shutdown()
        writer.close()


@pytest.mark.parametrize("change", ["session", "path"])
def test_owner_thread_submission_requires_bound_session_and_path(tmp_path, change):
    path = tmp_path / "project.json"
    spec = operation(path)
    if change == "session":
        spec = replace(spec, session_id=None)
    runtime = JobRuntime(JobStore(tmp_path / "jobs.db"))
    try:
        with pytest.raises(ValueError, match="bound project session"):
            runtime.submit(
                kind=spec.kind,
                args=spec.arguments,
                operation=spec,
                project_path=path if change == "session" else None,
                run=lambda *_: {},
            )
        assert runtime.store.list() == []
    finally:
        runtime.shutdown()


def test_history_sweep_recovers_orphans_but_preserves_live_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.settings.load_settings", lambda: SimpleNamespace(cache_dir=tmp_path)
    )
    path = tmp_path / "project.json"
    spec = operation(path)
    store = JobStore(tmp_path / "jobs.db")
    orphan = store.insert(
        kind="gui_transcribe", args={}, owner_id=str(uuid4()), status="running"
    )
    runtime = gui_job_runtime(spec)
    assert store.get(orphan.id).status == "crashed"
    entered, release = Event(), Event()

    def compute(*args):
        entered.set()
        assert release.wait(5)
        return {}

    try:
        task = runtime.submit(
            kind=spec.kind,
            args=spec.arguments,
            operation=spec,
            project_path=path,
            run=compute,
        )["task_id"]
        assert entered.wait(5)
        other = gui_job_runtime(spec)
        try:
            assert store.get(task).status == "running"
        finally:
            close_gui_job_runtime(other)
    finally:
        release.set()
        close_gui_job_runtime(runtime)
    assert store.get(task).status == "completed"


def test_worker_publication_still_respects_editor_writer(tmp_path):
    path = tmp_path / "project.json"
    path.write_text("{}")
    spec = replace(operation(path), publication="worker")
    writer = ProjectWriter(path).acquire()
    runtime = JobRuntime(JobStore(tmp_path / "jobs.db"))
    compute = Mock(return_value={})
    try:
        task = runtime.submit(
            kind=spec.kind,
            args=spec.arguments,
            operation=spec,
            project_path=path,
            run=compute,
        )["task_id"]
        runtime.shutdown()
        assert runtime.store.get(task).status == "failed"
        compute.assert_not_called()
        assert path.read_text() == "{}"
    finally:
        runtime.shutdown()
        writer.close()


def test_durable_store_connection_lifetimes_do_not_overlap(tmp_path):
    first = JobStore(tmp_path / "jobs.db")
    second = JobStore(tmp_path / "jobs.db")
    entered, release, other_entered = Event(), Event(), Event()

    def hold():
        with first._connect():
            entered.set()
            assert release.wait(5)

    def open_other():
        with second._connect():
            other_entered.set()

    one = Thread(target=hold)
    two = Thread(target=open_other)
    one.start()
    assert entered.wait(5)
    two.start()
    try:
        overlapped = other_entered.wait(0.2)
    finally:
        release.set()
        one.join(5)
        two.join(5)
    assert not one.is_alive() and not two.is_alive()
    assert other_entered.is_set()
    assert not overlapped

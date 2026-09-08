"""Job ownership includes referenced files, not just serialized receipt bodies."""

import json

import pytest

from core.artifacts import ArtifactStore
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore


@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("location", ["args", "operation", "result", "receipt"])
def test_job_retains_referenced_artifacts_across_producer_retirement(tmp_path, large, location):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    producer = artifacts.create_pin()
    ref = artifacts.put_bytes(b"derived input", pin=producer)
    value = {"analysis_json": json.dumps({"record_json": json.dumps({"artifact": ref.to_dict()})})}
    if large:
        value["padding"] = "x" * 100_000
    store = JobStore(tmp_path / "jobs.db")
    if location == "receipt":
        store.record_result("receipt", "{}", json.dumps(value), "digest")
        row = None
    else:
        operation = OperationSpec.build(kind="test", version=1, arguments={}, inputs=value,
                                        persistence="job_history") if location == "operation" else None
        row = store.insert(kind="test", args=value if location == "args" else {}, operation=operation)
        if location == "result":
            store.update_status(row.id, "completed", result=value)
    artifacts.release_pin(producer)
    store.close()
    assert artifacts.collect() == []
    assert artifacts.read_bytes(ref) == b"derived input"
    recovered = JobStore(store.db_path)
    if row is not None:
        recovered.delete(row.id)
        assert ref.digest in artifacts.collect()
    else:
        recovered.checkpoint_result("receipt", "digest")
        assert artifacts.collect() == []


def test_session_only_jobs_retain_existing_inputs_until_close(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    monkeypatch.setattr("core.paths.get_artifact_store_dir", lambda: root)
    artifacts = ArtifactStore(root)
    producer = artifacts.create_pin()
    ref = artifacts.put_bytes(b"derived input", pin=producer)
    store = JobStore.in_memory()
    row = store.insert(kind="test", args={"artifact": ref.to_dict()})
    artifacts.release_pin(producer)
    assert artifacts.collect() == []
    assert store.get(row.id).args == {"artifact": ref.to_dict()}
    assert not (tmp_path / "jobs.db").exists()
    store.close()
    assert artifacts.collect() == [ref.digest]
    store.close()


def test_replacing_a_result_releases_its_old_referenced_file(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    producer = artifacts.create_pin()
    ref = artifacts.put_bytes(b"derived output", pin=producer)
    store = JobStore(tmp_path / "jobs.db")
    row = store.insert(kind="test", args={})
    store.update_status(row.id, "running", result={"prerender_artifact": ref.to_dict()})
    artifacts.release_pin(producer)
    assert artifacts.collect() == []
    store.update_status(row.id, "completed", result={"done": True})
    assert artifacts.collect() == [ref.digest]


@pytest.mark.parametrize("phase", ["queued", "running"])
def test_runtime_keeps_inputs_when_the_producing_project_retires(tmp_path, phase):
    from threading import Event
    from core.jobs.runtime import JobRuntime

    artifacts = ArtifactStore(tmp_path / "artifacts")
    producer = artifacts.create_pin()
    ref = artifacts.put_bytes(b"input", pin=producer)
    runtime = JobRuntime(JobStore(tmp_path / "jobs.db"), max_workers=1)
    first_release, entered, release = Event(), Event(), Event()

    def first(progress, cancel):
        assert first_release.wait(5)
        return {}

    def consume(progress, cancel):
        entered.set()
        assert release.wait(5)
        return {"data": artifacts.read_bytes(ref).decode()}

    try:
        runtime.submit(kind="blocker", args={}, run=first)
        task = runtime.submit(kind="consumer", args={"artifact": ref.to_dict()}, run=consume)
        if phase == "running":
            first_release.set()
            assert entered.wait(5)
        artifacts.release_pin(producer)
        assert artifacts.collect() == []
        release.set()
        first_release.set()
        runtime.shutdown()
        row = runtime.store.get(task["task_id"])
        assert row.status == "completed" and row.result == {"data": "input"}
        runtime.store.delete(row.id)
        assert artifacts.collect() == [ref.digest]
    finally:
        first_release.set()
        release.set()
        runtime.shutdown()


def test_session_retirement_uses_the_original_artifact_root(tmp_path, monkeypatch):
    root = tmp_path / "original"
    monkeypatch.setattr("core.paths.get_artifact_store_dir", lambda: root)
    artifacts = ArtifactStore(root)
    producer = artifacts.create_pin()
    ref = artifacts.put_bytes(b"input", pin=producer)
    store = JobStore.in_memory()
    store.record_result("r", "{}", json.dumps({"artifact": ref.to_dict()}), "digest")
    artifacts.release_pin(producer)
    assert artifacts.collect() == []
    monkeypatch.setattr("core.paths.get_artifact_store_dir", lambda: tmp_path / "new-root")
    store.close()
    assert artifacts.collect() == [ref.digest]
    assert not (tmp_path / "new-root").exists()


def test_ordinary_user_strings_are_not_decoded_as_references(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    producer = artifacts.create_pin()
    ref = artifacts.put_bytes(b"unused", pin=producer)
    store = JobStore(tmp_path / "jobs.db")
    store.insert(kind="test", args={"notes": json.dumps({"artifact": ref.to_dict()})})
    artifacts.release_pin(producer)
    assert artifacts.collect() == [ref.digest]

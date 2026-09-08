"""Managed job inputs/results follow history ownership, including pruning."""

import json
import sqlite3
import time
from contextlib import contextmanager

import pytest

from core.artifacts import ArtifactStore
from core.jobs.spec import OperationSpec
from core.jobs.store import JobStore, STATUS_COMPLETED
from core.jobs.errors import StaleJobResult
from models.analysis_record import ArtifactRef


def test_job_inputs_and_result_roundtrip_and_retire_with_history(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    args = {"vectors": [0.123456789] * 4000}
    operation = OperationSpec.build(kind="example", version=1, arguments=args,
                                    inputs={}, persistence="job_history")
    row = store.insert(kind="example", args=args, operation=operation, idempotency_key="key")
    result = {"vectors": [0.987654321] * 4000}
    assert store.update_status(row.id, STATUS_COMPLETED, result=result)
    with sqlite3.connect(store.db_path) as db:
        saved = db.execute("SELECT args_json,operation_json,result_json FROM jobs").fetchone()
    assert saved == ("", "", "")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    assert artifacts.collect() == []
    recovered = JobStore(store.db_path)
    found = recovered.get(row.id)
    assert found.args == args and found.result == result
    assert found.operation_json == operation.to_json()
    assert recovered.list() == [found]
    assert recovered.find_by_idempotency("example", None, "key") == found
    assert "vectors" not in json.dumps(found.to_safe_projection())
    assert recovered.delete(row.id)
    assert len(artifacts.collect()) == 2
    assert not recovered.delete(row.id)


def test_pruning_preserves_live_jobs_and_computed_receipts(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    old = store.insert(kind="old", args={"data": "a" * 100_000})
    live = store.insert(kind="live", args={"data": "b" * 100_000})
    store.update_status(old.id, STATUS_COMPLETED, result={"data": "c" * 100_000})
    store.record_result("receipt", "{}", "d" * 100_000, "digest")
    with sqlite3.connect(store.db_path) as db:
        db.execute("UPDATE jobs SET finished_at=? WHERE id=?", (time.time() - 90 * 86400, old.id))
    assert store.purge_old_jobs() == 1
    artifacts = ArtifactStore(tmp_path / "artifacts")
    assert len(artifacts.collect()) == 2
    assert store.get(live.id).args == {"data": "b" * 100_000}
    assert store.get_result("receipt")["payload_json"] == "d" * 100_000


def test_replaced_results_release_previous_payload_and_rejected_update_releases_new(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    row = store.insert(kind="example", args={})
    store.update_status(row.id, "running", result={"data": "a" * 100_000})
    store.update_status(row.id, STATUS_COMPLETED, result={"data": "b" * 100_000})
    assert not store.update_status(row.id, STATUS_COMPLETED, result={"data": "c" * 100_000})
    artifacts = ArtifactStore(tmp_path / "artifacts")
    assert len(artifacts.collect()) == 2
    assert store.get(row.id).result == {"data": "b" * 100_000}


@pytest.mark.parametrize("large", [False, True])
def test_session_history_stays_in_memory(tmp_path, monkeypatch, large):
    monkeypatch.chdir(tmp_path)
    store = JobStore.in_memory()
    args = {"data": "x" * (100_000 if large else 1)}
    row = store.insert(kind="test", args=args)
    store.update_status(row.id, STATUS_COMPLETED, result=args)
    assert store.get(row.id).args == args
    assert store.get(row.id).result == args
    assert store.delete(row.id)
    assert not list(tmp_path.iterdir())
    store.close()


def test_delete_during_hydration_does_not_collect_an_active_reader(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs.db")
    data = {"data": "x" * 100_000}
    row = store.insert(kind="example", args=data)
    read = ArtifactStore.read_bytes

    def delete_then_read(artifacts, ref):
        assert store.delete(row.id)
        assert artifacts.collect() == []
        return read(artifacts, ref)

    monkeypatch.setattr(ArtifactStore, "read_bytes", delete_then_read)
    assert store.get(row.id).args == data
    assert len(ArtifactStore(tmp_path / "artifacts").collect()) == 1


def test_rejected_insert_releases_only_its_own_inputs(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    original = store.insert(kind="example", args={"data": "a" * 100_000},
                            project_path="/project", idempotency_key="key")
    with pytest.raises(sqlite3.IntegrityError):
        store.insert(kind="example", args={"data": "b" * 100_000},
                     project_path="/project", idempotency_key="key")
    assert len(ArtifactStore(tmp_path / "artifacts").collect()) == 1
    assert store.get(original.id).args == {"data": "a" * 100_000}


@pytest.mark.parametrize("damage", ["missing", "corrupt", "inline_conflict"])
def test_job_history_payload_damage_is_detected_and_still_deletable(tmp_path, damage):
    store = JobStore(tmp_path / "jobs.db")
    row = store.insert(kind="example", args={"data": "a" * 100_000})
    with sqlite3.connect(store.db_path) as db:
        reference = db.execute("SELECT input_artifact_json FROM jobs").fetchone()[0]
        if damage == "inline_conflict":
            db.execute("UPDATE jobs SET args_json='{}'")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    file = artifacts.path_for(ArtifactRef.from_dict(json.loads(reference)))
    if damage == "missing":
        file.unlink()
    elif damage == "corrupt":
        file.write_bytes(b"bad")
    with pytest.raises(StaleJobResult):
        store.get(row.id)
    assert store.delete(row.id)
    with sqlite3.connect(artifacts.database) as db:
        assert db.execute("SELECT count(*) FROM owners").fetchone()[0] == 0


def test_large_result_replaced_with_inline_result_releases_file(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    row = store.insert(kind="example", args={})
    store.update_status(row.id, "running", result={"data": "a" * 100_000})
    store.update_status(row.id, STATUS_COMPLETED, result={"done": True})
    assert len(ArtifactStore(tmp_path / "artifacts").collect()) == 1
    assert store.get(row.id).result == {"done": True}


@pytest.mark.parametrize("action", ["insert", "update", "delete"])
def test_uncertain_history_publication_conservatively_retains_files(tmp_path, monkeypatch, action):
    store = JobStore(tmp_path / "jobs.db")
    row = store.insert(kind="example", args={"data": "a" * 100_000})
    connect = store._connect

    @contextmanager
    def uncertain_close():
        with connect() as db:
            yield db
        raise OSError("uncertain close")

    monkeypatch.setattr(store, "_connect", uncertain_close)
    with pytest.raises(OSError, match="uncertain close"):
        if action == "insert":
            store.insert(kind="new", args={"data": "b" * 100_000})
        elif action == "update":
            store.update_status(row.id, STATUS_COMPLETED, result={"data": "b" * 100_000})
        else:
            store.delete(row.id)
    assert ArtifactStore(tmp_path / "artifacts").collect() == []
    recovered = JobStore(store.db_path)
    if action == "insert":
        assert recovered.list(kind_filter="new")[0].args == {"data": "b" * 100_000}
    elif action == "update":
        assert recovered.get(row.id).result == {"data": "b" * 100_000}
    else:
        assert recovered.list() == []

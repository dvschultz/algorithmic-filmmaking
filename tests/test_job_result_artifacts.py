"""Durable computed outputs retain their payloads across recovery and cleanup."""

import json
import sqlite3
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.artifacts import ArtifactStore
from core.jobs.errors import StaleJobResult
from core.jobs.store import JobStore
from models.analysis_record import ArtifactRef


def test_large_receipt_is_external_and_survives_reopen_and_cleanup(tmp_path):
    path = tmp_path / "jobs.db"
    store = JobStore(path)
    spec = json.dumps({"prior_vectors": [0.123456789] * 4000})
    payload = json.dumps({"vectors": [0.987654321] * 4000})
    original = store.record_result("result", spec, payload, "digest")
    store.close()
    with sqlite3.connect(path) as db:
        row = db.execute(
            "SELECT spec_json,payload_json,spec_artifact_json,payload_artifact_json,artifact_pin FROM job_results"
        ).fetchone()
    assert row[:2] == ("", "")
    assert row[4]
    artifacts = ArtifactStore(tmp_path / "artifacts")
    assert artifacts.collect() == []
    assert artifacts.read_bytes(ArtifactRef.from_dict(json.loads(row[2]))).decode() == spec
    reopened = JobStore(path)
    assert reopened.get_result("result") == original
    assert reopened.get_pending_results(["result"]) == [original]
    assert reopened.record_result("result", spec, payload, "digest") == original
    with pytest.raises(ValueError, match="different data"):
        reopened.record_result("result", spec, payload + " ", "digest")
    # A duplicate attempt must not release the winning row's ownership.
    assert artifacts.collect()  # conflicting payload is unreferenced
    assert reopened.get_result("result") == original
    reopened.checkpoint_result("result", "digest")
    assert reopened.get_pending_results(["result"]) == []
    assert artifacts.collect() == []


@pytest.mark.parametrize("session_only", [False, True])
def test_inline_storage_does_not_create_artifact_cache(tmp_path, session_only):
    store = JobStore.in_memory() if session_only else JobStore(tmp_path / "jobs.db")
    payload = "x" * 100_000 if session_only else "{}"
    assert store.record_result("r", "{}", payload, "d")["payload_json"] == payload
    assert not (tmp_path / "artifacts").exists()
    store.close()


@pytest.mark.parametrize("damage", ["missing", "corrupt", "inline_conflict"])
def test_damaged_external_receipt_is_not_returned_as_valid(tmp_path, damage):
    path = tmp_path / "jobs.db"
    store = JobStore(path)
    store.record_result("r", "{}", "x" * 100_000, "d")
    with sqlite3.connect(path) as db:
        reference = db.execute("SELECT payload_artifact_json FROM job_results").fetchone()[0]
        if damage == "inline_conflict":
            db.execute("UPDATE job_results SET payload_json='{}'")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    file = artifacts.path_for(ArtifactRef.from_dict(json.loads(reference)))
    if damage == "missing":
        file.unlink()
    elif damage == "corrupt":
        file.write_bytes(b"corrupt")
    with pytest.raises(StaleJobResult):
        store.get_result("r")
    with pytest.raises(StaleJobResult):
        store.get_pending_results(["r"])


def test_legacy_inline_database_upgrades_without_rewriting_receipts(tmp_path):
    path = tmp_path / "jobs.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE job_results (result_id TEXT PRIMARY KEY, spec_json TEXT NOT NULL, payload_json TEXT NOT NULL, payload_digest TEXT NOT NULL, committed INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL)")
        db.execute("INSERT INTO job_results VALUES ('r','{}','{\"value\":1}','d',0,1)")
    store = JobStore(path)
    assert store.get_result("r")["payload_json"] == '{"value":1}'
    assert not (tmp_path / "artifacts").exists()


def test_partial_staging_does_not_publish_a_receipt(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs.db")
    original = ArtifactStore.put_bytes
    calls = 0

    def fail_second(self, data, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        return original(self, data, **kwargs)

    monkeypatch.setattr(ArtifactStore, "put_bytes", fail_second)
    with pytest.raises(OSError, match="disk full"):
        store.record_result("r", "s" * 100_000, "p" * 100_000, "d")
    assert store.get_result("r") is None
    assert len(ArtifactStore(tmp_path / "artifacts").collect()) == 1


def test_uncertain_publication_retains_payload_for_recovery(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs.db")
    connect = store._connect

    @contextmanager
    def uncertain_close():
        with connect() as db:
            yield db
        raise OSError("connection close failed")

    monkeypatch.setattr(store, "_connect", uncertain_close)
    with pytest.raises(OSError, match="connection close failed"):
        store.record_result("r", "{}", "p" * 100_000, "d")
    assert ArtifactStore(tmp_path / "artifacts").collect() == []
    recovered = JobStore(store.db_path)
    assert recovered.get_result("r")["payload_json"] == "p" * 100_000


def test_concurrent_duplicate_writers_keep_one_durable_owner(tmp_path):
    path = tmp_path / "jobs.db"
    first, second = JobStore(path), JobStore(path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(store.record_result, "r", "{}", "p" * 100_000, "d")
                   for store in (first, second)]
        rows = [future.result(timeout=10) for future in futures]
    assert rows[0] == rows[1]
    artifacts = ArtifactStore(tmp_path / "artifacts")
    assert artifacts.collect() == []
    with sqlite3.connect(artifacts.database) as db:
        assert db.execute("SELECT count(*) FROM owners").fetchone()[0] == 1


@pytest.mark.parametrize("pending", [False, True])
def test_receipt_reader_retains_payload_during_concurrent_cleanup(tmp_path, monkeypatch, pending):
    store = JobStore(tmp_path / "jobs.db")
    payload = "p" * 100_000
    store.record_result("r", "{}", payload, "d")
    read = ArtifactStore.read_bytes

    def delete_then_read(artifacts, ref):
        with sqlite3.connect(store.db_path) as db:
            pin = db.execute("SELECT artifact_pin FROM job_results WHERE result_id='r'").fetchone()[0]
            db.execute("DELETE FROM job_results WHERE result_id='r'")
        artifacts.release_pin(pin)
        assert artifacts.collect() == []
        return read(artifacts, ref)

    monkeypatch.setattr(ArtifactStore, "read_bytes", delete_then_read)
    result = store.get_pending_results(["r"])[0] if pending else store.get_result("r")
    assert result["payload_json"] == payload
    assert len(ArtifactStore(tmp_path / "artifacts").collect()) == 1
    store.close()


def test_failed_receipt_read_releases_temporary_lease(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs.db")
    store.record_result("r", "{}", "p" * 100_000, "d")
    artifacts = ArtifactStore(tmp_path / "artifacts")

    def fail_read(*args):
        raise OSError("read interrupted")

    monkeypatch.setattr(ArtifactStore, "read_bytes", fail_read)
    with pytest.raises(StaleJobResult, match="unavailable or corrupt"):
        store.get_result("r")
    with sqlite3.connect(artifacts.database) as db:
        assert db.execute("SELECT count(*) FROM owners").fetchone()[0] == 1
    assert artifacts.collect() == []
    store.close()

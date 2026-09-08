"""Receipt pruning preserves project copies, undo, active work, and recovery."""

from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import time

import pytest

from core.artifacts import ArtifactStore
from core.jobs.commits import ResultSpec
from core.jobs.store import JobStore
from core.project import Project
from core.project_lock import ProjectWriter
from core.settings import Settings


@pytest.fixture
def obsolete(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr("core.settings.load_settings", lambda: Settings(cache_dir=cache))
    store = JobStore(cache / "jobs.db")
    project = Project.new()
    path = tmp_path / "project.sceneripper"
    spec = ResultSpec.build(path, kind="test", version=1, target_id="target", arguments={}, inputs={})
    payload = json.dumps({"value": "x" * 100_000})
    digest = sha256(payload.encode()).hexdigest()
    store.record_result(spec.result_id, spec.identity_json, payload, digest)
    project.record_job_result(spec.result_id, digest)
    assert project.save(path)
    store.checkpoint_result(spec.result_id, digest)
    project.metadata.job_results.clear()
    assert project.save(path)
    with store._connect() as db:
        db.execute("UPDATE job_results SET created_at=0")
    yield store, project, path, spec.result_id, digest
    project.session.close()
    store.close()


def test_prune_releases_only_obsolete_receipt_payloads(obsolete, tmp_path):
    store, _, path, rid, _ = obsolete
    original = path.read_bytes()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source media")
    artifacts = ArtifactStore(store.db_path.parent / "artifacts")
    assert artifacts.collect() == []
    assert store.purge_old_results() == 1
    assert store.get_result(rid) is None
    assert len(artifacts.collect()) == 1
    assert path.read_bytes() == original
    assert source.read_bytes() == b"source media"
    assert store.purge_old_results() == 0


@pytest.mark.parametrize("reason", ["uncommitted", "legacy", "recent", "unknown_owner"])
def test_uncertain_or_recoverable_rows_are_retained(obsolete, reason):
    store, _, _, rid, _ = obsolete
    with store._connect() as db:
        if reason == "uncommitted":
            db.execute("UPDATE job_results SET committed=0")
        elif reason == "legacy":
            db.execute("UPDATE job_results SET retention_managed=0")
        elif reason == "recent":
            db.execute("UPDATE job_results SET created_at=?", (time.time(),))
        else:
            db.execute("DELETE FROM receipt_project_history")
    assert store.purge_old_results() == 0
    assert store.get_result(rid) is not None
    assert ArtifactStore(store.db_path.parent / "artifacts").collect() == []


@pytest.mark.parametrize("status", ["queued", "running", "cancelling"])
def test_active_jobs_including_legacy_owners_prevent_pruning(obsolete, status):
    store, _, _, rid, _ = obsolete
    job = store.insert(kind="test", args={}, status=status)
    assert store.purge_old_results() == 0
    store.update_status(job.id, "completed", terminal=True)
    assert store.purge_old_results() == 1
    assert store.get_result(rid) is None


def test_closed_copy_and_live_undo_owners_are_protected(obsolete, tmp_path):
    store, _, path, rid, digest = obsolete
    copy = tmp_path / "copy.sceneripper"
    document = json.loads(path.read_bytes())
    document["job_results"] = {rid: digest}
    copy.write_text(json.dumps(document))
    loaded = Project.load(copy)
    try:
        assert store.purge_old_results() == 0
        loaded.metadata.job_results.clear()
        assert loaded.save(copy)
        with ProjectWriter(copy):
            # The saved file released the receipt, but an editor may still
            # restore it from live undo history under this writer.
            assert store.purge_old_results() == 0
        assert store.purge_old_results() == 1
    finally:
        loaded.session.close()


@pytest.mark.parametrize("state", ["missing", "invalid", "future", "referenced", "malformed_receipts"])
def test_disk_state_is_verified_before_reclaiming(obsolete, state):
    store, _, path, rid, digest = obsolete
    data = json.loads(path.read_bytes())
    if state == "missing":
        path.unlink()
    elif state == "invalid":
        path.write_text("{")
    else:
        if state == "future":
            data["version"] = "999.0"
        elif state == "referenced":
            data["job_results"] = {rid: digest}
        else:
            data["job_results"] = [rid]
        path.write_text(json.dumps(data))
    assert store.purge_old_results() == 0
    assert store.get_result(rid) is not None


@pytest.mark.parametrize("race", ["job", "copy", "document"])
def test_prune_revalidates_after_acquiring_project_writers(obsolete, tmp_path, monkeypatch, race):
    from core.jobs.retention import retain_loaded_receipts

    store, _, path, rid, digest = obsolete
    read = Path.read_bytes
    reads = 0

    def racing_read(target):
        nonlocal reads
        data = read(target)
        if target == path:
            reads += 1
            if reads == 1 and race == "job":
                store.insert(kind="late", args={})
            elif reads == 1 and race == "copy":
                copy = tmp_path / "late-copy.sceneripper"
                document = json.loads(data)
                document["job_results"] = {rid: digest}
                copy.write_text(json.dumps(document))
                retain_loaded_receipts(copy, {rid: digest})
            elif reads == 2 and race == "document":
                return data + b" "
        return data

    monkeypatch.setattr(Path, "read_bytes", racing_read)
    assert store.purge_old_results() == 0
    assert store.get_result(rid) is not None


def test_concurrent_receipt_reader_retains_deleted_body(obsolete, monkeypatch):
    store, _, _, rid, _ = obsolete
    read = ArtifactStore.read_bytes

    def prune_then_read(artifacts, ref):
        assert store.purge_old_results() == 1
        assert artifacts.collect() == []
        return read(artifacts, ref)

    monkeypatch.setattr(ArtifactStore, "read_bytes", prune_then_read)
    assert json.loads(store.get_result(rid)["payload_json"])["value"] == "x" * 100_000
    assert len(ArtifactStore(store.db_path.parent / "artifacts").collect()) == 1


def test_legacy_schema_migration_does_not_assume_tracked_ownership(tmp_path):
    from core.jobs.schema import RESULT_SCHEMA

    database = tmp_path / "jobs.db"
    with sqlite3.connect(database) as db:
        db.executescript(RESULT_SCHEMA)
        db.execute("INSERT INTO job_results VALUES ('old', '{}', '{}', 'digest', 1, 0)")
    store = JobStore(database)
    try:
        with store._connect() as db:
            assert db.execute("SELECT retention_managed FROM job_results").fetchone()[0] == 0
        assert store.purge_old_results() == 0
        assert store.get_result("old")["payload_json"] == "{}"
    finally:
        store.close()


@pytest.mark.parametrize("days", [-1, True, 1.5])
def test_invalid_retention_age_is_rejected(obsolete, days):
    store, *_ = obsolete
    with pytest.raises(ValueError, match="nonnegative integer"):
        store.purge_old_results(days)


@pytest.mark.asyncio
async def test_mcp_receipt_pruning_is_explicit(obsolete, monkeypatch):
    from scene_ripper_mcp.tools import jobs

    store, _, _, rid, _ = obsolete
    monkeypatch.setattr(jobs, "_lifespan", lambda ctx: {"job_store": store})
    default = json.loads(await jobs.purge_old_jobs())
    assert default == {"success": True, "deleted_count": 0, "days": 30}
    assert store.get_result(rid) is not None
    result = json.loads(await jobs.purge_old_jobs(include_results=True))
    assert result == {"success": True, "deleted_count": 0, "deleted_result_count": 1, "days": 30}
    assert store.get_result(rid) is None


def test_pending_save_is_protected_until_abandoned_reconciliation(obsolete):
    from core.jobs.retention import retain_receipt_manifest

    store, _, path, rid, digest = obsolete
    with ProjectWriter(path):
        with pytest.raises(OSError):
            with retain_receipt_manifest(store, path, {"job_results": {rid: digest}}):
                assert store.purge_old_results() == 0
                raise OSError("save interrupted")
    # Saved disk state is still the released version; only after ownership
    # reconciliation may the committed receipt be reclaimed.
    assert store.purge_old_results() == 1


def test_artifact_pin_release_follows_durable_receipt_deletion(obsolete, monkeypatch):
    store, _, _, rid, _ = obsolete
    release = ArtifactStore.release_pin

    def checked_release(artifacts, owner):
        with sqlite3.connect(store.db_path) as db:
            assert db.execute("SELECT 1 FROM job_results WHERE result_id=?", (rid,)).fetchone() is None
        return release(artifacts, owner)

    monkeypatch.setattr(ArtifactStore, "release_pin", checked_release)
    assert store.purge_old_results() == 1


def test_prune_walks_multiple_bounded_batches(obsolete):
    store, project, path, _, _ = obsolete
    digest = sha256(b"{}").hexdigest()
    acknowledgements = []
    for index in range(100):
        spec = ResultSpec.build(path, kind="test", version=1, target_id=str(index), arguments={}, inputs={})
        store.record_result(spec.result_id, spec.identity_json, "{}", digest)
        project.record_job_result(spec.result_id, digest)
        acknowledgements.append((spec.result_id, digest))
    assert project.save(path)
    store.checkpoint_results(acknowledgements)
    project.metadata.job_results.clear()
    assert project.save(path)
    with store._connect() as db:
        db.execute("UPDATE job_results SET created_at=0")
    assert store.purge_old_results() == 101
    assert store.purge_old_results() == 0

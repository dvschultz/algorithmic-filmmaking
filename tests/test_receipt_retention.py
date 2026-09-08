"""Project receipt references survive atomic publication and Save As."""

from hashlib import sha256
import json

import pytest

from core.jobs.retention import retain_receipt_manifest
from core.jobs.store import JobStore
from core.project import Project
from core.settings import Settings


@pytest.fixture
def saved(tmp_path, monkeypatch):
    settings = Settings(cache_dir=tmp_path / "cache")
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    store = JobStore(settings.cache_dir / "jobs.db")
    project = Project.new()
    path = tmp_path / "project.sceneripper"
    payload = "{}"
    spec = json.dumps({"project_path": str(path), "kind": "test", "target_id": "target"})
    rid, digest = sha256(spec.encode()).hexdigest(), sha256(payload.encode()).hexdigest()
    store.record_result(rid, spec, payload, digest)
    project.metadata.job_results[rid] = digest
    yield store, project, path, rid
    project.session.close()
    store.close()


def refs(store, table):
    with store._connect() as db:
        return {tuple(row) for row in db.execute(f"SELECT * FROM {table}")}


def test_saved_receipts_are_retained_and_save_as_adds_an_owner(saved, tmp_path):
    store, project, path, rid = saved
    assert project.save(path)
    assert refs(store, "receipt_manifests") == {(str(path), rid)}
    copy = tmp_path / "copy.sceneripper"
    assert project.save(copy)
    assert refs(store, "receipt_manifests") == {(str(path), rid), (str(copy), rid)}
    assert refs(store, "receipt_pending_saves") == set()


def test_removing_saved_receipts_keeps_historical_owners_for_undo(saved):
    store, project, path, rid = saved
    assert project.save(path)
    project.metadata.job_results.clear()
    assert project.save(path)
    assert refs(store, "receipt_manifests") == set()
    assert refs(store, "receipt_project_history") == {(str(path), rid)}


def test_loading_an_external_copy_registers_its_receipts(saved, tmp_path):
    store, project, path, rid = saved
    assert project.save(path)
    copy = tmp_path / "external-copy.sceneripper"
    copy.write_bytes(path.read_bytes())
    loaded = Project.load(copy, retain_writer=True)
    try:
        assert (str(copy), rid) in refs(store, "receipt_manifests")
        assert (str(copy), rid) in refs(store, "receipt_project_history")
    finally:
        loaded.session.close()


def test_loaded_older_snapshot_does_not_release_newer_manifest_refs(saved):
    from core.jobs.retention import retain_loaded_receipts

    store, project, path, old = saved
    assert project.save(path)
    new = "c" * 64
    project.metadata.job_results = {new: "d" * 64}
    assert project.save(path)
    retain_loaded_receipts(path, {old: "b" * 64})
    assert refs(store, "receipt_manifests") == {(str(path), old), (str(path), new)}


def test_failed_publication_keeps_old_and_incoming_receipt_owners(saved, monkeypatch):
    store, project, path, old = saved
    assert project.save(path)
    incoming = "c" * 64
    project.metadata.job_results = {incoming: "d" * 64}

    def fail(*args):
        raise OSError("publication interrupted")

    monkeypatch.setattr("core.project_lock.replace_project_file", fail)
    assert not project.save(path)
    assert refs(store, "receipt_manifests") == {(str(path), old)}
    assert {rid for _, rid in refs(store, "receipt_pending_refs")} == {incoming}
    assert set(json.loads(path.read_text())["job_results"]) == {old}


def test_ordinary_save_does_not_open_cache_settings(tmp_path, monkeypatch):
    def blocked():
        raise AssertionError("ordinary saves must not initialize job storage")

    monkeypatch.setattr("core.settings.load_settings", blocked)
    project = Project.new()
    assert project.save(tmp_path / "plain.sceneripper")
    project.session.close()


def test_imported_receipts_do_not_create_an_empty_cache(tmp_path, monkeypatch):
    settings = Settings(cache_dir=tmp_path / "cache")
    monkeypatch.setattr("core.settings.load_settings", lambda: settings)
    project = Project.new()
    project.metadata.job_results = {"a" * 64: "b" * 64}
    assert project.save(tmp_path / "imported.sceneripper")
    assert not (settings.cache_dir / "jobs.db").exists()
    project.session.close()


def test_postpublication_index_failure_retains_pending_refs(saved, monkeypatch):
    from contextlib import contextmanager

    store, project, path, rid = saved
    connect = store._connect
    calls = 0

    @contextmanager
    def fail_second_connection():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("ownership index unavailable")
        with connect() as db:
            yield db

    monkeypatch.setattr(store, "_connect", fail_second_connection)
    with retain_receipt_manifest(store, path, {"job_results": project.metadata.job_results}):
        path.write_text("saved")
    monkeypatch.setattr(store, "_connect", connect)
    assert {result for _, result in refs(store, "receipt_pending_refs")} == {rid}

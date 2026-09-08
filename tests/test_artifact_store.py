"""Publication and cross-project retention for managed derived files."""

import pytest

from core.artifacts import ArtifactStore, ArtifactUnavailable
from models.analysis_record import AnalysisRecord
from tests.test_analysis_records import identity


def test_staged_artifact_is_pinned_before_manifest_publication(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    with store.pin() as writer:
        ref = store.put_bytes(b"result", pin=writer)
        assert store.collect() == []
        store.set_manifest(tmp_path / "project.json", [ref])
    assert store.read_bytes(ref) == b"result"
    assert store.collect() == []


@pytest.mark.parametrize("damaged", ["embeddings", "boundary_embeddings"])
def test_boundary_arrays_are_managed_and_damage_is_isolated(tmp_path, monkeypatch, damaged):
    import json
    from core.project import Project
    from models.analysis_record import ArtifactRef
    from tests.test_description_operations import project_with_thumbnails

    root = tmp_path / "artifacts"
    monkeypatch.setattr("core.paths.get_artifact_store_dir", lambda: root)
    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.notes = "Keep these notes"
    clip.embedding = [0.1] * 768
    clip.first_frame_embedding = [0.2] * 768
    clip.last_frame_embedding = [0.3] * 768
    clip.embedding_model = "dinov2-vit-b-14"
    values = {
        "embeddings": {"embedding": clip.embedding, "embedding_model": clip.embedding_model},
        "boundary_embeddings": {"first_frame_embedding": clip.first_frame_embedding,
                                "last_frame_embedding": clip.last_frame_embedding, "embedding_model": clip.embedding_model},
    }
    for operation, value in values.items():
        project.record_analysis("clip", clip.id, operation, AnalysisRecord.success(identity(operation=operation), value))
    path = tmp_path / "project.json"
    assert project.save(path)
    saved = json.loads(path.read_text())["clips"][0]
    assert not {"embedding", "first_frame_embedding", "last_frame_embedding"} & saved.keys()
    store = ArtifactStore(root)
    refs = {operation: ArtifactRef.from_dict(saved["analysis_records"][operation]["artifact"]) for operation in values}
    assert store.collect() == []
    store.path_for(refs[damaged]).unlink()
    restored = Project.load(path)
    result = restored.clips[0]
    assert result.analysis_records[damaged].state == "missing"
    assert result.embedding_model == "dinov2-vit-b-14"
    assert result.notes == "Keep these notes"
    if damaged == "embeddings":
        assert result.embedding is None
        assert result.first_frame_embedding == [0.2] * 768
        assert result.last_frame_embedding == [0.3] * 768
    else:
        assert result.embedding == [0.1] * 768
        assert result.first_frame_embedding is None and result.last_frame_embedding is None
    assert restored.save()
    assert Project.load(path).clips[0].embedding_model == "dinov2-vit-b-14"


def test_failed_boundary_attempt_keeps_managed_display_vectors_without_becoming_successful(tmp_path, monkeypatch):
    import json
    from core.project import Project
    from models.analysis_record import ArtifactRef
    from tests.test_description_operations import project_with_thumbnails

    root = tmp_path / "artifacts"
    monkeypatch.setattr("core.paths.get_artifact_store_dir", lambda: root)
    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.first_frame_embedding = [0.2] * 768
    clip.last_frame_embedding = [0.3] * 768
    clip.embedding_model = "dinov2-vit-b-14"
    project.record_analysis("clip", clip.id, "boundary_embeddings",
                            AnalysisRecord.failure(identity(operation="boundary_embeddings"), "Refresh failed"))
    path = tmp_path / "project.json"
    assert project.save(path)
    saved = json.loads(path.read_text())["clips"][0]
    assert "first_frame_embedding" not in saved and "last_frame_embedding" not in saved
    record = saved["analysis_records"]["boundary_embeddings"]
    assert record["state"] == "failed"
    ref = ArtifactRef.from_dict(record["artifact"])
    loaded = Project.load(path)
    assert loaded.clips[0].first_frame_embedding == [0.2] * 768
    assert loaded.clips[0].analysis_records["boundary_embeddings"].state == "failed"
    store = ArtifactStore(root)
    payload = store.read_bytes(ref)
    store.path_for(ref).unlink()
    damaged = Project.load(path)
    assert damaged.clips[0].first_frame_embedding is None
    assert damaged.clips[0].analysis_records["boundary_embeddings"].state == "failed"
    assert damaged.save()
    with store.pin() as producer:
        store.put_bytes(payload, pin=producer, media_type="application/json")
    recovered = Project.load(path)
    assert recovered.clips[0].first_frame_embedding == [0.2] * 768
    assert recovered.clips[0].analysis_records["boundary_embeddings"].state == "failed"


def test_closed_projects_and_job_export_history_pins_retain_artifacts(tmp_path):
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    with store.pin() as writer:
        refs = [store.put_bytes(label.encode(), pin=writer) for label in ("closed", "job", "export", "history", "unused")]
        store.set_manifest(tmp_path / "closed.json", [refs[0]])
        pins = [store.create_pin([ref]) for ref in refs[1:4]]
    reopened = ArtifactStore(root)
    assert reopened.collect() == [refs[4].digest]
    for ref in refs[:4]:
        assert reopened.read_bytes(ref)
    for pin in pins:
        reopened.release_pin(pin)
    assert set(reopened.collect()) == {ref.digest for ref in refs[1:4]}
    assert reopened.read_bytes(refs[0]) == b"closed"


def test_missing_or_corrupt_artifact_is_unavailable_without_changing_record(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    with store.pin() as writer:
        ref = store.put_bytes(b"[]", pin=writer)
        record = AnalysisRecord.success(identity(), artifact=ref)
        assert record.reusable(identity(), artifact_available=store.available)
        path = store.path_for(ref)
        path.write_bytes(b"xx")
        assert not record.reusable(identity(), artifact_available=store.available)
        with pytest.raises(ArtifactUnavailable):
            store.read_bytes(ref)
        path.unlink()
        assert not record.reusable(identity(), artifact_available=store.available)
        assert record.artifact == ref


def test_failed_publication_does_not_register_partial_artifact(tmp_path, monkeypatch):
    import core.artifacts as module

    store = ArtifactStore(tmp_path / "artifacts")
    with store.pin() as writer:
        monkeypatch.setattr(module.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError, match="disk full"):
            store.put_bytes(b"payload", pin=writer)
    assert store.collect() == []
    assert not list((store.root / "objects").rglob("*.tmp"))


def test_cleanup_never_deletes_source_unknown_or_replaced_files(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original media")
    store = ArtifactStore(tmp_path / "artifacts")
    unknown = store.root / "objects" / "unknown"
    unknown.write_bytes(b"untracked")
    with store.pin() as writer:
        ref = store.put_file(source, pin=writer)
        managed = store.path_for(ref)
    managed.unlink()
    managed.hardlink_to(source)
    assert store.collect() == []
    assert source.read_bytes() == b"original media"
    assert unknown.read_bytes() == b"untracked"


def test_unknown_pin_cannot_publish_and_manifest_can_retain_offline_reference(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="pin"):
        store.put_bytes(b"unowned", pin="not-a-pin")
    with store.pin() as writer:
        ref = store.put_bytes(b"offline", pin=writer)
        store.path_for(ref).unlink()
        store.set_manifest(tmp_path / "closed.json", [ref])
    assert not store.available(ref)
    assert store.collect() == []


def test_database_failure_after_file_publication_allows_safe_retry(tmp_path, monkeypatch):
    from contextlib import contextmanager

    store = ArtifactStore(tmp_path / "artifacts")
    original = store._transaction
    fail = True

    @contextmanager
    def fail_commit():
        nonlocal fail
        with original() as db:
            yield db
            if fail and db.execute("SELECT COUNT(*) FROM objects").fetchone()[0]:
                fail = False
                raise OSError("database commit failed")

    with store.pin() as writer:
        monkeypatch.setattr(store, "_transaction", fail_commit)
        with pytest.raises(OSError, match="database commit failed"):
            store.put_bytes(b"result", pin=writer)
        orphan = next(store.objects.glob("*.blob"))
        ref = store.put_bytes(b"result", pin=writer)
        assert store.read_bytes(ref) == b"result"
        assert store.path_for(ref) != orphan and orphan.exists()
    assert store.collect() == [ref.digest]
    assert orphan.read_bytes() == b"result"


def test_revoked_producer_cannot_publish_after_finishing_copy(tmp_path):
    import io

    store = ArtifactStore(tmp_path / "artifacts")
    pin = store.create_pin()

    class CancelledWriter(io.BytesIO):
        def read(self, size=-1):
            store.release_pin(pin)
            return super().read(size)

    with pytest.raises(ValueError, match="pin"):
        store._put(CancelledWriter(b"result"), pin=pin, media_type="application/json")
    assert not list(store.objects.iterdir())


@pytest.mark.parametrize("operation", ["embeddings", "boundary_embeddings", "face_embeddings"])
def test_project_save_and_source_undo_retain_real_references(tmp_path, monkeypatch, operation):
    from core.project import Project
    from models.clip import Clip, Source

    root = tmp_path / "artifacts"
    monkeypatch.setattr("core.paths.get_artifact_store_dir", lambda: root)
    store = ArtifactStore(root)
    media = tmp_path / "source.mp4"
    media.write_bytes(b"source")
    source = Source(id="source", file_path=media)
    clip = Clip(id="clip", source_id=source.id)
    project = Project(sources=[source], clips=[clip])
    with store.pin() as writer:
        ref = store.put_bytes(b"embedding", pin=writer)
        record = AnalysisRecord.success(identity(operation=operation), artifact=ref)
        project.record_analysis("clip", clip.id, operation, record)
    assert store.collect() == []  # Unsaved live project owns the result.
    path = tmp_path / "project.json"
    assert project.save(path)
    project.remove_sources([source.id])
    assert project.save()
    assert store.collect() == []  # Saved manifest is empty; undo still owns it.
    project.session.undo()
    assert project.clips[0].analysis_records[operation].artifact == ref
    project.session.redo()
    project.session.close()
    assert store.collect() == [ref.digest]


def test_saved_project_keeps_artifacts_after_session_closes(tmp_path, monkeypatch):
    from core.project import Project
    from models.clip import Clip

    root = tmp_path / "artifacts"
    monkeypatch.setattr("core.paths.get_artifact_store_dir", lambda: root)
    store = ArtifactStore(root)
    project = Project(clips=[Clip(id="clip")])
    with store.pin() as writer:
        ref = store.put_bytes(b"result", pin=writer)
        project.record_analysis("clip", "clip", "embeddings", AnalysisRecord.success(identity(operation="embeddings"), artifact=ref))
    assert project.save(tmp_path / "closed.json")
    project.session.close()
    assert ArtifactStore(root).collect() == []
    assert store.read_bytes(ref) == b"result"


def test_unrelated_project_edits_do_not_rewrite_artifact_pins(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from core.project import Project
    from models.clip import Clip

    monkeypatch.setattr("core.paths.get_artifact_store_dir", lambda: tmp_path / "artifacts")
    project = Project(clips=[Clip(id="clip")])
    with project.artifact_store.pin() as writer:
        ref = project.artifact_store.put_bytes(b"result", pin=writer)
        project.record_analysis("clip", "clip", "embeddings", AnalysisRecord.success(identity(operation="embeddings"), artifact=ref))
    replace_pin = Mock(wraps=project.artifact_store.replace_pin)
    monkeypatch.setattr(project.artifact_store, "replace_pin", replace_pin)
    for _ in range(5):
        project.mark_dirty()
    replace_pin.assert_not_called()
    assert project.artifact_store.collect() == []
    project.session.close()
    assert project.artifact_store.collect() == [ref.digest]


def test_lease_finalization_during_transaction_defers_release(tmp_path, monkeypatch):
    import gc
    import sqlite3
    import sys
    from core.artifacts import ArtifactLease

    store = ArtifactStore(tmp_path / "artifacts")
    with store.pin() as writer:
        ref = store.put_bytes(b"result", pin=writer)
        lease = ArtifactLease([ref], store.root)
    errors = []
    monkeypatch.setattr(sys, "unraisablehook", errors.append)
    connect = sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: connect(*args, **{**kwargs, "timeout": 0}))
    with store._transaction():
        del lease
        gc.collect()
    assert errors == []
    assert store.collect() == [ref.digest]


def test_manifest_registration_failure_after_save_keeps_safety_pin(tmp_path, monkeypatch):
    from core.project import Project
    from models.clip import Clip

    root = tmp_path / "artifacts"
    monkeypatch.setattr("core.paths.get_artifact_store_dir", lambda: root)
    store = ArtifactStore(root)
    project = Project(clips=[Clip(id="clip")])
    with store.pin() as writer:
        ref = store.put_bytes(b"saved", pin=writer)
        project.record_analysis("clip", "clip", "embeddings", AnalysisRecord.success(identity(operation="embeddings"), artifact=ref))
    original = ArtifactStore.set_manifest
    monkeypatch.setattr(ArtifactStore, "set_manifest", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("index unavailable")))
    assert project.save(tmp_path / "saved.json")
    project.session.close()
    assert ArtifactStore(root).collect() == []
    assert store.read_bytes(ref) == b"saved"
    # A later successful replacement reconciles abandoned save pins.
    monkeypatch.setattr(ArtifactStore, "set_manifest", original)
    replacement = Project()
    assert replacement.save(tmp_path / "saved.json")
    replacement.session.close()
    assert store.collect() == [ref.digest]


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_unavailable_embedding_preserves_notes_and_sequences(tmp_path, monkeypatch, damage):
    import json
    from core.project import Project
    from models.clip import Clip, Source

    root = tmp_path / "artifacts"
    monkeypatch.setattr("core.paths.get_artifact_store_dir", lambda: root)
    store = ArtifactStore(root)
    media = tmp_path / "source.mp4"
    media.write_bytes(b"source")
    source = Source(id="source", file_path=media, fps=24.0)
    clip = Clip(id="clip", source_id=source.id, start_frame=0, end_frame=24, notes="Keep editorial notes")
    project = Project(sources=[source], clips=[clip])
    project.add_to_sequence([clip.id])
    sequence = project.sequence.to_dict()
    with store.pin() as producer:
        ref = store.put_bytes(json.dumps({"embedding": [1.0] * 768, "embedding_model": "dinov2-vit-b-14"}).encode(), pin=producer, media_type="application/json")
        project.record_analysis("clip", clip.id, "embeddings", AnalysisRecord.success(identity(operation="embeddings"), artifact=ref))
    assert project.save(tmp_path / "project.json")
    if damage == "missing":
        store.path_for(ref).unlink()
    else:
        store.path_for(ref).write_bytes(b"corrupt")
    restored = Project.load(tmp_path / "project.json")
    assert restored.clips[0].embedding is None
    record = restored.clips[0].analysis_records["embeddings"]
    assert record.state == "missing" and record.artifact == ref
    assert restored.clips[0].notes == "Keep editorial notes"
    assert restored.sequence.to_dict() == sequence
    assert not record.reusable(identity(operation="embeddings"), artifact_available=store.available)
    assert restored.save()
    restored.session.close()
def test_collection_reconciles_interrupted_save_against_saved_manifest(tmp_path):
    import json
    from core.artifacts import ArtifactStore
    from core.project import Project

    store = ArtifactStore(tmp_path / "artifacts")
    path = tmp_path / "project.sceneripper"
    project = Project.new()
    project.save(path)
    project.session.close()
    document = json.loads(path.read_text())
    pin = store.create_manifest_pin(path, ())
    retained = store.put_bytes(b"saved artifact", pin=pin)
    abandoned = store.put_bytes(b"abandoned artifact", pin=pin)
    document["custom_data"] = {"artifact": retained.to_dict()}
    path.write_text(json.dumps(document))
    assert store.collect() == [abandoned.digest]
    assert store.available(retained)
    with store._connection() as db:
        assert db.execute("SELECT count(*) FROM pending_manifests").fetchone()[0] == 0


def test_collection_preserves_active_manifest_writer(tmp_path):
    from core.artifacts import ArtifactStore
    from core.project import Project
    from core.project_lock import ProjectWriter

    store = ArtifactStore(tmp_path / "artifacts")
    path = tmp_path / "project.sceneripper"
    project = Project.new()
    project.save(path)
    project.session.close()
    with ProjectWriter(path):
        pin = store.create_manifest_pin(path, ())
        artifact = store.put_bytes(b"in-flight save", pin=pin)
        assert store.collect() == []
        assert store.available(artifact)
    assert store.collect() == [artifact.digest]


@pytest.mark.parametrize("state", ["missing", "invalid_json", "future", "invalid_reference"])
def test_pending_manifest_reconciliation_preserves_uncertain_projects(tmp_path, state):
    import json
    from core.project import Project

    store = ArtifactStore(tmp_path / "artifacts")
    path = tmp_path / "project.sceneripper"
    project = Project.new()
    project.save(path)
    project.session.close()
    document = json.loads(path.read_text())
    if state == "missing":
        path.unlink()
    elif state == "invalid_json":
        path.write_bytes(b"{")
    else:
        if state == "future":
            document["version"] = "999.0"
        else:
            document["custom_data"] = {"artifact": {"digest": "unknown"}}
        path.write_text(json.dumps(document))
    pin = store.create_manifest_pin(path, ())
    artifact = store.put_bytes(b"uncertain", pin=pin)
    assert store.collect() == []
    assert store.available(artifact)
    with store._connection() as db:
        assert db.execute("SELECT count(*) FROM pending_manifests").fetchone()[0] == 1


def test_pending_manifest_reconciliation_detects_external_document_change(tmp_path, monkeypatch):
    from pathlib import Path
    from core.project import Project

    store = ArtifactStore(tmp_path / "artifacts")
    path = tmp_path / "project.sceneripper"
    project = Project.new()
    project.save(path)
    project.session.close()
    pin = store.create_manifest_pin(path, ())
    artifact = store.put_bytes(b"uncertain", pin=pin)
    read = Path.read_bytes
    reads = []

    def changed_read(current):
        data = read(current)
        if current == path:
            reads.append(current)
            if len(reads) == 2:
                return data + b" "
        return data

    monkeypatch.setattr(Path, "read_bytes", changed_read)
    assert store.collect() == []
    assert store.available(artifact)

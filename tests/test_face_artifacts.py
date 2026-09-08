"""Managed face payloads retain verified reuse and isolate damaged artifacts."""

import json
import pytest

from core.project import Project
from core.artifacts import ArtifactStore
from core.analysis_availability import face_analysis_is_complete
from models.analysis_record import ArtifactRef
from tests.test_face_records import setup as face_setup, run  # noqa: F401


@pytest.fixture
def analyzed(request, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.paths.get_artifact_store_dir", lambda: tmp_path / "artifacts"
    )
    project, provider, directory = request.getfixturevalue("face_setup")
    run(project)
    return project, provider, directory


@pytest.mark.parametrize("empty", [False, True])
def test_face_vectors_are_externalized_and_reusable(analyzed, tmp_path, empty):
    project, provider, _ = analyzed
    if empty:
        original = provider.side_effect
        provider.side_effect = lambda **kwargs: original(**kwargs) and []
        run(project, skip=False)
    calls = provider.call_count
    path = tmp_path / "project.sceneripper"
    project.save(path)
    document = json.loads(path.read_text())["clips"][0]
    assert "face_embeddings" not in document
    ref = ArtifactRef.from_dict(
        document["analysis_records"]["face_embeddings"]["artifact"]
    )
    assert ref.media_type == "application/json"
    assert ArtifactStore().collect() == []
    restored = Project.load(path)
    assert restored.clips[0].face_embeddings == project.clips[0].face_embeddings
    assert face_analysis_is_complete(restored.clips[0], restored.sources[0])
    outcome, _ = run(restored)
    assert outcome.status == "skipped"
    assert provider.call_count == calls


def test_missing_face_payload_preserves_editorial_data(analyzed, tmp_path):
    project, _, _ = analyzed
    project.clips[0].notes = "Keep this note"
    path = tmp_path / "project.sceneripper"
    project.save(path)
    document = json.loads(path.read_text())["clips"][0]
    ref = ArtifactRef.from_dict(
        document["analysis_records"]["face_embeddings"]["artifact"]
    )
    store = ArtifactStore()
    payload = store.read_bytes(ref)
    store.path_for(ref).unlink()
    restored = Project.load(path)
    assert restored.clips[0].face_embeddings is None
    assert restored.clips[0].notes == "Keep this note"
    assert restored.clips[0].analysis_records["face_embeddings"].state == "missing"
    assert restored.save()
    with store.pin() as pin:
        store.put_bytes(payload, pin=pin, media_type="application/json")
    recovered = Project.load(path)
    assert face_analysis_is_complete(recovered.clips[0], recovered.sources[0])


@pytest.mark.parametrize("state", ["failed", "legacy"])
def test_saved_face_payload_preserves_nonverified_state(analyzed, tmp_path, state):
    from dataclasses import replace

    project, _, _ = analyzed
    clip = project.clips[0]
    if state == "failed":
        clip.analysis_records["face_embeddings"] = replace(
            clip.analysis_records["face_embeddings"], state="failed"
        )
    else:
        clip.analysis_records.clear()
    path = tmp_path / "project.sceneripper"
    project.save(path)
    restored = Project.load(path)
    assert restored.clips[0].face_embeddings == clip.face_embeddings
    record = restored.clips[0].analysis_records["face_embeddings"]
    assert record.artifact is not None
    assert not face_analysis_is_complete(restored.clips[0], restored.sources[0])
    assert (
        record.state == "failed"
        if state == "failed"
        else record.provenance == "unknown"
    )


def test_bundle_restores_face_payload_in_fresh_store(analyzed, tmp_path, monkeypatch):
    from core.project_export import export_project_bundle

    project, _, _ = analyzed
    destination = tmp_path / "bundle"
    result = export_project_bundle(project, destination, include_clips=False)
    assert result.artifacts_copied == 1
    monkeypatch.setattr(
        "core.paths.get_artifact_store_dir", lambda: tmp_path / "second-artifacts"
    )
    restored = Project.load(destination / f"{project.metadata.name}.sceneripper")
    assert restored.clips[0].face_embeddings == project.clips[0].face_embeddings
    assert restored.clips[0].analysis_records["face_embeddings"].artifact is not None


def test_malformed_managed_face_payload_is_missing(analyzed, tmp_path):
    from dataclasses import replace

    project, _, _ = analyzed
    clip = project.clips[0]
    store = ArtifactStore()
    with store.pin() as pin:
        ref = store.put_bytes(
            json.dumps({"face_embeddings": [{"embedding": [1]}]}).encode(),
            pin=pin,
            media_type="application/json",
        )
        project.record_analysis(
            "clip",
            clip.id,
            "face_embeddings",
            replace(
                clip.analysis_records["face_embeddings"], artifact=ref, value_json=None
            ),
        )
    clip.face_embeddings = None
    path = tmp_path / "invalid.sceneripper"
    project.save(path)
    restored = Project.load(path)
    assert restored.clips[0].face_embeddings is None
    assert restored.clips[0].analysis_records["face_embeddings"].state == "missing"


def test_missing_payload_can_be_recomputed_by_durable_job(analyzed, tmp_path):
    from core.jobs.faces import run_face_job
    from core.jobs.store import JobStore
    from threading import Event

    project, _, _ = analyzed
    path = tmp_path / "project.sceneripper"
    project.save(path)
    store = JobStore(tmp_path / "jobs.db")
    try:
        run_face_job(store, path, None, lambda *_: None, Event(), force=True)
        loaded = Project.load(path)
        ref = loaded.clips[0].analysis_records["face_embeddings"].artifact
        ArtifactStore().path_for(ref).unlink()
        result = run_face_job(store, path, None, lambda *_: None, Event())["result"]
        assert len(result["succeeded"]) == 1
        restored = Project.load(path)
        assert face_analysis_is_complete(restored.clips[0], restored.sources[0])
    finally:
        store.close()

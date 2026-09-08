"""Semantic analysis identity and completion state govern reuse."""

from dataclasses import replace

import pytest

from models.analysis_record import AnalysisIdentity, AnalysisRecord


def identity(**changes):
    arguments = dict(
        operation="detect_objects", operation_version=1,
        sources={"video": "a" * 64, "image": "b" * 64},
        source_range={"start_frame": 24, "end_frame": 48, "rate": "24"},
        model={"name": "detector", "version": "one"},
        parameters={"confidence": 0.5}, sampling={"frames": [24]}, prompt=None,
    )
    return AnalysisIdentity.build(**(arguments | changes))


def test_valid_empty_is_reusable_but_failed_or_missing_is_not():
    key = identity()
    success = AnalysisRecord.success(key, {"detected_objects": []})
    assert success.reusable(key)
    assert success.value == {"detected_objects": []}
    assert not AnalysisRecord.failure(key, "provider unavailable").reusable(key)
    assert not replace(success, state="missing").reusable(key)
    assert AnalysisRecord.from_dict(success.to_dict()) == success


@pytest.mark.parametrize("change", [
    {"sources": {"video": "c" * 64, "image": "b" * 64}},
    {"source_range": {"start_frame": 25, "end_frame": 48, "rate": "24"}},
    {"model": {"name": "detector", "version": "two"}},
    {"operation_version": 2}, {"schema_version": 2},
    {"parameters": {"confidence": 0.8}}, {"sampling": {"frames": [25]}},
    {"prompt": "find people"},
])
def test_input_and_computation_changes_invalidate(change):
    original = identity()
    changed = identity(**change)
    assert original.key != changed.key
    assert not AnalysisRecord.success(original, []).reusable(changed)


def test_identity_detaches_mutable_values_and_normalizes_dictionary_order():
    parameters = {"b": [1, 2], "a": 3}
    first = identity(parameters=parameters)
    second = identity(parameters={"a": 3, "b": [1, 2]})
    parameters["b"].append(4)
    assert first == second and first.key == second.key
    with pytest.raises(ValueError):
        identity(parameters={"nan": float("nan")})


def test_embedding_reuse_checks_source_trim_and_verified_record(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from core.spine.analyze import embeddings
    from tests.test_description_operations import project_with_thumbnails

    project = project_with_thumbnails(tmp_path, 1)
    clip = project.clips[0]
    clip.embedding = [0.2] * 768  # A legacy projection must be recomputed.
    clip.embedding_model = "dinov2-vit-b-14"
    compute = Mock(side_effect=lambda paths: [[0.1] * 768 for _ in paths])
    monkeypatch.setattr("core.analysis.embeddings.extract_clip_embeddings_batch", compute)
    monkeypatch.setattr("core.analysis.embeddings.unload_model", lambda: None)
    assert len(embeddings(project)["result"]["succeeded"]) == 1
    assert clip.analysis_records["embeddings"].provenance == "verified"
    assert len(embeddings(project)["result"]["skipped"]) == 1
    assert compute.call_count == 1
    clip.start_frame += 1
    assert len(embeddings(project)["result"]["succeeded"]) == 1
    project.sources[0].file_path.write_bytes(b"changed source")
    assert len(embeddings(project)["result"]["succeeded"]) == 1
    assert compute.call_count == 3


def test_saved_embedding_jobs_recompute_legacy_and_reuse_without_old_job_cache(tmp_path, monkeypatch):
    from threading import Event
    from unittest.mock import Mock
    from core.jobs.embeddings import run_embedding_job
    from core.jobs.store import JobStore
    from tests.test_description_operations import project_with_thumbnails

    project = project_with_thumbnails(tmp_path, 1)
    project.clips[0].embedding = [0.2] * 768
    project.clips[0].embedding_model = "dinov2-vit-b-14"
    path = tmp_path / "project.json"
    assert project.save(path)
    compute = Mock(side_effect=lambda paths: [[0.1] * 768 for _ in paths])
    monkeypatch.setattr("core.analysis.embeddings.extract_clip_embeddings_batch", compute)
    monkeypatch.setattr("core.analysis.embeddings.unload_model", lambda: None)
    store = JobStore(tmp_path / "jobs.db")
    try:
        assert len(run_embedding_job(store, path, None, lambda *_: None, Event())["result"]["succeeded"]) == 1
    finally:
        store.close()
    fresh = JobStore(tmp_path / "empty-jobs.db")
    try:
        assert len(run_embedding_job(fresh, path, None, lambda *_: None, Event())["result"]["skipped"]) == 1
        assert compute.call_count == 1
    finally:
        fresh.close()


def test_saved_embedding_payload_is_external_and_restores_exact_values(tmp_path, monkeypatch):
    import json
    from core.project import Project
    from core.spine.analyze import embeddings
    from tests.test_description_operations import project_with_thumbnails

    root = tmp_path / "artifacts"
    monkeypatch.setattr("core.paths.get_artifact_store_dir", lambda: root)
    project = project_with_thumbnails(tmp_path, 1)
    vector = [0.1234567890123] * 768
    monkeypatch.setattr("core.analysis.embeddings.extract_clip_embeddings_batch", lambda paths: [vector for _ in paths])
    monkeypatch.setattr("core.analysis.embeddings.unload_model", lambda: None)
    embeddings(project)
    path = tmp_path / "project.json"
    assert project.save(path)
    saved = json.loads(path.read_text())["clips"][0]
    assert "embedding" not in saved
    record = AnalysisRecord.from_dict(saved["analysis_records"]["embeddings"])
    assert record.value is None and record.artifact is not None
    restored = Project.load(path)
    assert restored.clips[0].embedding == vector
    monkeypatch.setattr("core.analysis.embeddings.extract_clip_embeddings_batch", lambda *_: pytest.fail("valid artifact must avoid inference"))
    assert len(embeddings(restored)["result"]["skipped"]) == 1


def test_missing_embedding_recomputes_only_its_referencing_clip(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from core.project import Project
    from core.spine.analyze import embeddings
    from tests.test_description_operations import project_with_thumbnails

    monkeypatch.setattr("core.paths.get_artifact_store_dir", lambda: tmp_path / "artifacts")
    project = project_with_thumbnails(tmp_path, 2)
    project.clips[0].notes = "Keep the edit"
    project.add_to_sequence([clip.id for clip in project.clips])
    compute = Mock(side_effect=lambda paths: [[(i + 1) / 10] * 768 for i, _ in enumerate(paths)])
    monkeypatch.setattr("core.analysis.embeddings.extract_clip_embeddings_batch", compute)
    monkeypatch.setattr("core.analysis.embeddings.unload_model", lambda: None)
    embeddings(project)
    assert project.save(tmp_path / "project.json")
    loaded = Project.load(project.path)
    ref = loaded.clips[0].analysis_records["embeddings"].artifact
    loaded.artifact_store.path_for(ref).unlink()
    recovered = Project.load(project.path)
    result = embeddings(recovered)["result"]
    assert len(result["succeeded"]) == len(result["skipped"]) == 1
    assert len(compute.call_args.args[0]) == 1
    assert recovered.clips[0].notes == "Keep the edit"
    assert len(recovered.sequence.get_all_clips()) == 2


def test_embedding_loader_uses_the_recorded_model_revision(monkeypatch):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock
    import core.analysis.embeddings as backend
    from core.analysis_model_identity import embedding_runtime

    processor, model = Mock(), Mock()
    monkeypatch.setattr(backend, "_model", None)
    monkeypatch.setattr(backend, "_processor", None)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoImageProcessor=processor, AutoModel=model))
    backend._get_model()
    expected = embedding_runtime()
    processor.from_pretrained.assert_called_once_with(expected["model"], revision=expected["revision"])
    model.from_pretrained.assert_called_once_with(expected["model"], revision=expected["revision"])


def test_legacy_values_require_explicit_acceptance_and_remain_unknown():
    legacy = AnalysisRecord.legacy({"dominant_colors": [[1, 2, 3]]})
    key = identity(operation="colors")
    assert not legacy.reusable(key)
    accepted = legacy.accept_legacy(key)
    assert accepted.reusable(key) and accepted.provenance == "unknown"
    assert not accepted.reusable(identity(operation="colors", operation_version=2))
    assert AnalysisRecord.from_dict(accepted.to_dict()) == accepted


def test_model_roundtrip_preserves_records_and_unknown_record_bytes():
    from models.clip import Clip
    from models.frame import Frame
    from models.audio_source import AudioSource

    key = identity()
    for model in (Clip(), Frame(), AudioSource()):
        model.analysis_records["detect_objects"] = AnalysisRecord.success(key, [])
        serialized = model.to_dict()
        serialized["analysis_records"]["future_operation"] = {"version": 100, "new": [1, 2]}
        restored = type(model).from_dict(serialized)
        assert restored.analysis_records["detect_objects"].reusable(key)
        assert restored.to_dict()["analysis_records"]["future_operation"] == {"version": 100, "new": [1, 2]}


def test_legacy_migration_preserves_notes_sequences_and_empty_results():
    from core.project_migrations import migrate_project_data
    from models.clip import Clip

    original = {
        "version": "1.6", "clips": [{"id": "clip", "source_id": "source", "notes": "keep", "detected_objects": []}],
        "sequences": [{"id": "edit", "tracks": []}],
    }
    migrated = migrate_project_data(original)
    assert original["version"] == "1.6" and "analysis_records" not in original["clips"][0]
    assert migrated["sequences"] == original["sequences"]
    clip = Clip.from_dict(migrated["clips"][0])
    assert clip.notes == "keep" and clip.detected_objects == []
    assert clip.analysis_records["detect_objects"].provenance == "unknown"
    assert not clip.analysis_records["detect_objects"].reusable(identity())


def test_color_reuse_tracks_content_range_parameters_and_projection(tmp_path, monkeypatch):
    import os
    from unittest.mock import Mock
    from core.operations.colors import ColorApplication, color_request, compute_colors
    from core.analysis_availability import operation_is_complete_for_clip
    from tests.test_spine_analyze import _build_project

    project = _build_project(tmp_path, 1, populate_colors=1)
    extract = Mock(return_value=[(1, 2, 3)])
    monkeypatch.setattr("core.analysis.color.extract_dominant_colors", extract)

    def run(count=5):
        request = color_request(project, num_colors=count)
        return ColorApplication(project, request).apply(compute_colors(request)).outcomes[0]

    assert run().status == "succeeded"  # Legacy values are not proof of reuse.
    record = project.clips[0].analysis_records["colors"]
    assert record.provenance == "verified"
    assert operation_is_complete_for_clip("colors", project.clips[0])
    assert run().status == "skipped" and extract.call_count == 1
    media = project.sources[0].file_path
    previous = media.stat()
    media.write_bytes(b"new!")  # Same size and restored mtime still invalidate.
    os.utime(media, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert not operation_is_complete_for_clip("colors", project.clips[0])
    assert run().status == "succeeded" and extract.call_count == 2
    assert project.clips[0].analysis_records["colors"].identity != record.identity
    project.clips[0].start_frame += 1
    assert run().status == "succeeded" and extract.call_count == 3
    assert run(3).status == "succeeded" and extract.call_count == 4
    assert run(3).status == "skipped"
    project.clips[0].dominant_colors = [(9, 9, 9)]
    assert run(3).status == "succeeded" and extract.call_count == 5


def test_revalidation_refreshes_stamps_without_repeating_color_inference(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from core.analysis_availability import operation_is_complete_for_clip
    from core.operations.colors import ColorApplication, color_request, compute_colors
    from tests.test_spine_analyze import _build_project

    project = _build_project(tmp_path, 1)
    extract = Mock(return_value=[(1, 2, 3)])
    monkeypatch.setattr("core.analysis.color.extract_dominant_colors", extract)

    def run():
        request = color_request(project)
        return ColorApplication(project, request).apply(compute_colors(request)).outcomes[0]

    run()
    project.sources[0].file_path.touch()
    assert not operation_is_complete_for_clip("colors", project.clips[0])
    assert run().status == "skipped" and extract.call_count == 1
    assert operation_is_complete_for_clip("colors", project.clips[0])
    project.clips[0].source_id = "relinked"
    assert not operation_is_complete_for_clip("colors", project.clips[0])


def test_saved_color_jobs_record_provenance_and_recover_over_legacy_values(tmp_path, monkeypatch):
    from threading import Event
    from unittest.mock import Mock
    from core.jobs.colors import run_colors
    from core.jobs.store import JobStore
    from core.project import Project, ProjectSaveError
    from tests.test_spine_analyze import _build_project

    path = tmp_path / "project.sceneripper"
    project = _build_project(tmp_path, 1, populate_colors=1)
    project.clips[0].notes = "keep me"
    project.save(path)
    store = JobStore(tmp_path / "jobs.db")
    extract = Mock(return_value=[(1, 2, 3)])
    monkeypatch.setattr("core.analysis.color.extract_dominant_colors", extract)
    save = Project.save
    monkeypatch.setattr(Project, "save", lambda *args, **kwargs: False)
    with pytest.raises(ProjectSaveError):
        run_colors(store, path, ["c-0"], 5, lambda *args: None, Event())
    monkeypatch.setattr(Project, "save", save)
    run_colors(store, path, ["c-0"], 5, lambda *args: None, Event())
    assert extract.call_count == 1
    restored = Project.load(path)
    assert restored.clips[0].analysis_records["colors"].provenance == "verified"
    assert restored.clips[0].notes == "keep me"


def test_embedding_completion_checks_current_thumbnail_path(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from core.analysis_availability import operation_is_complete_for_clip
    from core.spine.analyze import embeddings
    from tests.test_description_operations import project_with_thumbnails

    project = project_with_thumbnails(tmp_path, 1)
    monkeypatch.setattr("core.analysis.embeddings.extract_clip_embeddings_batch", Mock(return_value=[[0.1] * 768]))
    monkeypatch.setattr("core.analysis.embeddings.unload_model", Mock())
    embeddings(project)
    clip = project.clips[0]
    assert operation_is_complete_for_clip("embeddings", clip)
    original = clip.thumbnail_path
    replacement = tmp_path / "replacement.jpg"
    replacement.write_bytes(b"new thumbnail")
    clip.thumbnail_path = replacement
    assert original.exists()
    assert not operation_is_complete_for_clip("embeddings", clip)

"""Relocated face analysis rebinds only after full media and model verification."""

import shutil
from unittest.mock import Mock

import pytest

from core.project import Project
from core.project_export import export_project_bundle
from core.analysis_records import AnalysisFingerprints
from core.operations.face_records import face_snapshot, reusable_face_record
from core.operations import faces
from tests.test_face_records import setup as face_setup, run  # noqa: F401


@pytest.fixture
def relocated(request, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.paths.get_artifact_store_dir", lambda: tmp_path / "artifacts"
    )
    project, provider, directory = request.getfixturevalue("face_setup")
    run(project)
    destination = tmp_path / "bundle"
    export_project_bundle(project, destination, include_clips=False)
    new_root = tmp_path / "new-models"
    new_directory = new_root / "insightface" / "models" / "buffalo_l"
    shutil.copytree(directory, new_directory)
    shutil.rmtree(directory)
    project.sources[0].file_path.unlink()
    monkeypatch.setattr("core.analysis.faces._get_model_cache_dir", lambda: new_root)
    restored = Project.load(destination / f"{project.metadata.name}.sceneripper")
    return restored, provider, new_directory


def test_relocated_faces_reuse_without_inference(relocated, monkeypatch):
    project, provider, directory = relocated
    assert project.clips[0].analysis_records["face_embeddings"].input_json is None
    blocked = Mock(side_effect=AssertionError("relocation must not require inference"))
    monkeypatch.setattr("core.analysis.faces._load_insightface", blocked)
    outcome, _ = run(project)
    assert outcome.status == "skipped", outcome
    assert provider.call_count == 1
    assert str(directory) in outcome.record_json
    from core.analysis_availability import face_analysis_is_complete

    assert face_analysis_is_complete(project.clips[0], project.sources[0])
    blocked.assert_not_called()


@pytest.mark.parametrize(
    "change", ["video", "weights", "extra_weight", "range", "sampling", "packages"]
)
def test_relocation_does_not_hide_changed_inputs(relocated, change):
    from threading import Event

    project, _, directory = relocated
    clip, source = project.clips[0], project.sources[0]
    environment = faces.face_environment()
    interval = 1.0
    if change == "video":
        source.file_path.write_bytes(b"changed video")
    elif change == "weights":
        (directory / "recognition.onnx").write_bytes(b"changed weights")
    elif change == "extra_weight":
        (directory / "extra.onnx").write_bytes(b"extra")
    elif change == "range":
        clip.end_frame -= 1
    elif change == "sampling":
        interval = 0.5
    else:
        environment = {**environment, "packages": {"different": "1"}}
    assert (
        reusable_face_record(
            face_snapshot(clip, source),
            interval,
            AnalysisFingerprints(Event()),
            environment,
        )
        is None
    )

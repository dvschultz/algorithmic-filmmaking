"""Face completion checks saved bindings without running inference or file hashes."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.analysis_availability import face_analysis_is_complete, operation_is_complete_for_clip, compute_operation_need_counts
from tests.test_face_records import setup as face_setup, run  # noqa: F401


@pytest.fixture
def analyzed(request):
    project, provider, directory = request.getfixturevalue("face_setup")
    run(project)
    return project, provider, directory


def test_completion_never_hashes_or_loads_runtime(analyzed, monkeypatch):
    import builtins
    project, _, _ = analyzed
    blocked = Mock(side_effect=AssertionError("must remain cheap"))
    monkeypatch.setattr("core.jobs.media.MediaFingerprints.get", blocked)
    monkeypatch.setattr("core.analysis.face_weights.FaceWeights.capture", blocked)
    monkeypatch.setattr("core.analysis.faces._load_insightface", blocked)
    original = builtins.__import__
    def imported(name, *args, **kwargs):
        assert not name.startswith(("onnxruntime", "insightface")), name
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", imported)
    assert face_analysis_is_complete(project.clips[0], project.sources[0])
    assert compute_operation_need_counts(project.clips, ["face_embeddings"], sources_by_id=project.sources_by_id) == {"face_embeddings": 0}
    blocked.assert_not_called()


@pytest.mark.parametrize("change", ["legacy", "failed", "media", "path", "fps", "range", "source_id", "source", "weights", "new_weight", "missing_weight", "value", "packages", "model_path"])
def test_stale_faces_are_pending(analyzed, tmp_path, monkeypatch, change):
    project, _, directory = analyzed
    clip, source = project.clips[0], project.sources[0]
    if change == "legacy":
        clip.analysis_records.clear()
    elif change == "failed":
        clip.analysis_records["face_embeddings"] = replace(clip.analysis_records["face_embeddings"], state="failed")
    elif change == "media":
        source.file_path.write_bytes(b"changed")
    elif change == "path":
        source.file_path = tmp_path / "different.mp4"
    elif change == "fps":
        source.fps = 24
    elif change == "range":
        clip.end_frame -= 1
    elif change == "source_id":
        source = replace(source, id="different")
    elif change == "source":
        source = None
    elif change == "weights":
        (directory / "recognition.onnx").write_bytes(b"changed")
    elif change == "new_weight":
        (directory / "new.onnx").write_bytes(b"added")
    elif change == "missing_weight":
        (directory / "recognition.onnx").unlink()
    elif change == "packages":
        monkeypatch.setattr("core.operations.face_records.face_packages", lambda: {"changed": "1"})
    elif change == "model_path":
        monkeypatch.setattr("core.analysis.faces._get_model_cache_dir", lambda: tmp_path / "new-models")
    else:
        clip.face_embeddings = []
    assert not operation_is_complete_for_clip("face_embeddings", clip, source=source)


def test_requested_sampling_interval_must_match(analyzed):
    project, _, _ = analyzed
    assert not face_analysis_is_complete(project.clips[0], project.sources[0], sample_interval=0.5)
    from core.operations.faces import FaceOptions
    run(project, options=FaceOptions(0.5))
    assert face_analysis_is_complete(project.clips[0], project.sources[0], sample_interval=0.5)
    assert not operation_is_complete_for_clip("face_embeddings", project.clips[0], source=project.sources[0])


def test_verified_empty_faces_are_complete_in_mcp(analyzed, tmp_path):
    import asyncio
    import json
    from scene_ripper_mcp.tools.analyze import get_analysis_status
    project, provider, _ = analyzed
    original = provider.side_effect
    provider.side_effect = lambda **kwargs: original(**kwargs) and []
    run(project, skip=False)
    assert project.clips[0].face_embeddings == []
    assert face_analysis_is_complete(project.clips[0], project.sources[0])
    path = tmp_path / "project.sceneripper"
    project.save(path)
    project.close_writer()
    status = json.loads(asyncio.run(get_analysis_status(str(path))))
    assert status["success"], status
    assert status["analysis"]["faces"] == {"analyzed": 1, "pending": 0}
    project.clips[0].analysis_records.clear()
    project.save(path)
    project.close_writer()
    status = json.loads(asyncio.run(get_analysis_status(str(path))))
    assert status["analysis"]["faces"] == {"analyzed": 0, "pending": 1}

"""Verified face results bind media, actual weights, and persisted projections."""

from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.analysis.face_weights import FaceWeights
from core.analysis.faces import face_model_execution
from core.operations.faces import FaceApplication, FaceOptions, face_task, run_faces
from core.operations.face_records import face_packages
from tests.test_spine_analyze import _build_project


def test_isolated_face_environment_does_not_import_onnx_in_host(monkeypatch):
    import builtins

    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKERS", "1")
    monkeypatch.setenv("SCENE_RIPPER_NATIVE_WORKER_FAMILIES", "vision")
    monkeypatch.delenv("SCENE_RIPPER_WORKER_PROCESS", raising=False)
    attempted = []
    actual_import = builtins.__import__

    def host_import(name, *args, **kwargs):
        if name == "onnxruntime":
            attempted.append(name)
            raise AssertionError("ONNX Runtime must only be imported by the vision worker")
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", host_import)
    monkeypatch.setattr(
        "core.runtime_supervisor.default_supervisor",
        lambda: SimpleNamespace(
            run=lambda *args, **kwargs: {
                "value": {
                    "packages": face_packages(),
                    "available_providers": ["CPUExecutionProvider"],
                }
            }
        ),
    )

    from core.operations.face_records import face_environment

    assert face_environment()["available_providers"] == ["CPUExecutionProvider"]
    assert attempted == []


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = _build_project(tmp_path, 1)
    directory = tmp_path / "insightface" / "models" / "buffalo_l"
    directory.mkdir(parents=True)
    for name in ("detector", "recognition"):
        (directory / f"{name}.onnx").write_bytes(name.encode())
    monkeypatch.setattr("core.analysis.faces._get_model_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "core.operations.faces.face_environment",
        lambda: {
            "packages": face_packages(),
            "available_providers": ["CPUExecutionProvider"],
        },
    )
    monkeypatch.setattr("core.analysis.faces._load_insightface", Mock())
    monkeypatch.setattr("core.analysis.faces.unload_model", Mock())

    def execute(**kwargs):
        weights = FaceWeights.capture(directory)
        model = SimpleNamespace(
            _scene_ripper_weights=weights,
            models={
                task: SimpleNamespace(
                    model_file=str(directory / f"{name}.onnx"),
                    session=SimpleNamespace(
                        get_providers=lambda: ["CPUExecutionProvider"]
                    ),
                )
                for task, name in (
                    ("detection", "detector"),
                    ("recognition", "recognition"),
                )
            },
        )
        kwargs["on_execution"](face_model_execution(model))
        return [
            {
                "bbox": [1, 2, 3, 4],
                "embedding": [0.123456789] * 512,
                "confidence": 0.9,
                "frame_number": kwargs["start_frame"],
            }
        ]

    provider = Mock(side_effect=execute)
    monkeypatch.setattr("core.analysis.faces.extract_faces_from_clip", provider)
    return project, provider, directory


def run(project, *, skip=True, options=FaceOptions(), apply=True, **kwargs):
    task = face_task(project.clips[0], project.sources[0], skip_existing=skip)
    application = FaceApplication(project, (task,), options)
    result = run_faces((task,), options, **kwargs)[0]
    if apply:
        assert application.apply(project, result), result
    return result, application


@pytest.mark.parametrize("empty", [False, True])
def test_results_reuse_after_save_without_loading_model(
    setup, tmp_path, monkeypatch, empty
):
    from core.project import Project

    project, provider, _ = setup
    if empty:
        original = provider.side_effect
        provider.side_effect = lambda **kwargs: original(**kwargs) and []
    assert run(project)[0].status == "succeeded"
    path = tmp_path / "project.sceneripper"
    project.save(path)
    project.close_writer()
    restored = Project.load(path)
    monkeypatch.setattr(
        "core.analysis.faces._load_insightface",
        Mock(side_effect=AssertionError("must reuse")),
    )
    result, _ = run(restored)
    assert result.status == "skipped" and result.code == "valid_analysis"
    assert provider.call_count == 1
    if not empty:
        assert restored.clips[0].face_embeddings[0]["embedding"][0] == 0.12346


@pytest.mark.parametrize(
    "change",
    ["legacy", "media", "range", "fps", "weights", "new_weight", "value", "failed"],
)
def test_changed_inputs_recompute(setup, change):
    project, provider, directory = setup
    run(project)
    clip, source = project.clips[0], project.sources[0]
    if change == "legacy":
        clip.analysis_records.clear()
    elif change == "media":
        source.file_path.write_bytes(b"different media")
    elif change == "range":
        clip.end_frame -= 1
    elif change == "fps":
        source.fps = 24
    elif change == "weights":
        (directory / "recognition.onnx").write_bytes(b"new weights")
    elif change == "new_weight":
        (directory / "extra.onnx").write_bytes(b"added")
    elif change == "value":
        clip.face_embeddings = []
    else:
        clip.analysis_records["face_embeddings"] = replace(
            clip.analysis_records["face_embeddings"], state="failed"
        )
    assert run(project)[0].status == "succeeded"
    assert provider.call_count == 2


def test_identical_replaced_weight_refreshes_binding_without_inference(setup):
    project, provider, directory = setup
    run(project)
    path = directory / "recognition.onnx"
    content = path.read_bytes()
    path.unlink()
    path.write_bytes(content)
    assert run(project)[0].status == "skipped"
    assert provider.call_count == 1


def test_owned_failure_preserves_faces(setup):
    project, provider, _ = setup
    run(project)
    before = project.clips[0].face_embeddings
    provider.side_effect = RuntimeError("provider failed")
    assert run(project, skip=False)[0].status == "failed"
    assert project.clips[0].face_embeddings == before
    assert project.clips[0].analysis_records["face_embeddings"].state == "failed"


@pytest.mark.parametrize("change", ["value", "record", "media", "weights", "save_as"])
def test_late_changes_reject_publication(setup, tmp_path, change):
    project, _, directory = setup
    result, application = run(project, apply=False)
    if change == "value":
        project.clips[0].face_embeddings = []
    elif change == "record":
        from models.analysis_record import AnalysisRecord

        project.clips[0].analysis_records["face_embeddings"] = AnalysisRecord.legacy(
            {"face_embeddings": []}
        )
    elif change == "media":
        project.sources[0].file_path.write_bytes(b"changed")
    elif change == "weights":
        (directory / "recognition.onnx").write_bytes(b"changed")
    else:
        project.save(tmp_path / "new.sceneripper")
    assert not application.apply(project, result)


def test_cancelled_inference_does_not_publish(setup):
    project, provider, _ = setup
    cancel = Event()
    original = provider.side_effect

    def compute(**kwargs):
        value = original(**kwargs)
        cancel.set()
        return value

    provider.side_effect = compute
    result, application = run(project, apply=False, cancel_event=cancel)
    assert result.status == "unprocessed"
    assert not application.apply(project, result)
    assert project.clips[0].face_embeddings is None


def test_observers_see_matching_record_once(setup, monkeypatch):
    project, _, _ = setup
    seen = []
    original = project.update_clips

    def observe(clips):
        clip = clips[0]
        seen.append(
            clip.analysis_records["face_embeddings"].value
            == {"face_embeddings": clip.face_embeddings}
        )
        return original(clips)

    monkeypatch.setattr(project, "update_clips", observe)
    result, application = run(project)
    assert seen == [True]
    assert not application.apply(project, result)


def test_spine_uses_verified_reuse(setup):
    from core.spine.analyze import face_embeddings

    project, provider, _ = setup
    assert len(face_embeddings(project)["result"]["succeeded"]) == 1
    assert len(face_embeddings(project)["result"]["skipped"]) == 1
    assert provider.call_count == 1


@pytest.mark.parametrize(
    "change",
    ["no_execution", "confidence", "outside", "unsampled", "weights_during_inference"],
)
def test_unverifiable_provider_output_is_not_published(setup, change):
    project, provider, directory = setup
    original = provider.side_effect

    def compute(**kwargs):
        if change == "no_execution":
            return []
        value = original(**kwargs)
        if change == "confidence":
            value[0]["confidence"] = True
        elif change == "outside":
            value[0]["frame_number"] = project.clips[0].end_frame
        elif change == "unsampled":
            value[0]["frame_number"] = project.clips[0].start_frame + 1
        else:
            (directory / "recognition.onnx").write_bytes(b"changed during inference")
        return value

    provider.side_effect = compute
    result, _ = run(project)
    assert result.status == "failed"
    assert project.clips[0].face_embeddings is None
    assert project.clips[0].analysis_records["face_embeddings"].state == "failed"


def test_changed_sampling_recomputes(setup):
    project, provider, _ = setup
    run(project)
    assert run(project, options=FaceOptions(0.5))[0].status == "succeeded"
    assert provider.call_count == 2


def test_missing_runtime_retains_owned_failure(setup, monkeypatch):
    project, provider, _ = setup
    monkeypatch.setattr(
        "core.operations.faces.face_environment",
        Mock(side_effect=ImportError("runtime missing")),
    )
    assert run(project)[0].status == "failed"
    provider.assert_not_called()
    assert project.clips[0].analysis_records["face_embeddings"].state == "failed"

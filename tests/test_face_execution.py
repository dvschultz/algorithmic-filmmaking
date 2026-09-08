"""Face model initialization publishes only prepared runtimes."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.analysis import faces


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(faces, "_model", None)
    monkeypatch.setattr(faces, "_get_model_cache_dir", lambda: tmp_path)
    from core.analysis.face_weights import FaceWeights
    (tmp_path / "recognition.onnx").write_bytes(b"weights")
    monkeypatch.setattr(faces, "_stage_face_weights", lambda _: FaceWeights.capture(tmp_path))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ))
    return tmp_path


def test_failed_prepare_never_leaves_cached_model(runtime, monkeypatch):
    factory = Mock(side_effect=lambda **kw: SimpleNamespace(prepare=Mock(side_effect=RuntimeError("prepare failed"))))
    monkeypatch.setattr(faces, "ensure_face_detection_runtime_available", lambda: factory)
    for _ in range(2):
        with pytest.raises(Exception, match="prepare failed"):
            faces._load_insightface()
        assert faces._model is None
    assert factory.call_count == 4


def test_accelerator_failure_reports_actual_cpu_components(runtime, monkeypatch):
    failed = SimpleNamespace(prepare=Mock(side_effect=RuntimeError("accelerator failed")))
    component = SimpleNamespace(
        model_file=str(runtime / "recognition.onnx"),
        session=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]),
    )
    prepared = SimpleNamespace(prepare=Mock(), models={"recognition": component})
    factory = Mock(side_effect=[failed, prepared])
    monkeypatch.setattr(faces, "ensure_face_detection_runtime_available", lambda: factory)
    assert faces._load_insightface() is prepared
    assert faces._load_insightface() is prepared
    assert factory.call_count == 2
    report = faces.face_model_execution(prepared)
    assert report["model"] == "buffalo_l"
    assert report["components"] == [{
        "task": "recognition", "path": str(runtime / "recognition.onnx"),
        "providers": ["CPUExecutionProvider"],
        "weights": prepared._scene_ripper_weights.component(runtime / "recognition.onnx"),
    }]


def test_video_reports_execution_before_inference(runtime, monkeypatch):
    events = []
    model = SimpleNamespace(models={}, get=lambda frame: events.append("infer") or [])
    monkeypatch.setattr(faces, "_load_insightface", lambda: model)
    cap = SimpleNamespace(isOpened=lambda: True, set=lambda *a: True,
        read=lambda: (True, object()), release=Mock())
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=lambda _: cap, CAP_PROP_POS_FRAMES=1))
    assert faces.extract_faces_from_clip(runtime / "video.mp4", 0, 1, 30,
        on_execution=lambda data: events.append(data)) == []
    assert events[0]["backend"] == "insightface"
    assert events[0]["components"] == []
    assert events[1:] == ["infer"]
    cap.release.assert_called_once()


def test_failed_execution_capture_releases_video_without_inference(runtime, monkeypatch):
    model = SimpleNamespace(models={}, get=Mock())
    monkeypatch.setattr(faces, "_load_insightface", lambda: model)
    cap = SimpleNamespace(isOpened=lambda: True, release=Mock())
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=lambda _: cap))
    with pytest.raises(ValueError, match="capture failed"):
        faces.extract_faces_from_clip(runtime / "video.mp4", 0, 1, 30,
            on_execution=Mock(side_effect=ValueError("capture failed")))
    cap.release.assert_called_once()
    model.get.assert_not_called()


def test_image_reports_execution_before_inference(runtime, monkeypatch):
    events = []
    model = SimpleNamespace(models={}, get=lambda image: events.append("infer") or [])
    monkeypatch.setattr(faces, "_load_insightface", lambda: model)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(imread=lambda _: SimpleNamespace(shape=(20, 20, 3))))
    assert faces.extract_faces_from_image(runtime / "image.jpg",
        on_execution=lambda data: events.append(data)) == []
    assert events[0]["backend"] == "insightface"
    assert events[1:] == ["infer"]


@pytest.mark.parametrize("change", ["replace", "add", "remove"])
def test_weight_change_invalidates_loaded_execution(runtime, monkeypatch, change):
    def build(**kwargs):
        component = SimpleNamespace(model_file=str(runtime / "recognition.onnx"),
            session=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]))
        return SimpleNamespace(prepare=Mock(), models={"recognition": component})

    factory = Mock(side_effect=build)
    monkeypatch.setattr(faces, "ensure_face_detection_runtime_available", lambda: factory)
    loaded = faces._load_insightface()
    original = faces.face_model_execution(loaded)["components"][0]["weights"]["sha256"]
    if change == "replace":
        (runtime / "recognition.onnx").write_bytes(b"new weights")
    elif change == "add":
        (runtime / "another.onnx").write_bytes(b"new component")
    else:
        (runtime / "recognition.onnx").unlink()
    with pytest.raises(ValueError, match="changed after initialization"):
        faces.face_model_execution(loaded)
    if change == "remove":
        with pytest.raises(ValueError, match="no ONNX"):
            faces._load_insightface()
        assert faces._model is None
    else:
        refreshed = faces._load_insightface()
        assert refreshed is not loaded
        assert factory.call_count == 2
        if change == "replace":
            assert faces.face_model_execution(refreshed)["components"][0]["weights"]["sha256"] != original


def test_weights_changed_during_prepare_are_not_published(runtime, monkeypatch):
    def prepare(**kwargs):
        (runtime / "recognition.onnx").write_bytes(b"changed while loading")

    candidate = SimpleNamespace(prepare=prepare)
    monkeypatch.setattr(faces, "ensure_face_detection_runtime_available", lambda: Mock(return_value=candidate))
    with pytest.raises(ValueError, match="changed during initialization"):
        faces._load_insightface()
    assert faces._model is None


def test_component_outside_captured_pack_is_rejected(runtime):
    from core.analysis.face_weights import FaceWeights
    weights = FaceWeights.capture(runtime)
    with pytest.raises(ValueError, match="not part"):
        weights.component(runtime / "other.onnx")


def test_loaded_execution_does_not_rehash_weights(runtime, monkeypatch):
    from core.analysis.face_weights import FaceWeights
    from hashlib import sha256
    weights = FaceWeights.capture(runtime)
    component = SimpleNamespace(model_file=str(runtime / "recognition.onnx"),
        session=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]))
    model = SimpleNamespace(models={"recognition": component}, _scene_ripper_weights=weights)
    monkeypatch.setattr("core.jobs.media.MediaFingerprints.get", Mock(side_effect=AssertionError("must not rehash")))
    assert faces.face_model_execution(model)["components"][0]["weights"]["sha256"] == sha256(b"weights").hexdigest()

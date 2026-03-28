"""Tests for analysis-operation dependency resolution."""

from types import SimpleNamespace

from core.analysis_dependencies import get_operation_feature_candidates


def _settings(**overrides):
    defaults = {
        "transcription_backend": "auto",
        "description_model_tier": "local",
        "text_extraction_method": "hybrid",
        "cinematography_tier": "cloud",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_shot_and_content_classification_map_to_local_ml_features():
    settings = _settings()

    assert get_operation_feature_candidates("shots", settings) == ["shot_classify"]
    assert get_operation_feature_candidates("classify", settings) == ["image_classify"]


def test_detection_and_face_ops_map_to_their_features():
    settings = _settings()

    assert get_operation_feature_candidates("detect_objects", settings) == ["object_detect"]
    assert get_operation_feature_candidates("face_embeddings", settings) == ["face_detect"]
    assert get_operation_feature_candidates("embeddings", settings) == ["embeddings"]
    assert get_operation_feature_candidates("boundary_embeddings", settings) == ["embeddings"]


def test_text_extraction_only_requires_ocr_in_paddleocr_mode():
    assert get_operation_feature_candidates(
        "extract_text",
        _settings(text_extraction_method="paddleocr"),
    ) == ["ocr"]
    assert get_operation_feature_candidates(
        "extract_text",
        _settings(text_extraction_method="hybrid"),
    ) == []


def test_transcription_auto_prefers_mlx_then_faster_whisper_on_apple(monkeypatch):
    monkeypatch.setattr("core.analysis_dependencies.platform.system", lambda: "Darwin")
    monkeypatch.setattr("core.analysis_dependencies.platform.machine", lambda: "arm64")

    assert get_operation_feature_candidates("transcribe", _settings()) == [
        "transcribe_mlx",
        "transcribe",
    ]


def test_transcription_backend_specific_features():
    settings = _settings(transcription_backend="faster-whisper")
    assert get_operation_feature_candidates("transcribe", settings) == ["transcribe"]

    settings = _settings(transcription_backend="mlx-whisper")
    assert get_operation_feature_candidates("transcribe", settings) == ["transcribe_mlx"]

    settings = _settings(transcription_backend="groq")
    assert get_operation_feature_candidates("transcribe", settings) == ["transcribe_cloud"]


def test_local_description_and_cinematography_only_gate_on_apple_silicon(monkeypatch):
    monkeypatch.setattr("core.analysis_dependencies.platform.system", lambda: "Windows")
    monkeypatch.setattr("core.analysis_dependencies.platform.machine", lambda: "AMD64")

    assert get_operation_feature_candidates("describe", _settings()) == [
        "describe_local_cpu"
    ]
    assert get_operation_feature_candidates(
        "cinematography",
        _settings(cinematography_tier="local"),
    ) == []

    monkeypatch.setattr("core.analysis_dependencies.platform.system", lambda: "Darwin")
    monkeypatch.setattr("core.analysis_dependencies.platform.machine", lambda: "arm64")

    assert get_operation_feature_candidates("describe", _settings()) == [
        "describe_local",
        "describe_local_cpu",
    ]
    assert get_operation_feature_candidates(
        "cinematography",
        _settings(cinematography_tier="local"),
    ) == ["describe_local"]

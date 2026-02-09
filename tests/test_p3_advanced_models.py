"""Tests pinning P3 advanced model upgrades.

Covers:
- P3.1: Local cinematography analysis with configurable VLM
- P3.2: YOLOE-26 open-vocabulary detection
- P3.3: Groq cloud transcription backend
"""

import json

from core.settings import Settings, _settings_to_json, _load_from_json


# --- P3.1: Local cinematography analysis ---

class TestLocalCinematography:
    """Verify local VLM tier for cinematography analysis."""

    def test_cinematography_tier_default_is_cloud(self):
        settings = Settings()
        assert settings.cinematography_tier == "cloud"

    def test_cinematography_local_model_default(self):
        settings = Settings()
        assert "Qwen2.5-VL-7B" in settings.cinematography_local_model

    def test_cinematography_tier_setting_roundtrip(self, tmp_path):
        s = Settings()
        s.cinematography_tier = "local"
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(_settings_to_json(s)))

        loaded = Settings()
        _load_from_json(config_path, loaded)
        assert loaded.cinematography_tier == "local"

    def test_cinematography_local_model_setting_roundtrip(self, tmp_path):
        s = Settings()
        s.cinematography_local_model = "custom-vlm-model"
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(_settings_to_json(s)))

        loaded = Settings()
        _load_from_json(config_path, loaded)
        assert loaded.cinematography_local_model == "custom-vlm-model"

    def test_analyze_cinematography_local_function_exists(self):
        from core.analysis.cinematography import analyze_cinematography_local
        assert callable(analyze_cinematography_local)

    def test_analyze_cinematography_dispatches_on_tier(self):
        """analyze_cinematography should check tier setting."""
        import inspect
        import core.analysis.cinematography as cine
        source = inspect.getsource(cine.analyze_cinematography)
        assert "cinematography_tier" in source or 'tier == "local"' in source

    def test_cost_estimate_cinematography_has_local(self):
        from core.cost_estimates import TIME_PER_CLIP
        assert "local" in TIME_PER_CLIP["cinematography"]

    def test_cost_estimate_cinematography_local_time(self):
        from core.cost_estimates import TIME_PER_CLIP
        assert TIME_PER_CLIP["cinematography"]["local"] <= 5.0

    def test_tiered_operations_includes_cinematography(self):
        from core.cost_estimates import TIERED_OPERATIONS
        assert "cinematography" in TIERED_OPERATIONS
        assert "local" in TIERED_OPERATIONS["cinematography"]


# --- P3.2: YOLOE-26 open-vocabulary detection ---

class TestYOLOE26OpenVocab:
    """Verify YOLOE-26 open-vocabulary detection mode."""

    def test_yoloe_model_name(self):
        from core.analysis.detection import _YOLOE_MODEL_NAME
        assert _YOLOE_MODEL_NAME == "yoloe-26s.pt"

    def test_detect_objects_open_vocab_function_exists(self):
        from core.analysis.detection import detect_objects_open_vocab
        assert callable(detect_objects_open_vocab)

    def test_detection_mode_default_is_fixed(self):
        settings = Settings()
        assert settings.detection_mode == "fixed"

    def test_detection_custom_classes_default_empty(self):
        settings = Settings()
        assert settings.detection_custom_classes == []

    def test_detection_mode_setting_roundtrip(self, tmp_path):
        s = Settings()
        s.detection_mode = "open_vocab"
        s.detection_custom_classes = ["camera", "microphone", "tripod"]
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(_settings_to_json(s)))

        loaded = Settings()
        _load_from_json(config_path, loaded)
        assert loaded.detection_mode == "open_vocab"
        assert loaded.detection_custom_classes == ["camera", "microphone", "tripod"]

    def test_unload_model_clears_ov_model(self):
        """unload_model should clear both fixed and open-vocab models."""
        import inspect
        import core.analysis.detection as det
        source = inspect.getsource(det.unload_model)
        assert "_ov_model" in source

    def test_ov_model_globals_exist(self):
        import core.analysis.detection as det
        assert hasattr(det, "_ov_model")
        assert hasattr(det, "_ov_model_classes")
        assert hasattr(det, "_ov_model_lock")

    def test_no_clip_model_in_detection(self):
        """detection.py should not use openai/clip model."""
        import inspect
        import core.analysis.detection as det
        source = inspect.getsource(det)
        assert "openai/clip" not in source


# --- P3.3: Groq cloud transcription ---

class TestGroqCloudTranscription:
    """Verify Groq cloud transcription backend."""

    def test_groq_backend_in_resolve(self):
        """_resolve_backend should accept 'groq' as valid backend."""
        from core.transcription import _resolve_backend
        assert _resolve_backend("groq") == "groq"

    def test_groq_models_constant_exists(self):
        from core.transcription import GROQ_MODELS
        assert "whisper-large-v3-turbo" in GROQ_MODELS

    def test_default_groq_model(self):
        from core.transcription import _DEFAULT_GROQ_MODEL
        assert _DEFAULT_GROQ_MODEL == "whisper-large-v3-turbo"

    def test_transcription_cloud_model_default(self):
        settings = Settings()
        assert settings.transcription_cloud_model == "whisper-large-v3-turbo"

    def test_transcription_backend_accepts_groq(self):
        settings = Settings()
        settings.transcription_backend = "groq"
        assert settings.transcription_backend == "groq"

    def test_transcription_cloud_model_roundtrip(self, tmp_path):
        s = Settings()
        s.transcription_cloud_model = "whisper-large-v3"
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(_settings_to_json(s)))

        loaded = Settings()
        _load_from_json(config_path, loaded)
        assert loaded.transcription_cloud_model == "whisper-large-v3"

    def test_cost_estimate_transcribe_has_cloud(self):
        from core.cost_estimates import TIME_PER_CLIP
        assert "cloud" in TIME_PER_CLIP["transcribe"]

    def test_cost_estimate_transcribe_cloud_cost(self):
        from core.cost_estimates import COST_PER_CLIP
        assert "cloud" in COST_PER_CLIP["transcribe"]
        assert COST_PER_CLIP["transcribe"]["cloud"] < 0.001  # Very cheap per clip

    def test_tiered_operations_includes_transcribe(self):
        from core.cost_estimates import TIERED_OPERATIONS
        assert "transcribe" in TIERED_OPERATIONS
        assert "cloud" in TIERED_OPERATIONS["transcribe"]

    def test_transcribe_cloud_function_exists(self):
        from core.transcription import _transcribe_cloud_groq
        assert callable(_transcribe_cloud_groq)

    def test_docstring_mentions_groq(self):
        """Module docstring should mention Groq."""
        import core.transcription as trans
        assert "groq" in trans.__doc__.lower()

    def test_auto_backend_does_not_select_groq(self):
        """'auto' backend should not select groq (must be explicit)."""
        from core.transcription import _resolve_backend
        resolved = _resolve_backend("auto")
        assert resolved != "groq"

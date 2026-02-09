"""Tests pinning P0 model upgrade defaults.

Ensures model names, versions, and defaults aren't accidentally changed.
"""

from core.settings import Settings
from core.transcription import WHISPER_MODELS


class TestYOLO26ModelName:
    """Verify YOLO26 model name is used in detection module."""

    def test_yolo26_model_name_in_load_function(self):
        """The detection module should reference yolo26, not yolov8."""
        import core.analysis.detection as det
        import inspect
        source = inspect.getsource(det._load_yolo)
        assert "yolo26" in source
        assert "yolov8" not in source


class TestWhisperTurboModel:
    """Verify large-v3-turbo is available as a Whisper model option."""

    def test_large_v3_turbo_in_models_dict(self):
        assert "large-v3-turbo" in WHISPER_MODELS

    def test_large_v3_turbo_metadata(self):
        turbo = WHISPER_MODELS["large-v3-turbo"]
        assert "size" in turbo
        assert "speed" in turbo
        assert "accuracy" in turbo
        assert "vram" in turbo
        assert turbo["accuracy"] == "Best"

    def test_existing_models_preserved(self):
        """All pre-existing model entries must still exist."""
        for model in ["tiny.en", "small.en", "medium.en", "large-v3"]:
            assert model in WHISPER_MODELS, f"{model} missing from WHISPER_MODELS"


class TestGemini3FlashDefaults:
    """Verify cloud VLM defaults updated to Gemini 3 Flash."""

    def test_description_model_cloud_default(self):
        settings = Settings()
        assert settings.description_model_cloud == "gemini-3-flash-preview"

    def test_text_extraction_vlm_model_default(self):
        settings = Settings()
        assert settings.text_extraction_vlm_model == "gemini-3-flash-preview"

    def test_exquisite_corpus_model_default(self):
        settings = Settings()
        assert settings.exquisite_corpus_model == "gemini-3-flash-preview"

    def test_cinematography_model_default(self):
        settings = Settings()
        assert settings.cinematography_model == "gemini-3-flash-preview"

    def test_chat_gemini_model_unchanged(self):
        """The chat LLM Gemini default should NOT change (separate concern)."""
        settings = Settings()
        assert settings.gemini_model == "gemini-2.5-flash"

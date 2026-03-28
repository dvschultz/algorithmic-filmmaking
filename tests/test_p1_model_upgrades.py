"""Tests pinning P1 high-impact model upgrades.

Covers:
- P1.1: SigLIP 2 for shot classification (decoupled from CLIP embeddings)
- P1.2: Gemini Flash Lite for cloud shot classification
- P1.3: mlx-whisper transcription backend
"""

import platform
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import pytest

from core.settings import Settings


# --- P1.1a: Embeddings decoupled from shots ---

class TestEmbeddingsDecoupled:
    """Verify embeddings.py has its own model, separate from shots.py."""

    def test_embeddings_has_dinov2_model_name(self):
        from core.analysis.embeddings import _DINOV2_MODEL_NAME
        assert _DINOV2_MODEL_NAME == "facebook/dinov2-base"

    def test_embeddings_produces_768_dim(self):
        from core.analysis.embeddings import _EMBEDDING_DIM
        assert _EMBEDDING_DIM == 768

    def test_embeddings_does_not_import_from_shots(self):
        """embeddings.py must not import model loading from shots.py."""
        import inspect
        import core.analysis.embeddings as emb
        source = inspect.getsource(emb)
        assert "from core.analysis.shots import load_clip_model" not in source
        assert "from core.analysis.shots import _model" not in source

    def test_embeddings_has_model_lifecycle_functions(self):
        from core.analysis import embeddings
        assert hasattr(embeddings, "is_model_loaded")
        assert hasattr(embeddings, "unload_model")
        assert callable(embeddings.is_model_loaded)
        assert callable(embeddings.unload_model)


# --- P1.1b: SigLIP 2 for shot classification ---

class TestSigLIP2Classification:
    """Verify shots.py uses SigLIP 2 instead of CLIP."""

    def test_siglip_model_name(self):
        from core.analysis.shots import _SIGLIP_MODEL_NAME
        assert _SIGLIP_MODEL_NAME == "google/siglip2-base-patch16-224"

    def test_no_clip_model_reference(self):
        """shots.py should not reference openai/clip as its classification model."""
        import inspect
        import core.analysis.shots as shots_mod
        source = inspect.getsource(shots_mod)
        assert "openai/clip" not in source

    def test_shot_types_preserved(self):
        from core.analysis.shots import SHOT_TYPES
        expected = ["wide shot", "full shot", "medium shot", "close-up", "extreme close-up"]
        assert SHOT_TYPES == expected

    def test_shot_type_prompts_exist(self):
        from core.analysis.shots import SHOT_TYPE_PROMPTS, SHOT_TYPES
        for shot_type in SHOT_TYPES:
            assert shot_type in SHOT_TYPE_PROMPTS
            prompts = SHOT_TYPE_PROMPTS[shot_type]
            assert len(prompts) >= 2, f"{shot_type} should have multiple prompts"

    def test_backward_compatible_alias(self):
        """load_clip_model alias must still exist for backward compatibility."""
        from core.analysis.shots import load_clip_model, load_classification_model
        assert load_clip_model is load_classification_model

    def test_shots_has_model_lifecycle_functions(self):
        from core.analysis import shots
        assert hasattr(shots, "is_model_loaded")
        assert hasattr(shots, "unload_model")

    def test_classify_shot_type_raises_on_runtime_failure(self, tmp_path, monkeypatch):
        from core.analysis.shots import (
            ShotClassificationError,
            classify_shot_type,
        )

        image_path = tmp_path / "thumb.jpg"
        image_path.write_bytes(b"not-a-real-image")

        # This test is about runtime failures from model loading, not whether
        # optional torch is installed on the current CI image.
        monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
        monkeypatch.setattr(
            "core.analysis.shots.load_classification_model",
            lambda: (_ for _ in ()).throw(RuntimeError("transformers missing")),
        )

        with pytest.raises(ShotClassificationError, match="transformers missing"):
            classify_shot_type(image_path)

    def test_load_classification_model_uses_runtime_import_helper(self, monkeypatch):
        import core.analysis.shots as shots

        shots.unload_model()

        class _FakeProcessor:
            @staticmethod
            def from_pretrained(_name, **kwargs):
                return "processor"

        class _FakeModel:
            @staticmethod
            def from_pretrained(_name, **kwargs):
                return "model"

        monkeypatch.setattr(
            shots,
            "ensure_classification_runtime_available",
            lambda: (_FakeProcessor, _FakeModel),
        )

        model, processor = shots.load_classification_model()
        assert model == "model"
        assert processor == "processor"

        shots.unload_model()

    def test_load_classification_model_blocks_implicit_large_downloads_on_frozen_macos(
        self, monkeypatch
    ):
        import core.analysis.shots as shots

        shots.unload_model()

        monkeypatch.setattr(
            shots,
            "ensure_classification_runtime_available",
            lambda: (_ for _ in ()).throw(AssertionError("should not reach runtime import")),
        )
        monkeypatch.setattr(
            shots,
            "get_missing_large_models",
            lambda _model_cache_dir: [{"id": "siglip2_shot_classifier"}],
        )
        monkeypatch.setattr(shots, "large_model_downloads_allowed", lambda: False)

        class _Settings:
            model_cache_dir = Path("/tmp/models")

        monkeypatch.setattr("core.settings.load_settings", lambda: _Settings())

        with pytest.raises(RuntimeError, match="SigLIP 2 shot classifier is not downloaded"):
            shots.load_classification_model()

        shots.unload_model()


# --- P1.2: Gemini Flash Lite cloud shot classification ---

class TestGeminiCloudShots:
    """Verify cloud shot classification uses Gemini Flash Lite."""

    def test_default_cloud_model_constant(self):
        from core.analysis.shots_cloud import _DEFAULT_CLOUD_MODEL
        assert _DEFAULT_CLOUD_MODEL == "gemini-2.5-flash-lite"

    def test_classify_shot_cloud_function_exists(self):
        from core.analysis.shots_cloud import classify_shot_cloud
        assert callable(classify_shot_cloud)

    def test_shot_classifier_cloud_model_setting(self):
        settings = Settings()
        assert settings.shot_classifier_cloud_model == "gemini-2.5-flash-lite"

    def test_cloud_shots_cost_updated(self):
        """Cloud shot classification cost should reflect Gemini pricing (~$0.00026)."""
        from core.cost_estimates import COST_PER_CLIP
        cost = COST_PER_CLIP["shots"]["cloud"]
        assert cost < 0.001, f"Cloud shot cost should be <$0.001, got {cost}"
        assert cost == 0.00026

    def test_tiered_routing_imports_cloud(self):
        """classify_shot_type_tiered should reference classify_shot_cloud."""
        import inspect
        from core.analysis.shots import classify_shot_type_tiered
        source = inspect.getsource(classify_shot_type_tiered)
        assert "classify_shot_cloud" in source

    def test_legacy_replicate_code_preserved(self):
        """Replicate functions should still exist for backward compatibility."""
        from core.analysis.shots_cloud import (
            classify_shot_replicate,
            classify_shot_from_thumbnail,
        )
        assert callable(classify_shot_replicate)
        assert callable(classify_shot_from_thumbnail)


# --- P1.3: mlx-whisper transcription backend ---

class TestMlxWhisperBackend:
    """Verify mlx-whisper backend integration."""

    def test_mlx_model_map_covers_standard_models(self):
        from core.transcription import _MLX_MODEL_MAP, WHISPER_MODELS
        for model_name in WHISPER_MODELS:
            assert model_name in _MLX_MODEL_MAP, (
                f"{model_name} missing from _MLX_MODEL_MAP"
            )

    def test_mlx_model_map_uses_valid_lightning_whisper_names(self):
        from core.transcription import _MLX_MODEL_MAP

        assert _MLX_MODEL_MAP["tiny.en"] == "tiny"
        assert _MLX_MODEL_MAP["small.en"] == "small"
        assert _MLX_MODEL_MAP["medium.en"] == "medium"
        assert _MLX_MODEL_MAP["large-v3"] == "large-v3"

    def test_is_mlx_whisper_available_function_exists(self):
        from core.transcription import is_mlx_whisper_available
        assert callable(is_mlx_whisper_available)

    def test_resolve_backend_auto(self):
        from core.transcription import _resolve_backend
        # "auto" should return one of the valid backends
        result = _resolve_backend("auto")
        assert result in ("faster-whisper", "mlx-whisper")

    def test_resolve_backend_explicit_faster_whisper(self):
        from core.transcription import _resolve_backend
        assert _resolve_backend("faster-whisper") == "faster-whisper"

    def test_resolve_backend_mlx_fallback(self):
        """If mlx-whisper not available, requesting it should fall back."""
        from core.transcription import _resolve_backend
        with patch("core.transcription.is_mlx_whisper_available", return_value=False):
            assert _resolve_backend("mlx-whisper") == "faster-whisper"

    def test_transcription_backend_setting(self):
        settings = Settings()
        assert settings.transcription_backend == "auto"

    def test_transcribe_functions_accept_backend_param(self):
        """Both transcribe functions should accept a backend parameter."""
        import inspect
        from core.transcription import transcribe_video, transcribe_clip
        video_sig = inspect.signature(transcribe_video)
        clip_sig = inspect.signature(transcribe_clip)
        assert "backend" in video_sig.parameters
        assert "backend" in clip_sig.parameters

    def test_transcription_time_estimate_updated(self):
        """Transcription time should reflect mlx-whisper speeds."""
        from core.cost_estimates import TIME_PER_CLIP
        time_local = TIME_PER_CLIP["transcribe"]["local"]
        assert time_local <= 0.5, (
            f"Transcribe time should be <=0.5s (mlx-whisper), got {time_local}"
        )

    def test_transcription_worker_accepts_backend(self):
        """TranscriptionWorker should accept a backend parameter."""
        import inspect
        from ui.workers.transcription_worker import TranscriptionWorker
        sig = inspect.signature(TranscriptionWorker.__init__)
        assert "backend" in sig.parameters


# --- Settings persistence ---

class TestP1SettingsPersistence:
    """Verify new P1 settings are saved and loaded correctly."""

    def test_shot_classifier_cloud_model_roundtrip(self, tmp_path):
        import json
        from core.settings import Settings, _settings_to_json, _load_from_json
        s = Settings()
        s.shot_classifier_cloud_model = "gemini-custom-model"
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(_settings_to_json(s)))

        loaded = Settings()
        _load_from_json(config_path, loaded)
        assert loaded.shot_classifier_cloud_model == "gemini-custom-model"

    def test_transcription_backend_roundtrip(self, tmp_path):
        import json
        from core.settings import Settings, _settings_to_json, _load_from_json
        s = Settings()
        s.transcription_backend = "mlx-whisper"
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(_settings_to_json(s)))

        loaded = Settings()
        _load_from_json(config_path, loaded)
        assert loaded.transcription_backend == "mlx-whisper"

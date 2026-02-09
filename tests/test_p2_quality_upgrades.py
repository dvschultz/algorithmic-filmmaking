"""Tests pinning P2 quality upgrades.

Covers:
- P2.0: Embedding architecture migration (dimension-aware validation)
- P2.1: DINOv2 for visual similarity embeddings
- P2.2: Qwen3-VL 4B for local VLM descriptions
- P2.3: PaddleOCR PP-OCRv5 for text extraction
"""

import json
import platform

from core.settings import Settings, _settings_to_json, _load_from_json


# --- P2.0: Embedding architecture migration ---

class TestEmbeddingArchitecture:
    """Verify embedding dimension-aware validation and model tagging."""

    def test_valid_embedding_dims_include_512_and_768(self):
        from models.clip import VALID_EMBEDDING_DIMS
        assert 512 in VALID_EMBEDDING_DIMS
        assert 768 in VALID_EMBEDDING_DIMS

    def test_clip_has_embedding_model_field(self):
        from models.clip import Clip
        c = Clip(
            id="test",
            source_id="s1",
            start_frame=0,
            end_frame=100,
        )
        assert hasattr(c, "embedding_model")
        assert c.embedding_model is None

    def test_embedding_model_roundtrip(self):
        from models.clip import Clip
        c = Clip(
            id="test",
            source_id="s1",
            start_frame=0,
            end_frame=100,
            embedding_model="dinov2-vit-b-14",
        )
        data = c.to_dict()
        assert data["embedding_model"] == "dinov2-vit-b-14"

        restored = Clip.from_dict(data)
        assert restored.embedding_model == "dinov2-vit-b-14"

    def test_512_dim_embedding_still_valid(self):
        """Old 512-dim CLIP embeddings should still pass validation."""
        from models.clip import Clip
        c = Clip(
            id="test",
            source_id="s1",
            start_frame=0,
            end_frame=100,
            embedding=[0.0] * 512,
        )
        assert len(c.embedding) == 512

    def test_768_dim_embedding_valid(self):
        """New 768-dim DINOv2 embeddings should pass validation."""
        from models.clip import Clip
        c = Clip(
            id="test",
            source_id="s1",
            start_frame=0,
            end_frame=100,
            embedding=[0.0] * 768,
        )
        assert len(c.embedding) == 768

    def test_invalid_dim_embedding_discarded_on_load(self):
        """Embeddings with invalid dimensions should be discarded on from_dict."""
        from models.clip import Clip
        data = {
            "id": "test",
            "source_id": "s1",
            "start_frame": 0,
            "end_frame": 100,
            "embedding": [0.0] * 256,
        }
        c = Clip.from_dict(data)
        assert c.embedding is None

    def test_schema_version_bumped(self):
        from core.project import SCHEMA_VERSION
        assert SCHEMA_VERSION >= "1.2"


# --- P2.1: DINOv2 for visual similarity embeddings ---

class TestDINOv2Embeddings:
    """Verify DINOv2 replaces CLIP for embeddings."""

    def test_dinov2_model_name(self):
        from core.analysis.embeddings import _DINOV2_MODEL_NAME
        assert _DINOV2_MODEL_NAME == "facebook/dinov2-base"

    def test_embedding_dim_768(self):
        from core.analysis.embeddings import _EMBEDDING_DIM
        assert _EMBEDDING_DIM == 768

    def test_model_tag(self):
        from core.analysis.embeddings import _EMBEDDING_MODEL_TAG
        assert _EMBEDDING_MODEL_TAG == "dinov2-vit-b-14"

    def test_no_clip_import(self):
        """embeddings.py should not use openai/clip."""
        import inspect
        import core.analysis.embeddings as emb
        source = inspect.getsource(emb)
        assert "openai/clip" not in source

    def test_zero_embedding_is_768(self):
        """Zero embeddings should be 768-dim for DINOv2."""
        from core.analysis.embeddings import _EMBEDDING_DIM
        zero = [0.0] * _EMBEDDING_DIM
        assert len(zero) == 768

    def test_embedding_model_tag_set_in_remix(self):
        """remix/__init__.py should tag clips with embedding model."""
        import inspect
        import core.remix as remix
        source = inspect.getsource(remix)
        assert "_EMBEDDING_MODEL_TAG" in source


# --- P2.2: Qwen3-VL 4B for local VLM descriptions ---

class TestQwen3VLDescriptions:
    """Verify Qwen3-VL replaces Moondream for local descriptions."""

    def test_local_vlm_name(self):
        from core.analysis.description import _LOCAL_VLM_NAME
        assert _LOCAL_VLM_NAME == "mlx-community/Qwen3-VL-4B-4bit"

    def test_moondream_fallback_preserved(self):
        from core.analysis.description import _LOCAL_VLM_FALLBACK
        assert "moondream" in _LOCAL_VLM_FALLBACK.lower()

    def test_is_mlx_vlm_available_function_exists(self):
        from core.analysis.description import is_mlx_vlm_available
        assert callable(is_mlx_vlm_available)

    def test_describe_frame_local_function_exists(self):
        from core.analysis.description import describe_frame_local
        assert callable(describe_frame_local)

    def test_describe_frame_cpu_backward_compat(self):
        """describe_frame_cpu should still exist as an alias."""
        from core.analysis.description import describe_frame_cpu
        assert callable(describe_frame_cpu)

    def test_description_tier_default_is_local(self):
        settings = Settings()
        assert settings.description_model_tier == "local"

    def test_description_model_local_default(self):
        settings = Settings()
        assert settings.description_model_local == "mlx-community/Qwen3-VL-4B-4bit"

    def test_tier_migration_cpu_to_local(self, tmp_path):
        """Legacy 'cpu' tier should be migrated to 'local' on load."""
        config = {
            "description": {
                "model_tier": "cpu",
                "model_cpu": "vikhyatk/moondream2",
            }
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loaded = Settings()
        _load_from_json(config_path, loaded)
        assert loaded.description_model_tier == "local"

    def test_tier_migration_gpu_to_local(self, tmp_path):
        """Legacy 'gpu' tier should be migrated to 'local' on load."""
        config = {
            "description": {
                "model_tier": "gpu",
            }
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loaded = Settings()
        _load_from_json(config_path, loaded)
        assert loaded.description_model_tier == "local"

    def test_describe_time_estimate_updated(self):
        """Local describe time should reflect Qwen3-VL speeds."""
        from core.cost_estimates import TIME_PER_CLIP
        time_local = TIME_PER_CLIP["describe"]["local"]
        assert time_local <= 2.0, (
            f"Describe time should be <=2.0s (Qwen3-VL), got {time_local}"
        )

    def test_video_capable_model_includes_qwen(self):
        from core.analysis.description import is_video_capable_model
        assert is_video_capable_model("qwen3-vl-4b")

    def test_model_local_setting_roundtrip(self, tmp_path):
        s = Settings()
        s.description_model_local = "custom-local-model"
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(_settings_to_json(s)))

        loaded = Settings()
        _load_from_json(config_path, loaded)
        assert loaded.description_model_local == "custom-local-model"


# --- P2.3: PaddleOCR PP-OCRv5 for text extraction ---

class TestPaddleOCR:
    """Verify PaddleOCR replaces Tesseract for text extraction."""

    def test_paddleocr_availability_function_exists(self):
        from core.analysis.ocr import is_paddleocr_available
        assert callable(is_paddleocr_available)

    def test_tesseract_alias_exists(self):
        """is_tesseract_available should still exist as a legacy alias."""
        from core.analysis.ocr import is_tesseract_available
        assert callable(is_tesseract_available)

    def test_extract_text_from_frame_function_exists(self):
        from core.analysis.ocr import extract_text_from_frame
        assert callable(extract_text_from_frame)

    def test_extract_text_from_frame_returns_paddleocr_source(self):
        """Return tuple should use 'paddleocr' as source tag."""
        import inspect
        import core.analysis.ocr as ocr
        source_code = inspect.getsource(ocr)
        assert '"paddleocr"' in source_code

    def test_no_pytesseract_import(self):
        """ocr.py should not import pytesseract."""
        import inspect
        import core.analysis.ocr as ocr
        source_code = inspect.getsource(ocr)
        assert "import pytesseract" not in source_code

    def test_settings_method_default(self):
        settings = Settings()
        assert settings.text_extraction_method == "hybrid"

    def test_settings_method_migration_tesseract_to_paddleocr(self, tmp_path):
        """Legacy 'tesseract' method should be migrated to 'paddleocr'."""
        config = {
            "text_extraction": {
                "method": "tesseract",
            }
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loaded = Settings()
        _load_from_json(config_path, loaded)
        assert loaded.text_extraction_method == "paddleocr"

    def test_settings_method_hybrid_preserved(self, tmp_path):
        """'hybrid' method should not be migrated."""
        config = {
            "text_extraction": {
                "method": "hybrid",
            }
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loaded = Settings()
        _load_from_json(config_path, loaded)
        assert loaded.text_extraction_method == "hybrid"

    def test_text_detection_module_deprecated(self):
        """text_detection.py should be marked deprecated."""
        import inspect
        import core.analysis.text_detection as td
        docstring = inspect.getmodule(td).__doc__
        assert "deprecated" in docstring.lower()

    def test_extracted_text_source_documents_paddleocr(self):
        """ExtractedText source field should document paddleocr."""
        from models.clip import ExtractedText
        et = ExtractedText(
            frame_number=0,
            text="hello",
            confidence=0.9,
            source="paddleocr",
        )
        assert et.source == "paddleocr"

    def test_cost_estimate_extract_text_has_local(self):
        from core.cost_estimates import TIME_PER_CLIP
        assert "local" in TIME_PER_CLIP["extract_text"]

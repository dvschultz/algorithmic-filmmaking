"""Tests for model download error handling across analysis modules."""

from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from core.errors import ModelDownloadError


class TestEmbeddingsModelDownloadErrors:
    """Tests for DINOv2 embedding model error handling."""

    def setup_method(self):
        """Reset module-level singleton before each test."""
        import core.analysis.embeddings as mod

        mod._model = None
        mod._processor = None

    def teardown_method(self):
        """Clean up module-level singleton after each test."""
        import core.analysis.embeddings as mod

        mod._model = None
        mod._processor = None

    def test_network_failure_raises_model_download_error(self):
        """Network errors during from_pretrained should raise ModelDownloadError."""
        mock_proc_cls = MagicMock()
        mock_proc_cls.from_pretrained.side_effect = OSError("Connection timed out")
        mock_model_cls = MagicMock()

        mock_transformers = MagicMock()
        mock_transformers.AutoImageProcessor = mock_proc_cls
        mock_transformers.AutoModel = mock_model_cls

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            import core.analysis.embeddings as mod

            mod._model = None
            mod._processor = None

            with pytest.raises(ModelDownloadError, match="DINOv2"):
                mod._get_model()

    def test_additional_chat_templates_404_handled(self):
        """The additional_chat_templates 404 bug should trigger snapshot_download fallback."""
        mock_proc_cls = MagicMock()
        # First call (normal) raises the 404, second call (local_files_only) succeeds
        mock_proc_cls.from_pretrained.side_effect = [
            Exception("404 additional_chat_templates not found"),
            MagicMock(),  # success on local retry
        ]
        mock_model_cls = MagicMock()
        # AutoModel.from_pretrained is never called in the first try (processor fails first),
        # only called once in the except block with local_files_only

        mock_snapshot = MagicMock(return_value="/tmp/fake_model_dir")

        mock_transformers = MagicMock()
        mock_transformers.AutoImageProcessor = mock_proc_cls
        mock_transformers.AutoModel = mock_model_cls

        mock_hf_hub = MagicMock()
        mock_hf_hub.snapshot_download = mock_snapshot

        with patch.dict(
            "sys.modules",
            {
                "transformers": mock_transformers,
                "huggingface_hub": mock_hf_hub,
            },
        ):
            import core.analysis.embeddings as mod

            mod._model = None
            mod._processor = None

            mod._get_model()

            # snapshot_download should have been called
            mock_snapshot.assert_called_once()
            # from_pretrained called twice: first attempt + local retry
            assert mock_proc_cls.from_pretrained.call_count == 2
            second_call = mock_proc_cls.from_pretrained.call_args_list[1]
            assert second_call.kwargs.get("local_files_only") is True


class TestDetectionModelDownloadErrors:
    """Tests for YOLO model download error handling."""

    def setup_method(self):
        import core.analysis.detection as mod

        mod._model = None
        mod._ov_model = None

    def teardown_method(self):
        import core.analysis.detection as mod

        mod._model = None
        mod._ov_model = None

    @patch("core.analysis.detection.ensure_object_detection_runtime_available")
    @patch("core.analysis.detection._get_model_cache_dir")
    def test_yolo_network_failure_raises_model_download_error(
        self, mock_cache, mock_ensure
    ):
        mock_cache.return_value = Path("/tmp/fake_cache")
        mock_yolo_cls = MagicMock(side_effect=OSError("Connection refused"))
        mock_ensure.return_value = mock_yolo_cls

        from core.analysis.detection import _load_yolo

        with pytest.raises(ModelDownloadError, match="YOLO26n"):
            _load_yolo("n")

    @patch("core.analysis.detection.ensure_object_detection_runtime_available")
    @patch("core.analysis.detection._get_model_cache_dir")
    def test_yoloe_network_failure_raises_model_download_error(
        self, mock_cache, mock_ensure
    ):
        mock_cache.return_value = Path("/tmp/fake_cache")
        mock_yolo_cls = MagicMock(side_effect=OSError("Connection refused"))
        mock_ensure.return_value = mock_yolo_cls

        from core.analysis.detection import _load_yoloe

        with pytest.raises(ModelDownloadError, match="YOLOE"):
            _load_yoloe(["person", "car"])


class TestOCRModelDownloadErrors:
    """Tests for PaddleOCR model download error handling."""

    def setup_method(self):
        import core.analysis.ocr as mod

        mod._ocr_engine = None

    def teardown_method(self):
        import core.analysis.ocr as mod

        mod._ocr_engine = None

    @patch("core.analysis.ocr.ensure_ocr_runtime_available")
    def test_paddleocr_init_failure_raises_model_download_error(self, mock_ensure):
        mock_paddle_cls = MagicMock(side_effect=RuntimeError("Model download failed"))
        mock_ensure.return_value = mock_paddle_cls

        from core.analysis.ocr import _get_ocr_engine

        with pytest.raises(ModelDownloadError, match="PaddleOCR"):
            _get_ocr_engine()


class TestStemSeparationModelDownloadErrors:
    """Tests for Demucs model download error handling."""

    def test_get_model_failure_raises_model_download_error(self):
        """get_model("htdemucs") failure should raise ModelDownloadError.

        Tests the error wrapping in separate_stems() by mocking at the
        demucs_infer.pretrained level without reloading the module.
        """
        from core.analysis import stem_separation as mod

        mock_get_model = MagicMock(side_effect=OSError("Connection refused"))
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        # Patch the imports that happen inside separate_stems()
        with patch.dict(
            "sys.modules",
            {
                "demucs_infer.pretrained": MagicMock(get_model=mock_get_model),
                "demucs_infer.apply": MagicMock(),
                "demucs_infer.audio": MagicMock(),
            },
        ), patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            with pytest.raises(ModelDownloadError, match="Demucs"):
                mod.separate_stems(
                    music_path=Path("/tmp/fake.mp3"),
                    output_dir=Path("/tmp/fake_out"),
                )


class TestEASTModelDownloadTimeout:
    """Tests for EAST model download timeout."""

    def test_download_uses_timeout(self, tmp_path):
        """urlopen should be called with a timeout parameter."""
        import urllib.request

        mock_response = MagicMock()
        mock_response.read.side_effect = [b"fake model data", b""]

        dest = tmp_path / "model.pb"

        with patch.object(urllib.request, "urlopen", return_value=mock_response) as mock_urlopen:
            from core.analysis.text_detection import _download_east_model

            _download_east_model(dest)

            mock_urlopen.assert_called_once()
            call_args = mock_urlopen.call_args
            # Check timeout=60 in the call
            assert call_args.kwargs.get("timeout") == 60 or (
                len(call_args.args) >= 2 and call_args.args[1] == 60
            )


class TestAutoComputeEmbeddingsInstallGate:
    """Tests for install gate in auto-compute functions."""

    @patch("core.feature_registry.check_feature")
    def test_auto_compute_raises_when_deps_missing(self, mock_check):
        mock_check.return_value = (False, ["package:torch", "package:transformers"])

        from core.remix import _auto_compute_embeddings

        # Need a clip with embedding=None to trigger the check
        mock_clip = MagicMock()
        mock_clip.embedding = None
        mock_clip.thumbnail_path = "/tmp/fake.jpg"
        mock_source = MagicMock()

        with pytest.raises(RuntimeError, match="torch and transformers"):
            _auto_compute_embeddings([(mock_clip, mock_source)])

    @patch("core.feature_registry.check_feature")
    def test_auto_compute_boundary_raises_when_deps_missing(self, mock_check):
        mock_check.return_value = (False, ["package:torch", "package:transformers"])

        from core.remix import _auto_compute_boundary_embeddings

        # Need a clip with missing boundary embeddings to trigger the check
        mock_clip = MagicMock()
        mock_clip.first_frame_embedding = None
        mock_clip.last_frame_embedding = None
        mock_source = MagicMock()

        with pytest.raises(RuntimeError, match="torch and transformers"):
            _auto_compute_boundary_embeddings([(mock_clip, mock_source)])

    def test_auto_compute_skips_when_no_clips_need_embedding(self):
        """When no clips need embedding, should return without checking deps."""
        from core.remix import _auto_compute_embeddings

        # Empty clip list — should just return without error or dep check
        _auto_compute_embeddings([])

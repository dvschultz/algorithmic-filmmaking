"""Unit tests for core/analysis/text_detection module."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np


class TestTextDetectionModule:
    """Tests for the text detection module."""

    def test_import_module(self):
        """Test that the module can be imported."""
        from core.analysis import text_detection
        assert hasattr(text_detection, 'has_text_regions')
        assert hasattr(text_detection, 'detect_text_regions')
        assert hasattr(text_detection, 'is_text_detection_available')

    def test_default_thresholds(self):
        """Test default threshold values."""
        from core.analysis.text_detection import (
            DEFAULT_CONFIDENCE_THRESHOLD,
            DEFAULT_NMS_THRESHOLD,
        )
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.5
        assert DEFAULT_NMS_THRESHOLD == 0.4

    def test_model_filename(self):
        """Test model filename constant."""
        from core.analysis.text_detection import EAST_MODEL_FILENAME
        assert EAST_MODEL_FILENAME == "frozen_east_text_detection.pb"


class TestHasTextRegions:
    """Tests for has_text_regions function."""

    def test_returns_true_when_model_unavailable(self):
        """Test fail-safe: return True when model not available."""
        from core.analysis.text_detection import has_text_regions

        # Mock the model loading to return None (model not available)
        with patch('core.analysis.text_detection._load_east_model', return_value=None):
            # Should return True (fail-safe)
            result = has_text_regions(Path('/fake/image.jpg'))
            assert result is True

    def test_returns_true_when_image_unreadable(self):
        """Test fail-safe: return True when image cannot be read."""
        from core.analysis.text_detection import has_text_regions

        # Create a mock network
        mock_net = MagicMock()

        with patch('core.analysis.text_detection._load_east_model', return_value=mock_net):
            with patch('cv2.imread', return_value=None):
                result = has_text_regions(Path('/fake/nonexistent.jpg'))
                assert result is True

    def test_returns_true_on_exception(self):
        """Test fail-safe: return True on any exception."""
        from core.analysis.text_detection import has_text_regions

        # Create a mock network that raises on forward
        mock_net = MagicMock()
        mock_net.forward.side_effect = Exception("Test error")

        # Mock image reading to succeed
        fake_image = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch('core.analysis.text_detection._load_east_model', return_value=mock_net):
            with patch('cv2.imread', return_value=fake_image):
                result = has_text_regions(Path('/fake/image.jpg'))
                # Should return True due to fail-safe on exception
                assert result is True


class TestIsTextDetectionAvailable:
    """Tests for is_text_detection_available function."""

    def test_returns_true_when_model_exists(self):
        """Test returns True when model file exists."""
        from core.analysis.text_detection import is_text_detection_available

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "frozen_east_text_detection.pb"
            model_path.touch()  # Create empty file

            with patch('core.analysis.text_detection._get_model_path', return_value=model_path):
                assert is_text_detection_available() is True

    def test_returns_true_when_dir_writable(self):
        """Test returns True when model dir is writable (can download)."""
        from core.analysis.text_detection import is_text_detection_available

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "models" / "frozen_east_text_detection.pb"

            with patch('core.analysis.text_detection._get_model_path', return_value=model_path):
                # Model doesn't exist but parent can be created
                assert is_text_detection_available() is True


class TestIntegrationWithOCR:
    """Tests for integration with the OCR module."""

    def test_ocr_function_has_skip_detection_param(self):
        """Test that extract_text_from_frame has skip_detection parameter."""
        from core.analysis.ocr import extract_text_from_frame
        import inspect
        sig = inspect.signature(extract_text_from_frame)
        assert 'skip_detection' in sig.parameters

    def test_ocr_clip_function_has_use_text_detection_param(self):
        """Test that extract_text_from_clip has use_text_detection parameter."""
        from core.analysis.ocr import extract_text_from_clip
        import inspect
        sig = inspect.signature(extract_text_from_clip)
        assert 'use_text_detection' in sig.parameters


class TestEastModelDownload:
    """Tests for _download_east_model: failed downloads must not leave a model file."""

    class _FakeResponse:
        """Minimal urlopen() response that yields chunks, then optionally raises."""

        def __init__(self, chunks, error=None):
            self._chunks = list(chunks)
            self._error = error
            self.closed = False

        def read(self, n=-1):
            if self._chunks:
                return self._chunks.pop(0)
            if self._error is not None:
                raise self._error
            return b""

        def close(self):
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    def _patch_urlopen(self, monkeypatch, response):
        import urllib.request

        def fake_urlopen(url, *args, **kwargs):
            return response

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    def test_interrupted_download_leaves_no_partial_model(self, tmp_path, monkeypatch):
        """A connection reset mid-download must not leave a partial model at dest."""
        from core.analysis.text_detection import _download_east_model

        response = self._FakeResponse(
            [b"\x01" * 4096, b"\x02" * 4096],
            error=ConnectionResetError("connection reset by peer"),
        )
        self._patch_urlopen(monkeypatch, response)

        dest = tmp_path / "frozen_east_text_detection.pb"
        result = _download_east_model(dest)

        assert result is False
        assert not dest.exists()
        assert list(tmp_path.iterdir()) == []

    def test_empty_download_is_not_installed(self, tmp_path, monkeypatch):
        """An empty response body must not be installed as the model."""
        from core.analysis.text_detection import _download_east_model

        self._patch_urlopen(monkeypatch, self._FakeResponse([]))

        dest = tmp_path / "frozen_east_text_detection.pb"
        result = _download_east_model(dest)

        assert result is False
        assert not dest.exists()
        assert list(tmp_path.iterdir()) == []

    def test_successful_download_installs_model(self, tmp_path, monkeypatch):
        """A complete download installs exactly the downloaded bytes at dest."""
        from core.analysis.text_detection import _download_east_model

        payload = [b"EAST" * 1024, b"model-bytes" * 100]
        self._patch_urlopen(monkeypatch, self._FakeResponse(payload))

        dest = tmp_path / "frozen_east_text_detection.pb"
        result = _download_east_model(dest)

        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == b"".join(payload)
        assert list(tmp_path.iterdir()) == [dest]

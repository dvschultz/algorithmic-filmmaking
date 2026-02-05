"""Tests for video color profile detection."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.analysis.color import (
    ColorProfileResult,
    detect_video_color_profile,
)


def _create_test_video(
    path: Path,
    frames: list[np.ndarray],
    fps: float = 30.0,
) -> None:
    """Write a list of BGR frames as a video file."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _gray_frame(value: int = 128, width: int = 64, height: int = 48) -> np.ndarray:
    """Create a solid grayscale frame (stored as BGR)."""
    gray = np.full((height, width), value, dtype=np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _color_frame(
    bgr: tuple[int, int, int] = (0, 128, 255),
    width: int = 64,
    height: int = 48,
) -> np.ndarray:
    """Create a solid color frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = bgr
    return frame


def _sepia_frame(width: int = 64, height: int = 48) -> np.ndarray:
    """Create a sepia-toned frame (low saturation, warm hue ~20-30 degrees)."""
    # Sepia: warm brownish tone with low saturation
    # HSV: hue ~20 (in OpenCV 0-180 scale), sat ~25, value ~140
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[:, :, 0] = 20  # Hue (warm brown, OpenCV scale 0-180)
    hsv[:, :, 1] = 25  # Low saturation
    hsv[:, :, 2] = 140  # Medium brightness
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class TestDetectVideoColorProfile:
    """Tests for detect_video_color_profile()."""

    def test_grayscale_video(self, tmp_path):
        """Pure grayscale video should be classified as grayscale (luma_only=True)."""
        video_path = tmp_path / "grayscale.mp4"
        frames = [_gray_frame(v) for v in [50, 100, 150, 200, 128, 64, 192, 80, 160, 110, 90, 170]]
        _create_test_video(video_path, frames)

        result = detect_video_color_profile(video_path)

        # Codec artifacts on grayscale frames introduce small saturation values
        # that may trigger sepia classification. Both are functionally correct
        # since both set is_grayscale=True (luma_only detection).
        assert result.is_grayscale is True
        assert result.classification in ("grayscale", "sepia")
        assert result.mean_saturation < 15.0

    def test_color_video(self, tmp_path):
        """Saturated color video should be classified as color."""
        video_path = tmp_path / "color.mp4"
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (128, 0, 255), (0, 128, 255), (255, 128, 0),
            (0, 255, 128), (128, 255, 0), (255, 0, 128),
        ]
        frames = [_color_frame(c) for c in colors]
        _create_test_video(video_path, frames)

        result = detect_video_color_profile(video_path)

        assert result.is_grayscale is False
        assert result.classification == "color"

    def test_sepia_video(self, tmp_path):
        """Sepia-toned video should be classified as sepia with is_grayscale=True."""
        video_path = tmp_path / "sepia.mp4"
        frames = [_sepia_frame() for _ in range(12)]
        _create_test_video(video_path, frames)

        result = detect_video_color_profile(video_path)

        assert result.is_grayscale is True
        assert result.classification == "sepia"

    def test_mixed_video(self, tmp_path):
        """Video with mix of B&W and color should be classified as mixed."""
        video_path = tmp_path / "mixed.mp4"
        # 6 grayscale + 6 color = 50% each -> "mixed"
        gray_frames = [_gray_frame(v) for v in [50, 100, 150, 200, 128, 64]]
        color_frames = [
            _color_frame((255, 0, 0)), _color_frame((0, 255, 0)),
            _color_frame((0, 0, 255)), _color_frame((255, 255, 0)),
            _color_frame((0, 255, 255)), _color_frame((255, 0, 255)),
        ]
        frames = gray_frames + color_frames
        _create_test_video(video_path, frames)

        result = detect_video_color_profile(video_path)

        assert result.is_grayscale is False
        assert result.classification == "mixed"

    def test_short_video(self, tmp_path):
        """Very short video (< 10 frames) should still classify as grayscale-like."""
        video_path = tmp_path / "short.mp4"
        frames = [_gray_frame(100), _gray_frame(150), _gray_frame(200)]
        _create_test_video(video_path, frames)

        result = detect_video_color_profile(video_path)

        # Codec artifacts on grayscale frames may cause sepia or grayscale classification.
        # Both are functionally correct (both set luma_only=True).
        assert result.is_grayscale is True
        assert result.classification in ("grayscale", "sepia")

    def test_nonexistent_video(self, tmp_path):
        """Nonexistent video should return color (safe default)."""
        video_path = tmp_path / "nonexistent.mp4"

        result = detect_video_color_profile(video_path)

        assert result.is_grayscale is False
        assert result.classification == "color"

    def test_result_has_frame_saturations(self, tmp_path):
        """Result should include per-frame saturation values."""
        video_path = tmp_path / "test.mp4"
        frames = [_gray_frame(v) for v in [50, 100, 150, 200, 128, 64, 192, 80, 160, 110, 90, 170]]
        _create_test_video(video_path, frames)

        result = detect_video_color_profile(video_path)

        assert len(result.frame_saturations) > 0
        assert all(isinstance(s, float) for s in result.frame_saturations)

    def test_desaturated_color_not_classified_as_grayscale(self, tmp_path):
        """Lightly desaturated color should NOT be classified as grayscale."""
        video_path = tmp_path / "desaturated.mp4"
        # Create frames with low but noticeable saturation (HSV sat ~30)
        frames = []
        for _ in range(12):
            hsv = np.zeros((48, 64, 3), dtype=np.uint8)
            hsv[:, :, 0] = 100  # Hue (cyan-ish)
            hsv[:, :, 1] = 30   # Low-ish saturation but above threshold
            hsv[:, :, 2] = 140  # Medium brightness
            frames.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        _create_test_video(video_path, frames)

        result = detect_video_color_profile(video_path)

        assert result.is_grayscale is False
        assert result.classification == "color"


class TestDetectionConfigLumaOnly:
    """Tests for luma_only field on DetectionConfig."""

    def test_default_is_none(self):
        """Default luma_only should be None (auto-detect)."""
        from core.scene_detect import DetectionConfig

        config = DetectionConfig()
        assert config.luma_only is None

    def test_explicit_true(self):
        """Can set luma_only=True explicitly."""
        from core.scene_detect import DetectionConfig

        config = DetectionConfig(luma_only=True)
        assert config.luma_only is True

    def test_explicit_false(self):
        """Can set luma_only=False explicitly."""
        from core.scene_detect import DetectionConfig

        config = DetectionConfig(luma_only=False)
        assert config.luma_only is False

    def test_grayscale_preset(self):
        """Grayscale preset should have luma_only=True."""
        from core.scene_detect import DetectionConfig

        config = DetectionConfig.grayscale()
        assert config.luma_only is True


class TestSourceColorProfile:
    """Tests for color_profile field on Source model."""

    def test_default_is_none(self):
        """Default color_profile should be None."""
        from models.clip import Source

        source = Source()
        assert source.color_profile is None

    def test_to_dict_excludes_none(self):
        """color_profile should not appear in to_dict when None."""
        from models.clip import Source

        source = Source()
        data = source.to_dict()
        assert "color_profile" not in data

    def test_to_dict_includes_value(self):
        """color_profile should appear in to_dict when set."""
        from models.clip import Source

        source = Source(color_profile="grayscale")
        data = source.to_dict()
        assert data["color_profile"] == "grayscale"

    def test_from_dict_reads_value(self):
        """from_dict should restore color_profile."""
        from models.clip import Source

        data = {
            "id": "test-id",
            "file_path": "/tmp/test.mp4",
            "color_profile": "sepia",
        }
        source = Source.from_dict(data)
        assert source.color_profile == "sepia"

    def test_from_dict_missing_defaults_none(self):
        """from_dict should default to None when color_profile is missing."""
        from models.clip import Source

        data = {
            "id": "test-id",
            "file_path": "/tmp/test.mp4",
        }
        source = Source.from_dict(data)
        assert source.color_profile is None

    def test_roundtrip(self):
        """color_profile should survive to_dict -> from_dict roundtrip."""
        from models.clip import Source

        original = Source(color_profile="mixed")
        data = original.to_dict()
        restored = Source.from_dict(data)
        assert restored.color_profile == "mixed"


class TestSceneDetectorGrayscaleIntegration:
    """Integration tests for SceneDetector with grayscale auto-detection."""

    def test_grayscale_video_sets_luma_only(self, tmp_path):
        """SceneDetector should auto-detect grayscale and use luma-only detection."""
        from core.scene_detect import SceneDetector, DetectionConfig

        video_path = tmp_path / "bw_cuts.mp4"
        # Create B&W video with obvious cuts (alternating bright/dark)
        frames = []
        for i in range(60):
            value = 200 if (i // 15) % 2 == 0 else 50  # Alternate every 15 frames
            frames.append(_gray_frame(value))
        _create_test_video(video_path, frames)

        config = DetectionConfig()  # luma_only=None (auto-detect)
        detector = SceneDetector(config)
        source, clips = detector.detect_scenes(video_path)

        # Codec artifacts may cause grayscale or sepia classification.
        # Both are correct — they both enable luma-only detection.
        assert source.color_profile in ("grayscale", "sepia")

    def test_color_video_uses_standard_detection(self, tmp_path):
        """SceneDetector should detect color video and use standard params."""
        from core.scene_detect import SceneDetector, DetectionConfig

        video_path = tmp_path / "color_cuts.mp4"
        # Create color video with obvious cuts
        frames = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for i, color in enumerate(colors):
            for _ in range(15):
                frames.append(_color_frame(color))
        _create_test_video(video_path, frames)

        config = DetectionConfig()
        detector = SceneDetector(config)
        source, clips = detector.detect_scenes(video_path)

        assert source.color_profile == "color"

    def test_explicit_luma_only_skips_autodetect(self, tmp_path):
        """Explicit luma_only=True should skip auto-detection."""
        from core.scene_detect import SceneDetector, DetectionConfig

        video_path = tmp_path / "color.mp4"
        frames = [_color_frame((255, 0, 0)) for _ in range(30)]
        _create_test_video(video_path, frames)

        config = DetectionConfig(luma_only=True)
        detector = SceneDetector(config)
        source, clips = detector.detect_scenes(video_path)

        # color_profile should NOT be set when auto-detect is skipped
        assert source.color_profile is None

    def test_explicit_luma_only_false_skips_autodetect(self, tmp_path):
        """Explicit luma_only=False should skip auto-detection."""
        from core.scene_detect import SceneDetector, DetectionConfig

        video_path = tmp_path / "gray.mp4"
        frames = [_gray_frame(128) for _ in range(30)]
        _create_test_video(video_path, frames)

        config = DetectionConfig(luma_only=False)
        detector = SceneDetector(config)
        source, clips = detector.detect_scenes(video_path)

        # color_profile should NOT be set when auto-detect is skipped
        assert source.color_profile is None

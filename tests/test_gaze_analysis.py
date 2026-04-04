"""Tests for gaze direction analysis (core/analysis/gaze.py).

All tests work WITHOUT mediapipe installed. Pure logic functions are tested
directly; functions requiring mediapipe use unittest.mock to mock imports.
"""

import math
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# categorize_gaze — pure logic, no mediapipe needed
# ---------------------------------------------------------------------------


class TestCategorizeGaze:
    """Tests for categorize_gaze() threshold and priority logic."""

    def test_center_gaze_returns_at_camera(self):
        from core.analysis.gaze import categorize_gaze

        assert categorize_gaze(0.0, 0.0) == "at_camera"

    def test_small_angles_return_at_camera(self):
        from core.analysis.gaze import categorize_gaze

        # Below both thresholds
        assert categorize_gaze(5.0, 3.0) == "at_camera"
        assert categorize_gaze(-9.9, 7.9) == "at_camera"

    def test_positive_yaw_returns_looking_right(self):
        from core.analysis.gaze import categorize_gaze

        assert categorize_gaze(15.0, 0.0) == "looking_right"
        assert categorize_gaze(10.1, 0.0) == "looking_right"

    def test_negative_yaw_returns_looking_left(self):
        from core.analysis.gaze import categorize_gaze

        assert categorize_gaze(-15.0, 0.0) == "looking_left"
        assert categorize_gaze(-10.1, 0.0) == "looking_left"

    def test_positive_pitch_returns_looking_down(self):
        from core.analysis.gaze import categorize_gaze

        assert categorize_gaze(0.0, 12.0) == "looking_down"
        assert categorize_gaze(0.0, 8.1) == "looking_down"

    def test_negative_pitch_returns_looking_up(self):
        from core.analysis.gaze import categorize_gaze

        assert categorize_gaze(0.0, -12.0) == "looking_up"
        assert categorize_gaze(0.0, -8.1) == "looking_up"

    def test_yaw_priority_when_both_exceed_thresholds(self):
        """When both yaw and pitch exceed thresholds, yaw wins."""
        from core.analysis.gaze import categorize_gaze

        # Both exceed: yaw positive, pitch positive
        assert categorize_gaze(20.0, 15.0) == "looking_right"
        # Both exceed: yaw negative, pitch negative
        assert categorize_gaze(-20.0, -15.0) == "looking_left"
        # Both exceed: yaw positive, pitch negative
        assert categorize_gaze(15.0, -10.0) == "looking_right"

    def test_exactly_at_threshold_returns_at_camera(self):
        """Thresholds use strict inequality (>), not >=."""
        from core.analysis.gaze import categorize_gaze

        assert categorize_gaze(10.0, 0.0) == "at_camera"
        assert categorize_gaze(0.0, 8.0) == "at_camera"
        assert categorize_gaze(-10.0, -8.0) == "at_camera"

    def test_pitch_only_exceeds_when_yaw_below_threshold(self):
        from core.analysis.gaze import categorize_gaze

        assert categorize_gaze(5.0, 15.0) == "looking_down"
        assert categorize_gaze(-3.0, -12.0) == "looking_up"


# ---------------------------------------------------------------------------
# Constants validation
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify gaze constants are correct."""

    def test_threshold_values(self):
        from core.analysis.gaze import (
            GAZE_YAW_THRESHOLD,
            GAZE_PITCH_THRESHOLD,
            MAX_YAW_ANGLE,
            MAX_PITCH_ANGLE,
        )

        assert GAZE_YAW_THRESHOLD == 10.0
        assert GAZE_PITCH_THRESHOLD == 8.0
        assert MAX_YAW_ANGLE == 30.0
        assert MAX_PITCH_ANGLE == 20.0

    def test_iris_landmark_indices(self):
        from core.analysis.gaze import LEFT_IRIS, RIGHT_IRIS

        assert LEFT_IRIS == [468, 469, 470, 471, 472]
        assert RIGHT_IRIS == [473, 474, 475, 476, 477]

    def test_eye_corner_indices(self):
        from core.analysis.gaze import (
            LEFT_EYE_INNER,
            LEFT_EYE_OUTER,
            RIGHT_EYE_INNER,
            RIGHT_EYE_OUTER,
            LEFT_EYE_TOP,
            LEFT_EYE_BOTTOM,
            RIGHT_EYE_TOP,
            RIGHT_EYE_BOTTOM,
        )

        assert LEFT_EYE_INNER == 133
        assert LEFT_EYE_OUTER == 33
        assert RIGHT_EYE_INNER == 362
        assert RIGHT_EYE_OUTER == 263
        assert LEFT_EYE_TOP == 159
        assert LEFT_EYE_BOTTOM == 145
        assert RIGHT_EYE_TOP == 386
        assert RIGHT_EYE_BOTTOM == 374


# ---------------------------------------------------------------------------
# _select_largest_face
# ---------------------------------------------------------------------------


class TestSelectLargestFace:
    """Test face selection by bounding box area."""

    def _make_face(self, landmarks_data):
        """Create a mock face landmark list.

        landmarks_data: list of (x, y) normalized coordinates
        Returns a plain list of landmark objects (Tasks API format).
        """
        Landmark = namedtuple("Landmark", ["x", "y", "z"])
        return [Landmark(x=x, y=y, z=0.0) for x, y in landmarks_data]

    def test_returns_none_for_empty_list(self):
        from core.analysis.gaze import _select_largest_face

        assert _select_largest_face([], 640, 480) is None

    def test_returns_single_face(self):
        from core.analysis.gaze import _select_largest_face

        face = self._make_face([(0.1, 0.1), (0.5, 0.5)])
        result = _select_largest_face([face], 640, 480)
        assert result is face

    def test_returns_largest_of_multiple_faces(self):
        from core.analysis.gaze import _select_largest_face

        # Small face: 0.1 x 0.1 normalized = 64x48 pixels
        small = self._make_face([(0.1, 0.1), (0.2, 0.2)])
        # Large face: 0.5 x 0.5 normalized = 320x240 pixels
        large = self._make_face([(0.2, 0.2), (0.7, 0.7)])

        result = _select_largest_face([small, large], 640, 480)
        assert result is large

        # Order shouldn't matter
        result2 = _select_largest_face([large, small], 640, 480)
        assert result2 is large


# ---------------------------------------------------------------------------
# _compute_iris_ratios
# ---------------------------------------------------------------------------


class TestComputeIrisRatios:
    """Test iris ratio computation with mock landmarks."""

    def _make_landmarks(self, overrides=None):
        """Create mock landmark list with iris centered by default.

        Returns a plain list of 478 landmarks at (0.5, 0.5), then
        applies overrides for specific indices (Tasks API format).
        """
        Landmark = namedtuple("Landmark", ["x", "y", "z"])
        data = [(0.5, 0.5)] * 478
        if overrides:
            for idx, (x, y) in overrides.items():
                data[idx] = (x, y)
        return [Landmark(x=x, y=y, z=0.0) for x, y in data]

    def test_centered_iris_returns_half(self):
        """When iris is centered in the eye, ratios should be ~0.5."""
        from core.analysis.gaze import (
            _compute_iris_ratios,
            LEFT_EYE_OUTER,
            LEFT_EYE_INNER,
            RIGHT_EYE_INNER,
            RIGHT_EYE_OUTER,
            LEFT_EYE_TOP,
            LEFT_EYE_BOTTOM,
            RIGHT_EYE_TOP,
            RIGHT_EYE_BOTTOM,
            LEFT_IRIS,
            RIGHT_IRIS,
        )

        # Set up eye corners so iris is centered
        overrides = {
            # Left eye: outer at 0.3, inner at 0.7, iris center at 0.5
            LEFT_EYE_OUTER: (0.3, 0.5),
            LEFT_EYE_INNER: (0.7, 0.5),
            LEFT_EYE_TOP: (0.5, 0.4),
            LEFT_EYE_BOTTOM: (0.5, 0.6),
            # Right eye: inner at 0.3, outer at 0.7, iris center at 0.5
            RIGHT_EYE_INNER: (0.3, 0.5),
            RIGHT_EYE_OUTER: (0.7, 0.5),
            RIGHT_EYE_TOP: (0.5, 0.4),
            RIGHT_EYE_BOTTOM: (0.5, 0.6),
        }
        # All iris landmarks at center
        for idx in LEFT_IRIS + RIGHT_IRIS:
            overrides[idx] = (0.5, 0.5)

        face = self._make_landmarks(overrides)
        h_ratio, v_ratio = _compute_iris_ratios(face, 640, 480)

        assert abs(h_ratio - 0.5) < 0.01
        assert abs(v_ratio - 0.5) < 0.01

    def test_iris_shifted_right_increases_h_ratio(self):
        """When iris is shifted right (toward inner corner for left eye),
        horizontal ratio should be > 0.5."""
        from core.analysis.gaze import (
            _compute_iris_ratios,
            LEFT_EYE_OUTER,
            LEFT_EYE_INNER,
            RIGHT_EYE_INNER,
            RIGHT_EYE_OUTER,
            LEFT_EYE_TOP,
            LEFT_EYE_BOTTOM,
            RIGHT_EYE_TOP,
            RIGHT_EYE_BOTTOM,
            LEFT_IRIS,
            RIGHT_IRIS,
        )

        overrides = {
            # Left eye horizontal
            LEFT_EYE_OUTER: (0.3, 0.5),
            LEFT_EYE_INNER: (0.7, 0.5),
            LEFT_EYE_TOP: (0.5, 0.4),
            LEFT_EYE_BOTTOM: (0.5, 0.6),
            # Right eye horizontal
            RIGHT_EYE_INNER: (0.3, 0.5),
            RIGHT_EYE_OUTER: (0.7, 0.5),
            RIGHT_EYE_TOP: (0.5, 0.4),
            RIGHT_EYE_BOTTOM: (0.5, 0.6),
        }
        # Shift iris to the right (x = 0.6)
        for idx in LEFT_IRIS + RIGHT_IRIS:
            overrides[idx] = (0.6, 0.5)

        face = self._make_landmarks(overrides)
        h_ratio, v_ratio = _compute_iris_ratios(face, 640, 480)

        assert h_ratio > 0.5

    def test_iris_shifted_down_increases_v_ratio(self):
        """When iris is shifted down, vertical ratio should be > 0.5."""
        from core.analysis.gaze import (
            _compute_iris_ratios,
            LEFT_EYE_OUTER,
            LEFT_EYE_INNER,
            RIGHT_EYE_INNER,
            RIGHT_EYE_OUTER,
            LEFT_EYE_TOP,
            LEFT_EYE_BOTTOM,
            RIGHT_EYE_TOP,
            RIGHT_EYE_BOTTOM,
            LEFT_IRIS,
            RIGHT_IRIS,
        )

        overrides = {
            LEFT_EYE_OUTER: (0.3, 0.5),
            LEFT_EYE_INNER: (0.7, 0.5),
            LEFT_EYE_TOP: (0.5, 0.3),
            LEFT_EYE_BOTTOM: (0.5, 0.7),
            RIGHT_EYE_INNER: (0.3, 0.5),
            RIGHT_EYE_OUTER: (0.7, 0.5),
            RIGHT_EYE_TOP: (0.5, 0.3),
            RIGHT_EYE_BOTTOM: (0.5, 0.7),
        }
        # Shift iris down (y = 0.6)
        for idx in LEFT_IRIS + RIGHT_IRIS:
            overrides[idx] = (0.5, 0.6)

        face = self._make_landmarks(overrides)
        h_ratio, v_ratio = _compute_iris_ratios(face, 640, 480)

        assert v_ratio > 0.5


# ---------------------------------------------------------------------------
# extract_gaze_from_frame (mocked mediapipe)
# ---------------------------------------------------------------------------


class TestExtractGazeFromFrame:
    """Test per-frame gaze extraction with mocked FaceMesh results."""

    def _make_face_landmarks(self, overrides=None):
        """Create mock face landmark list (Tasks API format — plain list)."""
        Landmark = namedtuple("Landmark", ["x", "y", "z"])
        data = [(0.5, 0.5)] * 478
        if overrides:
            for idx, (x, y) in overrides.items():
                data[idx] = (x, y)
        return [Landmark(x=x, y=y, z=0.0) for x, y in data]

    def _centered_eye_overrides(self):
        """Return landmark overrides for a face looking at camera."""
        from core.analysis.gaze import (
            LEFT_EYE_OUTER,
            LEFT_EYE_INNER,
            RIGHT_EYE_INNER,
            RIGHT_EYE_OUTER,
            LEFT_EYE_TOP,
            LEFT_EYE_BOTTOM,
            RIGHT_EYE_TOP,
            RIGHT_EYE_BOTTOM,
            LEFT_IRIS,
            RIGHT_IRIS,
        )

        overrides = {
            LEFT_EYE_OUTER: (0.3, 0.5),
            LEFT_EYE_INNER: (0.7, 0.5),
            LEFT_EYE_TOP: (0.5, 0.4),
            LEFT_EYE_BOTTOM: (0.5, 0.6),
            RIGHT_EYE_INNER: (0.3, 0.5),
            RIGHT_EYE_OUTER: (0.7, 0.5),
            RIGHT_EYE_TOP: (0.5, 0.4),
            RIGHT_EYE_BOTTOM: (0.5, 0.6),
        }
        for idx in LEFT_IRIS + RIGHT_IRIS:
            overrides[idx] = (0.5, 0.5)
        return overrides

    @patch("core.analysis.gaze.mp")
    def test_no_faces_returns_none(self, mock_mp):
        from core.analysis.gaze import extract_gaze_from_frame

        mock_mp.Image.return_value = MagicMock()
        mock_mp.ImageFormat.SRGB = 1

        landmarker = MagicMock()
        landmarker.detect.return_value = SimpleNamespace(face_landmarks=[])
        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        result = extract_gaze_from_frame(landmarker, frame_bgr, 640, 480)
        assert result is None

    @patch("core.analysis.gaze.mp")
    def test_empty_face_list_returns_none(self, mock_mp):
        from core.analysis.gaze import extract_gaze_from_frame

        mock_mp.Image.return_value = MagicMock()
        mock_mp.ImageFormat.SRGB = 1

        landmarker = MagicMock()
        landmarker.detect.return_value = SimpleNamespace(face_landmarks=[])
        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        result = extract_gaze_from_frame(landmarker, frame_bgr, 640, 480)
        assert result is None

    @patch("core.analysis.gaze.mp")
    def test_centered_face_returns_at_camera(self, mock_mp):
        from core.analysis.gaze import extract_gaze_from_frame

        mock_mp.Image.return_value = MagicMock()
        mock_mp.ImageFormat.SRGB = 1

        landmarker = MagicMock()
        overrides = self._centered_eye_overrides()
        face = self._make_face_landmarks(overrides)
        landmarker.detect.return_value = SimpleNamespace(face_landmarks=[face])
        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        result = extract_gaze_from_frame(landmarker, frame_bgr, 640, 480)
        assert result is not None
        yaw, pitch, category = result
        assert category == "at_camera"
        assert abs(yaw) < 5.0
        assert abs(pitch) < 5.0

    @patch("core.analysis.gaze.mp")
    def test_returns_tuple_of_three(self, mock_mp):
        from core.analysis.gaze import extract_gaze_from_frame

        mock_mp.Image.return_value = MagicMock()
        mock_mp.ImageFormat.SRGB = 1

        landmarker = MagicMock()
        overrides = self._centered_eye_overrides()
        face = self._make_face_landmarks(overrides)
        landmarker.detect.return_value = SimpleNamespace(face_landmarks=[face])
        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        result = extract_gaze_from_frame(landmarker, frame_bgr, 640, 480)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)
        assert isinstance(result[2], str)


# ---------------------------------------------------------------------------
# extract_gaze_from_clip (mocked cv2 + mediapipe)
# ---------------------------------------------------------------------------


class TestExtractGazeFromClip:
    """Test clip-level gaze extraction with mocked dependencies."""

    @patch("core.analysis.gaze._load_face_mesh")
    @patch("core.analysis.gaze.cv2")
    def test_invalid_video_returns_none(self, mock_cv2, mock_load):
        from core.analysis.gaze import extract_gaze_from_clip

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        result = extract_gaze_from_clip("/nonexistent.mp4", 0, 90, 30.0)
        assert result is None

    @patch("core.analysis.gaze._load_face_mesh")
    @patch("core.analysis.gaze.cv2")
    def test_zero_duration_returns_none(self, mock_cv2, mock_load):
        from core.analysis.gaze import extract_gaze_from_clip

        result = extract_gaze_from_clip("/video.mp4", 100, 100, 30.0)
        assert result is None

    @patch("core.analysis.gaze._load_face_mesh")
    @patch("core.analysis.gaze.cv2")
    def test_negative_duration_returns_none(self, mock_cv2, mock_load):
        from core.analysis.gaze import extract_gaze_from_clip

        result = extract_gaze_from_clip("/video.mp4", 100, 50, 30.0)
        assert result is None

    @patch("core.analysis.gaze.extract_gaze_from_frame")
    @patch("core.analysis.gaze._load_face_mesh")
    @patch("core.analysis.gaze.cv2")
    def test_all_frames_no_face_returns_none(
        self, mock_cv2, mock_load, mock_extract_frame
    ):
        from core.analysis.gaze import extract_gaze_from_clip

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0}.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_POS_FRAMES = 1

        mock_extract_frame.return_value = None

        result = extract_gaze_from_clip("/video.mp4", 0, 90, 30.0)
        assert result is None

    @patch("core.analysis.gaze.extract_gaze_from_frame")
    @patch("core.analysis.gaze._load_face_mesh")
    @patch("core.analysis.gaze.cv2")
    def test_dominant_category_with_consistent_frames(
        self, mock_cv2, mock_load, mock_extract_frame
    ):
        from core.analysis.gaze import extract_gaze_from_clip

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0}.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_POS_FRAMES = 1

        # 3 frames looking left, 1 looking right
        mock_extract_frame.side_effect = [
            (-15.0, 0.0, "looking_left"),
            (-18.0, 0.0, "looking_left"),
            (-12.0, 0.0, "looking_left"),
            (20.0, 0.0, "looking_right"),
        ]

        # 4 seconds of video at 30fps = 120 frames, sampled at 1fps = 4 samples
        result = extract_gaze_from_clip("/video.mp4", 0, 120, 30.0, 1.0)

        assert result is not None
        assert result["gaze_category"] == "looking_left"
        # Median of [-15.0, -18.0, -12.0] = -15.0
        assert result["gaze_yaw"] == -15.0
        assert result["gaze_pitch"] == 0.0

    @patch("core.analysis.gaze.extract_gaze_from_frame")
    @patch("core.analysis.gaze._load_face_mesh")
    @patch("core.analysis.gaze.cv2")
    def test_short_clip_samples_midpoint(
        self, mock_cv2, mock_load, mock_extract_frame
    ):
        from core.analysis.gaze import extract_gaze_from_clip

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0}.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_POS_FRAMES = 1

        mock_extract_frame.return_value = (5.0, -3.0, "at_camera")

        # 0.5 second clip at 30fps = 15 frames, shorter than 1.0s interval
        result = extract_gaze_from_clip("/video.mp4", 100, 115, 30.0, 1.0)

        assert result is not None
        assert result["gaze_category"] == "at_camera"

        # Should sample midpoint: 100 + 15//2 = 107
        calls = mock_cap.set.call_args_list
        frame_positions = [c[0][1] for c in calls if c[0][0] == 1]
        assert 107 in frame_positions

    @patch("core.analysis.gaze.extract_gaze_from_frame")
    @patch("core.analysis.gaze._load_face_mesh")
    @patch("core.analysis.gaze.cv2")
    def test_result_dict_keys(self, mock_cv2, mock_load, mock_extract_frame):
        from core.analysis.gaze import extract_gaze_from_clip

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0}.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_POS_FRAMES = 1

        mock_extract_frame.return_value = (5.0, -3.0, "at_camera")

        result = extract_gaze_from_clip("/video.mp4", 0, 90, 30.0)

        assert result is not None
        assert "gaze_yaw" in result
        assert "gaze_pitch" in result
        assert "gaze_category" in result
        assert isinstance(result["gaze_yaw"], float)
        assert isinstance(result["gaze_pitch"], float)
        assert isinstance(result["gaze_category"], str)

    @patch("core.analysis.gaze.extract_gaze_from_frame")
    @patch("core.analysis.gaze._load_face_mesh")
    @patch("core.analysis.gaze.cv2")
    def test_read_failure_frames_skipped(
        self, mock_cv2, mock_load, mock_extract_frame
    ):
        """When cv2.read() fails for some frames, those are skipped."""
        from core.analysis.gaze import extract_gaze_from_clip

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # Alternate success and failure reads
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
        ]
        mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0}.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_POS_FRAMES = 1

        mock_extract_frame.return_value = (0.0, 0.0, "at_camera")

        # 3 seconds at 30fps = 90 frames, sampled at 1fps = 3 samples
        result = extract_gaze_from_clip("/video.mp4", 0, 90, 30.0, 1.0)

        assert result is not None
        # extract_gaze_from_frame called twice (2 successful reads)
        assert mock_extract_frame.call_count == 2

    @patch("core.analysis.gaze.extract_gaze_from_frame")
    @patch("core.analysis.gaze._load_face_mesh")
    @patch("core.analysis.gaze.cv2")
    def test_video_cap_released_on_success(
        self, mock_cv2, mock_load, mock_extract_frame
    ):
        from core.analysis.gaze import extract_gaze_from_clip

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0}.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_POS_FRAMES = 1

        mock_extract_frame.return_value = (0.0, 0.0, "at_camera")

        extract_gaze_from_clip("/video.mp4", 0, 90, 30.0)

        mock_cap.release.assert_called_once()

    @patch("core.analysis.gaze.extract_gaze_from_frame")
    @patch("core.analysis.gaze._load_face_mesh")
    @patch("core.analysis.gaze.cv2")
    def test_video_cap_released_on_exception(
        self, mock_cv2, mock_load, mock_extract_frame
    ):
        """VideoCapture is released even when processing raises."""
        from core.analysis.gaze import extract_gaze_from_clip

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = RuntimeError("corrupt frame")
        mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0}.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_POS_FRAMES = 1

        with pytest.raises(RuntimeError):
            extract_gaze_from_clip("/video.mp4", 0, 90, 30.0)

        mock_cap.release.assert_called_once()


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------


class TestModelLifecycle:
    """Test model loading/unloading state management."""

    def test_is_model_loaded_initially_false(self):
        from core.analysis.gaze import is_model_loaded

        # Reset module state (may have been set by other tests)
        import core.analysis.gaze as gaze_mod
        original = gaze_mod._model
        try:
            gaze_mod._model = None
            assert is_model_loaded() is False
        finally:
            gaze_mod._model = original

    def test_unload_model_sets_to_none(self):
        import core.analysis.gaze as gaze_mod

        original = gaze_mod._model
        try:
            gaze_mod._model = MagicMock()
            assert gaze_mod.is_model_loaded() is True

            gaze_mod.unload_model()
            assert gaze_mod.is_model_loaded() is False
        finally:
            gaze_mod._model = original

    def test_unload_calls_close_on_model(self):
        import core.analysis.gaze as gaze_mod

        original = gaze_mod._model
        mock_model = MagicMock()
        try:
            gaze_mod._model = mock_model
            gaze_mod.unload_model()
            mock_model.close.assert_called_once()
        finally:
            gaze_mod._model = original

    def test_unload_when_already_none_is_safe(self):
        import core.analysis.gaze as gaze_mod

        original = gaze_mod._model
        try:
            gaze_mod._model = None
            # Should not raise
            gaze_mod.unload_model()
            assert gaze_mod._model is None
        finally:
            gaze_mod._model = original


# ---------------------------------------------------------------------------
# Angle-to-category integration
# ---------------------------------------------------------------------------


class TestAngleToCategoryIntegration:
    """Test the full pipeline from iris ratios to categories.

    These test the math: h_ratio -> yaw_deg -> category chain.
    """

    def test_ratio_0_5_gives_center(self):
        """h_ratio=0.5, v_ratio=0.5 should give ~0 degrees, at_camera."""
        from core.analysis.gaze import MAX_YAW_ANGLE, MAX_PITCH_ANGLE, categorize_gaze

        yaw = (0.5 - 0.5) * 2 * MAX_YAW_ANGLE
        pitch = (0.5 - 0.5) * 2 * MAX_PITCH_ANGLE
        assert yaw == 0.0
        assert pitch == 0.0
        assert categorize_gaze(yaw, pitch) == "at_camera"

    def test_ratio_0_75_gives_looking_right(self):
        """h_ratio=0.75 should give +15 degrees yaw."""
        from core.analysis.gaze import MAX_YAW_ANGLE, categorize_gaze

        yaw = (0.75 - 0.5) * 2 * MAX_YAW_ANGLE  # = 15.0
        assert yaw == pytest.approx(15.0)
        assert categorize_gaze(yaw, 0.0) == "looking_right"

    def test_ratio_0_25_gives_looking_left(self):
        """h_ratio=0.25 should give -15 degrees yaw."""
        from core.analysis.gaze import MAX_YAW_ANGLE, categorize_gaze

        yaw = (0.25 - 0.5) * 2 * MAX_YAW_ANGLE  # = -15.0
        assert yaw == pytest.approx(-15.0)
        assert categorize_gaze(yaw, 0.0) == "looking_left"

    def test_ratio_0_75_vertical_gives_looking_down(self):
        """v_ratio=0.75 should give +10 degrees pitch."""
        from core.analysis.gaze import MAX_PITCH_ANGLE, categorize_gaze

        pitch = (0.75 - 0.5) * 2 * MAX_PITCH_ANGLE  # = 10.0
        assert pitch == pytest.approx(10.0)
        assert categorize_gaze(0.0, pitch) == "looking_down"

    def test_ratio_0_25_vertical_gives_looking_up(self):
        """v_ratio=0.25 should give -10 degrees pitch."""
        from core.analysis.gaze import MAX_PITCH_ANGLE, categorize_gaze

        pitch = (0.25 - 0.5) * 2 * MAX_PITCH_ANGLE  # = -10.0
        assert pitch == pytest.approx(-10.0)
        assert categorize_gaze(0.0, pitch) == "looking_up"

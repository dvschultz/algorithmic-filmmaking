"""Tests for gaze direction analysis fields on the Clip model.

Covers:
- Round-trip serialization (Clip -> to_dict -> from_dict -> same values)
- Compact serialization (None fields omitted from to_dict output)
- Backward compatibility (missing keys default to None)
- Invalid type handling (validated via _validate_optional_float)
- Unexpected category values (accepted without validation)
- Float rounding in serialization
"""

from models.clip import Clip


class TestGazeFieldDefaults:
    """Verify gaze fields default to None."""

    def test_gaze_yaw_default_none(self):
        clip = Clip(source_id="s1", start_frame=0, end_frame=100)
        assert clip.gaze_yaw is None

    def test_gaze_pitch_default_none(self):
        clip = Clip(source_id="s1", start_frame=0, end_frame=100)
        assert clip.gaze_pitch is None

    def test_gaze_category_default_none(self):
        clip = Clip(source_id="s1", start_frame=0, end_frame=100)
        assert clip.gaze_category is None


class TestGazeSerialization:
    """Test to_dict serialization of gaze fields."""

    def test_to_dict_includes_gaze_fields_when_set(self):
        clip = Clip(
            id="test-1",
            source_id="s1",
            start_frame=0,
            end_frame=100,
            gaze_yaw=-5.23,
            gaze_pitch=12.67,
            gaze_category="looking_left",
        )
        data = clip.to_dict()
        assert data["gaze_yaw"] == -5.23
        assert data["gaze_pitch"] == 12.67
        assert data["gaze_category"] == "looking_left"

    def test_to_dict_omits_none_gaze_fields(self):
        """None gaze fields should not appear in serialized dict."""
        clip = Clip(id="test-2", source_id="s1", start_frame=0, end_frame=100)
        data = clip.to_dict()
        assert "gaze_yaw" not in data
        assert "gaze_pitch" not in data
        assert "gaze_category" not in data

    def test_to_dict_rounds_floats_to_2_decimals(self):
        clip = Clip(
            source_id="s1",
            start_frame=0,
            end_frame=100,
            gaze_yaw=3.14159265,
            gaze_pitch=-7.777777,
        )
        data = clip.to_dict()
        assert data["gaze_yaw"] == 3.14
        assert data["gaze_pitch"] == -7.78

    def test_to_dict_partial_gaze_fields(self):
        """Only set gaze_yaw, others should be omitted."""
        clip = Clip(
            source_id="s1",
            start_frame=0,
            end_frame=100,
            gaze_yaw=10.0,
        )
        data = clip.to_dict()
        assert data["gaze_yaw"] == 10.0
        assert "gaze_pitch" not in data
        assert "gaze_category" not in data


class TestGazeDeserialization:
    """Test from_dict deserialization of gaze fields."""

    def test_from_dict_restores_gaze_fields(self):
        data = {
            "id": "test-3",
            "source_id": "s1",
            "start_frame": 0,
            "end_frame": 100,
            "gaze_yaw": -15.5,
            "gaze_pitch": 8.3,
            "gaze_category": "at_camera",
        }
        clip = Clip.from_dict(data)
        assert clip.gaze_yaw == -15.5
        assert clip.gaze_pitch == 8.3
        assert clip.gaze_category == "at_camera"

    def test_from_dict_backward_compat_missing_keys(self):
        """Old project files without gaze keys should default to None."""
        data = {
            "id": "test-4",
            "source_id": "s1",
            "start_frame": 0,
            "end_frame": 100,
        }
        clip = Clip.from_dict(data)
        assert clip.gaze_yaw is None
        assert clip.gaze_pitch is None
        assert clip.gaze_category is None

    def test_from_dict_invalid_yaw_type_string(self):
        """Invalid type for gaze_yaw (string) should be validated to None."""
        data = {
            "id": "test-5",
            "source_id": "s1",
            "start_frame": 0,
            "end_frame": 100,
            "gaze_yaw": "not_a_number",
            "gaze_pitch": 5.0,
        }
        clip = Clip.from_dict(data)
        assert clip.gaze_yaw is None
        assert clip.gaze_pitch == 5.0

    def test_from_dict_invalid_pitch_type_list(self):
        """Invalid type for gaze_pitch (list) should be validated to None."""
        data = {
            "id": "test-6",
            "source_id": "s1",
            "start_frame": 0,
            "end_frame": 100,
            "gaze_yaw": 2.0,
            "gaze_pitch": [1, 2, 3],
        }
        clip = Clip.from_dict(data)
        assert clip.gaze_yaw == 2.0
        assert clip.gaze_pitch is None

    def test_from_dict_int_values_converted_to_float(self):
        """Integer values for yaw/pitch should be accepted and converted to float."""
        data = {
            "source_id": "s1",
            "start_frame": 0,
            "end_frame": 100,
            "gaze_yaw": 10,
            "gaze_pitch": -5,
        }
        clip = Clip.from_dict(data)
        assert clip.gaze_yaw == 10.0
        assert isinstance(clip.gaze_yaw, float)
        assert clip.gaze_pitch == -5.0
        assert isinstance(clip.gaze_pitch, float)

    def test_from_dict_unexpected_category_accepted(self):
        """Unexpected gaze_category values should be accepted (no enum validation)."""
        data = {
            "source_id": "s1",
            "start_frame": 0,
            "end_frame": 100,
            "gaze_category": "looking_diagonal_upper_left",
        }
        clip = Clip.from_dict(data)
        assert clip.gaze_category == "looking_diagonal_upper_left"


class TestGazeRoundTrip:
    """Test full round-trip serialization."""

    def test_round_trip_with_all_gaze_fields(self):
        original = Clip(
            id="rt-1",
            source_id="s1",
            start_frame=0,
            end_frame=100,
            gaze_yaw=-12.34,
            gaze_pitch=5.67,
            gaze_category="looking_right",
        )
        data = original.to_dict()
        restored = Clip.from_dict(data)
        assert restored.gaze_yaw == -12.34
        assert restored.gaze_pitch == 5.67
        assert restored.gaze_category == "looking_right"

    def test_round_trip_with_no_gaze_fields(self):
        original = Clip(
            id="rt-2",
            source_id="s1",
            start_frame=0,
            end_frame=100,
        )
        data = original.to_dict()
        restored = Clip.from_dict(data)
        assert restored.gaze_yaw is None
        assert restored.gaze_pitch is None
        assert restored.gaze_category is None

    def test_round_trip_preserves_other_fields(self):
        """Gaze fields should not interfere with existing fields."""
        original = Clip(
            id="rt-3",
            source_id="s1",
            start_frame=10,
            end_frame=200,
            shot_type="close-up",
            average_brightness=0.75,
            rms_volume=-12.5,
            gaze_yaw=3.0,
            gaze_pitch=-1.5,
            gaze_category="at_camera",
        )
        data = original.to_dict()
        restored = Clip.from_dict(data)
        assert restored.shot_type == "close-up"
        assert restored.average_brightness == 0.75
        assert restored.rms_volume == -12.5
        assert restored.gaze_yaw == 3.0
        assert restored.gaze_pitch == -1.5
        assert restored.gaze_category == "at_camera"

    def test_round_trip_at_camera_category(self):
        original = Clip(
            source_id="s1",
            start_frame=0,
            end_frame=50,
            gaze_yaw=0.5,
            gaze_pitch=-0.3,
            gaze_category="at_camera",
        )
        data = original.to_dict()
        restored = Clip.from_dict(data)
        assert restored.gaze_category == "at_camera"
        assert restored.gaze_yaw == 0.5
        assert restored.gaze_pitch == -0.3

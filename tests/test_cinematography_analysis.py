"""Regression tests for cinematography analysis parsing."""

import pytest

from core.analysis.cinematography import _parse_json_response


def test_parse_json_response_accepts_list_wrapped_object():
    payload = """
    [
      {
        "shot_size": "CU",
        "camera_angle": "eye_level",
        "subject_position": "center",
        "subject_count": "single",
        "focus_type": "shallow",
        "lighting_style": "natural",
        "lighting_direction": "front",
        "emotional_intensity": "medium",
        "suggested_pacing": "medium"
      }
    ]
    """

    result = _parse_json_response(payload)

    assert result["shot_size"] == "CU"
    assert result["camera_angle"] == "eye_level"


def test_parse_json_response_rejects_list_without_object():
    with pytest.raises(ValueError, match="JSON list did not contain an object"):
        _parse_json_response('["CU", "eye_level"]')

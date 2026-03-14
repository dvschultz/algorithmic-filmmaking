"""Tests for Sequence chromatic color-bar persistence."""

from models.sequence import Sequence


def test_sequence_serializes_chromatic_bar_flag_when_enabled():
    sequence = Sequence(algorithm="color", show_chromatic_color_bar=True)

    data = sequence.to_dict()

    assert data["show_chromatic_color_bar"] is True
    restored = Sequence.from_dict(data)
    assert restored.show_chromatic_color_bar is True


def test_sequence_chromatic_bar_defaults_to_disabled():
    sequence = Sequence()

    data = sequence.to_dict()

    assert "show_chromatic_color_bar" not in data
    restored = Sequence.from_dict(data)
    assert restored.show_chromatic_color_bar is False


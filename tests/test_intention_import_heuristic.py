"""Tests for the clip count estimation heuristic in the intention import dialog."""

from ui.dialogs.intention_import_dialog import (
    _CLIPS_PER_MINUTE,
    _DEFAULT_URL_DURATION_SECONDS,
)


def test_clips_per_minute_constant():
    """Guard against accidental changes to the heuristic."""
    assert _CLIPS_PER_MINUTE == 12.0


def test_clip_count_heuristic_5min_video():
    """A 5-minute video should estimate ~60 clips."""
    total_minutes = _DEFAULT_URL_DURATION_SECONDS / 60.0
    expected = max(1, round(total_minutes * _CLIPS_PER_MINUTE))
    assert expected == 60


def test_clip_count_heuristic_short_video():
    """A 3-second video should estimate at least 1 clip."""
    total_minutes = 3.0 / 60.0
    expected = max(1, round(total_minutes * _CLIPS_PER_MINUTE))
    assert expected == 1

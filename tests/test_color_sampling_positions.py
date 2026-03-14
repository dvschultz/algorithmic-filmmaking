"""Tests for frame sampling positions in color analysis."""

from core.analysis.color import _sample_frame_positions


def test_sample_positions_single_frame_clip_stays_in_bounds():
    positions = _sample_frame_positions(start_frame=100, end_frame=101)
    assert positions == [100]


def test_sample_positions_two_frame_clip_stays_in_bounds():
    positions = _sample_frame_positions(start_frame=100, end_frame=102)
    assert positions == [101]


def test_sample_positions_never_exceed_exclusive_end():
    positions = _sample_frame_positions(start_frame=50, end_frame=51)
    assert all(50 <= p < 51 for p in positions)


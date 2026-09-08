"""Distinguish presentation gaps from timebase rounding."""

from core.media_timing import timing_from_probe


def test_frame_sized_timebase_does_not_hide_dropped_frame():
    timing = timing_from_probe({
        "streams": [{"time_base": "1/24", "r_frame_rate": "24/1"}],
        "frames": [{"best_effort_timestamp": pts, "duration": 1} for pts in (0, 1, 3, 4)],
    })
    assert timing.variable


def test_millisecond_rounding_does_not_mark_constant_rate_variable():
    timing = timing_from_probe({
        "streams": [{"time_base": "1/1000", "r_frame_rate": "24/1"}],
        "frames": [{"best_effort_timestamp": pts, "duration": 42} for pts in (0, 42, 83, 125)],
    })
    assert not timing.variable

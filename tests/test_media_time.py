"""Exact source/timeline coordinates and shared boundary rounding."""

from fractions import Fraction

import pytest

from models.media_time import (
    AudioRange, StillHold, TimelineRange, VideoRange, frame_boundary, frame_rate,
    nearest_source_frame, source_frame_time,
)


@pytest.mark.parametrize("rate", [24, 25, 30, Fraction(30000, 1001)])
def test_nonzero_source_range_keeps_source_rate(rate):
    source = VideoRange(240, 264, rate)
    timeline = TimelineRange(Fraction(3), Fraction(3) + source.duration)
    assert source.start == Fraction(240) / rate
    assert source.end == Fraction(264) / rate
    assert timeline.duration == source.duration
    assert frame_rate(float(rate)) == rate


def test_long_mixed_rate_sequence_rounds_shared_boundaries_once():
    cursor = Fraction(0)
    previous_end = 0
    total_frames = 0
    rates = [24, 25, 30, Fraction(30000, 1001)]
    for index in range(20_000):
        duration = VideoRange(240, 241, rates[index % 4]).duration
        span = TimelineRange(cursor, cursor + duration)
        start, end = span.frames(Fraction(30))
        assert start == previous_end
        total_frames += end - start
        cursor = span.end
        previous_end = end
    assert total_frames == frame_boundary(cursor, Fraction(30))
    assert frame_boundary(Fraction(1, 60), Fraction(30)) == 1


def test_still_audio_and_vfr_have_distinct_time_coordinates():
    assert StillHold(Fraction(3, 2)).duration == Fraction(3, 2)
    assert AudioRange(48_000, 72_000, 48_000).duration == Fraction(1, 2)
    timestamps = (Fraction(0), Fraction(1, 50), Fraction(7, 100), Fraction(1, 10))
    video = VideoRange(1, 3, 30, timestamps)
    assert video.start == Fraction(1, 50)
    assert video.duration == Fraction(2, 25)
    with pytest.raises(ValueError, match="presentation"):
        VideoRange(1, 4, 30, timestamps)


@pytest.mark.parametrize("rate", [0, -1, True, float("nan"), float("inf")])
def test_invalid_rates_are_rejected(rate):
    with pytest.raises(ValueError):
        frame_rate(rate)


def test_timeline_trim_delta_maps_to_source_rate_or_verified_timestamps():
    # Moving one timeline second trims 24 source frames, not 30.
    position = source_frame_time(240, Fraction(24)) + Fraction(30, 30)
    assert nearest_source_frame(position, Fraction(24)) == 264
    times = ("0", "1/50", "7/100", "1/10")
    assert nearest_source_frame(Fraction(3, 50), Fraction(30), times) == 2
    assert source_frame_time(2, Fraction(30), times) == Fraction(7, 100)

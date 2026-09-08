"""Exact media coordinates. All ranges have exclusive ends.

Source video frames, audio samples, still durations, and timeline seconds are
distinct types. Quantize shared timeline boundaries with nearest-frame rounding
(ties toward the later frame), never by accumulating rounded clip durations.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from bisect import bisect_left


def rational(value: Fraction | int | float | str) -> Fraction:
    """Read persisted rational seconds or recover a rate supplied as a float."""
    if isinstance(value, bool):
        raise ValueError("Boolean is not a media time")
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("Media time must be finite")
        return Fraction(value).limit_denominator(1_000_000)
    return Fraction(value)


def frame_rate(value: Fraction | int | float | str) -> Fraction:
    rate = rational(value)
    if rate <= 0:
        raise ValueError("Frame rate must be positive")
    return rate


def frame_boundary(seconds: Fraction, rate: Fraction) -> int:
    """Round one nonnegative timeline boundary, ties toward the later frame."""
    seconds, rate = rational(seconds), frame_rate(rate)
    if seconds < 0:
        raise ValueError("Timeline position must be nonnegative")
    value = seconds * rate + Fraction(1, 2)
    return value.numerator // value.denominator


def source_frame_time(index: int, rate: Fraction, timestamps: tuple[str, ...] | None = None) -> Fraction:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("Source frame must be a nonnegative integer")
    return rational(timestamps[index]) if timestamps is not None else Fraction(index) / frame_rate(rate)


def nearest_source_frame(
    seconds: Fraction, rate: Fraction, timestamps: tuple[str, ...] | None = None,
) -> int:
    """Choose a source boundary, using verified timestamps for variable rate."""
    seconds = max(Fraction(0), rational(seconds))
    if timestamps is None:
        return frame_boundary(seconds, rate)
    if not timestamps:
        raise ValueError("Presentation timestamp mapping is empty")
    index = bisect_left(timestamps, seconds, key=rational)
    if index == 0:
        return 0
    if index == len(timestamps):
        return index - 1
    return index if rational(timestamps[index]) - seconds <= seconds - rational(timestamps[index - 1]) else index - 1


def _indices(start: int, end: int) -> None:
    if (
        isinstance(start, bool) or isinstance(end, bool)
        or not isinstance(start, int) or not isinstance(end, int)
        or start < 0 or end <= start
    ):
        raise ValueError("Source range must contain nonnegative integer indices")


@dataclass(frozen=True)
class TimelineRange:
    start: Fraction
    end: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", rational(self.start))
        object.__setattr__(self, "end", rational(self.end))
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Timeline range must have positive duration")

    @property
    def duration(self) -> Fraction:
        return self.end - self.start

    def frames(self, rate: Fraction) -> tuple[int, int]:
        return frame_boundary(self.start, rate), frame_boundary(self.end, rate)

    def to_dict(self) -> dict[str, str]:
        return {"start": str(self.start), "end": str(self.end)}


@dataclass(frozen=True)
class VideoRange:
    start_frame: int
    end_frame: int
    rate: Fraction
    # VFR uses verified presentation boundaries, including the final end.
    # None means verified constant-rate mapping, not an inferred VFR rate.
    timestamps: tuple[Fraction, ...] | None = None
    presentation_range: tuple[Fraction, Fraction] | None = None

    def __post_init__(self) -> None:
        _indices(self.start_frame, self.end_frame)
        object.__setattr__(self, "rate", frame_rate(self.rate))
        if self.timestamps is not None:
            times = tuple(rational(t) for t in self.timestamps)
            if (
                len(times) <= self.end_frame or times[0] < 0
                or any(b <= a for a, b in zip(times, times[1:]))
            ):
                raise ValueError("VFR requires ordered presentation boundaries")
            object.__setattr__(self, "timestamps", times)
        if self.presentation_range is not None:
            begin, end = (rational(t) for t in self.presentation_range)
            if begin < 0 or end <= begin:
                raise ValueError("Presentation range must have positive duration")
            if self.timestamps is not None and (
                begin != self.timestamps[self.start_frame] or end != self.timestamps[self.end_frame]
            ):
                raise ValueError("Presentation range differs from verified timestamps")
            object.__setattr__(self, "presentation_range", (begin, end))

    @property
    def start(self) -> Fraction:
        if self.presentation_range is not None:
            return self.presentation_range[0]
        return (
            self.timestamps[self.start_frame] if self.timestamps is not None
            else Fraction(self.start_frame, 1) / self.rate
        )

    @property
    def end(self) -> Fraction:
        if self.presentation_range is not None:
            return self.presentation_range[1]
        return (
            self.timestamps[self.end_frame] if self.timestamps is not None
            else Fraction(self.end_frame, 1) / self.rate
        )

    @property
    def duration(self) -> Fraction:
        return self.end - self.start


@dataclass(frozen=True)
class StillHold:
    duration: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration", rational(self.duration))
        if self.duration <= 0:
            raise ValueError("Still hold must have positive duration")


@dataclass(frozen=True)
class AudioRange:
    start_sample: int
    end_sample: int
    sample_rate: int

    def __post_init__(self) -> None:
        _indices(self.start_sample, self.end_sample)
        if (
            isinstance(self.sample_rate, bool)
            or not isinstance(self.sample_rate, int) or self.sample_rate <= 0
        ):
            raise ValueError("Audio sample rate must be a positive integer")

    @property
    def start(self) -> Fraction:
        return Fraction(self.start_sample, self.sample_rate)

    @property
    def end(self) -> Fraction:
        return Fraction(self.end_sample, self.sample_rate)

    @property
    def duration(self) -> Fraction:
        return self.end - self.start

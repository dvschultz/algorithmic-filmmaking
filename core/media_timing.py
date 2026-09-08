"""Verify source presentation timing in background media operations."""

from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import time
from typing import Callable

from core.binary_resolver import find_binary, get_subprocess_kwargs
from models.media_time import frame_rate, rational


@dataclass(frozen=True)
class VerifiedVideoTiming:
    rate: Fraction
    boundaries: tuple[Fraction, ...]
    variable: bool
    origin: Fraction = Fraction(0)

    @property
    def frame_count(self) -> int:
        return len(self.boundaries) - 1


def probe_video_timing(path: Path, cancel_check: Callable[[], bool] | None = None) -> VerifiedVideoTiming:
    """Decode frame metadata, including the exclusive presentation end.

    A nominal frame-rate tag alone cannot establish constant-rate timing.
    Call this on a worker, never while handling a GUI interaction.
    """
    ffprobe = find_binary("ffprobe")
    if not ffprobe:
        raise ValueError("FFprobe is required to verify video timing")
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_streams",
        "-show_entries", "frame=best_effort_timestamp,duration,pkt_duration:stream=time_base,r_frame_rate,duration_ts,start_pts",
        "-of", "json", str(path),
    ]
    if cancel_check is not None and cancel_check():
        raise ValueError("Video timing verification cancelled")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **get_subprocess_kwargs())
    deadline = time.monotonic() + 300
    try:
        while True:
            if cancel_check is not None and cancel_check():
                raise ValueError("Video timing verification cancelled")
            if time.monotonic() >= deadline:
                raise ValueError("Video timing verification timed out")
            try:
                output, error = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                pass
        if process.returncode:
            raise ValueError(f"Cannot verify video timing for {path}: {error[-500:]}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
    return timing_from_probe(json.loads(output))


def timing_from_probe(data: dict) -> VerifiedVideoTiming:
    """Validate and normalize decoded PTS without guessing a VFR frame end."""
    streams, frames = data.get("streams", []), data.get("frames", [])
    if not streams or not frames:
        raise ValueError("Video contains no decodable frames")
    try:
        stream = streams[0]
        tick = rational(stream["time_base"])
        rate = frame_rate(stream["r_frame_rate"])
        times = tuple(int(frame["best_effort_timestamp"]) * tick for frame in frames)
        if tick <= 0 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Video presentation timestamps are not strictly ordered")
        last_duration = int(frames[-1].get("duration", frames[-1].get("pkt_duration", 0))) * tick
        if last_duration > 0:
            end = times[-1] + last_duration
        else:
            end = (int(stream["start_pts"]) + int(stream["duration_ts"])) * tick
        if end <= times[-1]:
            raise ValueError("Video has no verified exclusive presentation end")
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        raise ValueError("Video timing metadata is incomplete") from exc
    boundaries = tuple(value - times[0] for value in (*times, end))
    # Fine timebases can round a constant-rate timestamp by one tick. A
    # frame-sized tick must not hide an entire dropped frame as rounding noise.
    tolerance = min(tick, Fraction(1, 2) / rate)
    variable = any(abs(value - Fraction(index) / rate) > tolerance for index, value in enumerate(boundaries))
    return VerifiedVideoTiming(rate, boundaries, variable, times[0])

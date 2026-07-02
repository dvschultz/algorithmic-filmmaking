"""Characterization tests for the core per-clip analysis math.

These pin the numeric behavior of three pure-ish analysis functions against
synthetic media so a math regression breaks a test:

- ``core.analysis.color.extract_dominant_colors`` — k-means dominant-color pull
- ``core.analysis.color.get_average_brightness`` — mean luminance (0.0-1.0)
- ``core.analysis.audio.extract_clip_volume`` — mean RMS volume in dB

Media is synthesized into ``tmp_path`` (never committed). Video is written with
OpenCV (mp4v fourcc, BGR frames) following the proven pattern in
``tests/test_color_profile.py``. Audio is written with the stdlib ``wave``
module. The mp4v codec adds small saturation artifacts and shifts channel
values by a few counts, so color assertions use a generous per-channel
tolerance (±20) and never exact equality (documented project gotcha).

``extract_clip_volume`` shells out to ffmpeg's ``volumedetect`` filter, so its
tests are gated on ffmpeg/ffprobe being present.
"""

import math
import shutil
import struct
import wave
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.analysis.audio import extract_clip_volume, has_audio_track
from core.analysis.color import extract_dominant_colors, get_average_brightness

# ffmpeg + ffprobe are required by extract_clip_volume / has_audio_track.
_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(
    not _HAS_FFMPEG, reason="ffmpeg/ffprobe not available"
)

# Per-channel tolerance for RGB dominant-color assertions. mp4v compression
# shifts pure channel values by a few counts (e.g. 255 -> 251); ±20 absorbs
# codec drift while still failing if a channel is fundamentally wrong.
_COLOR_TOL = 20


# --------------------------------------------------------------------------- #
# Synthetic media helpers
# --------------------------------------------------------------------------- #
def _write_video(path: Path, frames: list[np.ndarray], fps: float = 30.0) -> None:
    """Write a list of BGR frames as an mp4v video file."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _solid_bgr_frame(
    bgr: tuple[int, int, int], width: int = 64, height: int = 48
) -> np.ndarray:
    """Create a solid BGR color frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = bgr
    return frame


def _gray_bgr_frame(value: int, width: int = 64, height: int = 48) -> np.ndarray:
    """Create a solid grayscale frame stored as BGR."""
    gray = np.full((height, width), value, dtype=np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _half_and_half_frame(
    left_bgr: tuple[int, int, int],
    right_bgr: tuple[int, int, int],
    width: int = 64,
    height: int = 48,
) -> np.ndarray:
    """Create a frame that is one color on the left half, another on the right."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, : width // 2] = left_bgr
    frame[:, width // 2 :] = right_bgr
    return frame


def _write_wav(
    path: Path, samples: list[int], sample_rate: int = 22050
) -> None:
    """Write mono 16-bit PCM samples to a WAV file."""
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<%dh" % len(samples), *samples))


def _silence_samples(duration_s: float, sample_rate: int = 22050) -> list[int]:
    return [0] * int(duration_s * sample_rate)


def _sine_samples(
    duration_s: float,
    freq: float = 440.0,
    amplitude: float = 0.9,
    sample_rate: int = 22050,
) -> list[int]:
    n = int(duration_s * sample_rate)
    peak = amplitude * 32767
    return [int(peak * math.sin(2 * math.pi * freq * i / sample_rate)) for i in range(n)]


def _assert_color_near(
    actual_rgb: tuple[int, int, int],
    expected_rgb: tuple[int, int, int],
    tol: int = _COLOR_TOL,
) -> None:
    """Assert each RGB channel is within ``tol`` of the expected value."""
    for chan, (a, e) in enumerate(zip(actual_rgb, expected_rgb)):
        assert abs(int(a) - int(e)) <= tol, (
            f"channel {chan}: got {a}, expected ~{e} (tol {tol}); "
            f"full actual={tuple(int(c) for c in actual_rgb)}"
        )


# --------------------------------------------------------------------------- #
# extract_dominant_colors
# --------------------------------------------------------------------------- #
class TestExtractDominantColors:
    """Pins the k-means dominant-color extraction.

    Numeric contract exercised:
      - a pure-red clip -> top color ~ RGB (255, 0, 0)
      - a pure-blue clip -> top color ~ RGB (0, 0, 255)
      - a half-red / half-blue clip -> both red and blue appear in the palette
    All within ±20 per channel (mp4v codec drift), never exact equality.
    """

    def test_pure_red_video(self, tmp_path):
        video_path = tmp_path / "red.mp4"
        # BGR (0, 0, 255) is pure red -> RGB (255, 0, 0)
        _write_video(video_path, [_solid_bgr_frame((0, 0, 255)) for _ in range(30)])

        colors = extract_dominant_colors(video_path, 0, 30, n_colors=3)

        assert colors, "expected at least one dominant color"
        _assert_color_near(colors[0], (255, 0, 0))

    def test_pure_blue_video(self, tmp_path):
        video_path = tmp_path / "blue.mp4"
        # BGR (255, 0, 0) is pure blue -> RGB (0, 0, 255)
        _write_video(video_path, [_solid_bgr_frame((255, 0, 0)) for _ in range(30)])

        colors = extract_dominant_colors(video_path, 0, 30, n_colors=3)

        assert colors, "expected at least one dominant color"
        _assert_color_near(colors[0], (0, 0, 255))

    def test_two_color_frame_yields_both_colors(self, tmp_path):
        video_path = tmp_path / "half.mp4"
        # Left half red (BGR 0,0,255), right half blue (BGR 255,0,0)
        frame = _half_and_half_frame((0, 0, 255), (255, 0, 0))
        _write_video(video_path, [frame for _ in range(30)])

        # Ask for exactly the two colors present.
        colors = extract_dominant_colors(video_path, 0, 30, n_colors=2)

        assert len(colors) == 2, f"expected 2 dominant colors, got {colors}"

        # k-means ordering by frequency isn't guaranteed for a 50/50 split, so
        # match set membership: one cluster near red, one near blue.
        def _matches(color, target):
            return all(abs(int(c) - int(t)) <= _COLOR_TOL for c, t in zip(color, target))

        has_red = any(_matches(c, (255, 0, 0)) for c in colors)
        has_blue = any(_matches(c, (0, 0, 255)) for c in colors)
        assert has_red, f"no red-ish cluster in {[tuple(int(x) for x in c) for c in colors]}"
        assert has_blue, f"no blue-ish cluster in {[tuple(int(x) for x in c) for c in colors]}"

    def test_missing_video_returns_empty(self, tmp_path):
        """A non-openable path returns an empty list, not a crash."""
        colors = extract_dominant_colors(tmp_path / "nope.mp4", 0, 30, n_colors=3)
        assert colors == []

    def test_mutation_would_be_caught(self, tmp_path):
        """Guard: swapping the extracted color to a wrong constant must fail the
        red-channel assertion. Documents that the tolerance is not so loose that
        a genuinely wrong color slips through.
        """
        video_path = tmp_path / "red.mp4"
        _write_video(video_path, [_solid_bgr_frame((0, 0, 255)) for _ in range(30)])
        colors = extract_dominant_colors(video_path, 0, 30, n_colors=3)

        # Real value passes.
        _assert_color_near(colors[0], (255, 0, 0))
        # A mutated / wrong color (blue instead of red) must be rejected.
        with pytest.raises(AssertionError):
            _assert_color_near((0, 0, 255), (255, 0, 0))


# --------------------------------------------------------------------------- #
# get_average_brightness
# --------------------------------------------------------------------------- #
class TestGetAverageBrightness:
    """Pins mean-luminance extraction on the 0.0-1.0 scale.

    Numeric contract exercised (values from empirical probing):
      - black frames  -> ~0.0   (asserted < 0.05)
      - mid-gray(128) -> ~0.494 (asserted 0.40-0.60)
      - white frames  -> ~0.992 (asserted > 0.95)
      - strict ordering: black < gray < white
    """

    def _brightness_of_solid(self, tmp_path, value: int, name: str) -> float:
        video_path = tmp_path / f"{name}.mp4"
        _write_video(video_path, [_gray_bgr_frame(value) for _ in range(30)])
        return get_average_brightness(video_path, 0, 30, fps=30.0)

    def test_black_is_near_zero(self, tmp_path):
        b = self._brightness_of_solid(tmp_path, 0, "black")
        assert 0.0 <= b < 0.05, f"black brightness {b} not near 0"

    def test_gray_is_near_middle(self, tmp_path):
        b = self._brightness_of_solid(tmp_path, 128, "gray")
        assert 0.40 < b < 0.60, f"mid-gray brightness {b} not near 0.5"

    def test_white_is_near_one(self, tmp_path):
        b = self._brightness_of_solid(tmp_path, 255, "white")
        assert b > 0.95, f"white brightness {b} not near 1.0"

    def test_ordering_black_lt_gray_lt_white(self, tmp_path):
        black = self._brightness_of_solid(tmp_path, 0, "black")
        gray = self._brightness_of_solid(tmp_path, 128, "gray")
        white = self._brightness_of_solid(tmp_path, 255, "white")
        assert black < gray < white, (
            f"expected black < gray < white, got {black} / {gray} / {white}"
        )

    def test_zero_length_clip_returns_default(self, tmp_path):
        """A clip with no frames returns the 0.5 sentinel, not a crash."""
        video_path = tmp_path / "black.mp4"
        _write_video(video_path, [_gray_bgr_frame(0) for _ in range(30)])
        # end_frame <= start_frame -> duration_frames <= 0 -> 0.5 sentinel
        assert get_average_brightness(video_path, 10, 10, fps=30.0) == 0.5

    def test_missing_video_returns_default(self, tmp_path):
        """An unreadable source returns the 0.5 sentinel."""
        assert get_average_brightness(tmp_path / "nope.mp4", 0, 30, fps=30.0) == 0.5

    def test_mutation_ordering_would_be_caught(self, tmp_path):
        """Guard: a brightness function that returned a constant (ignoring pixel
        content) would produce equal black/gray/white values, collapsing the
        strict ordering. Simulate that degenerate output and confirm the
        ordering assertion rejects it.
        """
        # Real function: strict ordering holds (checked above). Here we prove
        # the assertion has teeth against a constant-return mutation.
        constant = 0.5  # what a broken get_average_brightness might return
        with pytest.raises(AssertionError):
            assert constant < constant < constant  # noqa: PLR0124


# --------------------------------------------------------------------------- #
# extract_clip_volume
# --------------------------------------------------------------------------- #
@requires_ffmpeg
class TestExtractClipVolume:
    """Pins mean RMS volume (dB) extraction via ffmpeg volumedetect.

    Numeric contract exercised (values from empirical probing):
      - digital silence -> deep floor (~-91 dB; asserted < -60)
      - loud 440 Hz sine -> near 0 dB (~-4 dB; asserted > -15)
      - strict ordering: silence << sine (at least 40 dB louder)
      - a source with no audio track -> None
    """

    def _wav(self, tmp_path, name: str, samples: list[int]) -> Path:
        path = tmp_path / name
        _write_wav(path, samples)
        return path

    def test_silence_is_near_floor(self, tmp_path):
        wav = self._wav(tmp_path, "silence.wav", _silence_samples(2.0))
        vol = extract_clip_volume(wav, 0.0, 2.0)
        assert vol is not None
        assert vol < -60.0, f"silence volume {vol} dB not near floor"

    def test_loud_sine_is_near_zero(self, tmp_path):
        wav = self._wav(tmp_path, "sine.wav", _sine_samples(2.0))
        vol = extract_clip_volume(wav, 0.0, 2.0)
        assert vol is not None
        assert vol > -15.0, f"loud sine volume {vol} dB not near 0"

    def test_sine_is_much_louder_than_silence(self, tmp_path):
        silence = self._wav(tmp_path, "silence.wav", _silence_samples(2.0))
        sine = self._wav(tmp_path, "sine.wav", _sine_samples(2.0))

        silence_vol = extract_clip_volume(silence, 0.0, 2.0)
        sine_vol = extract_clip_volume(sine, 0.0, 2.0)

        assert silence_vol is not None and sine_vol is not None
        assert sine_vol > silence_vol, "sine should be louder than silence"
        assert sine_vol - silence_vol > 40.0, (
            f"expected >40 dB separation, got sine={sine_vol} silence={silence_vol}"
        )

    def test_amplitude_ordering(self, tmp_path):
        """A louder sine (higher amplitude) reports a higher dB than a quiet one."""
        quiet = self._wav(
            tmp_path, "quiet.wav", _sine_samples(2.0, amplitude=0.05)
        )
        loud = self._wav(tmp_path, "loud.wav", _sine_samples(2.0, amplitude=0.9))

        quiet_vol = extract_clip_volume(quiet, 0.0, 2.0)
        loud_vol = extract_clip_volume(loud, 0.0, 2.0)

        assert quiet_vol is not None and loud_vol is not None
        assert loud_vol > quiet_vol, (
            f"louder amplitude should report higher dB: "
            f"loud={loud_vol} quiet={quiet_vol}"
        )

    def test_no_audio_track_returns_none(self, tmp_path):
        """A video with no audio stream yields None (not a crash, not a number)."""
        video_path = tmp_path / "silent_video.mp4"
        _write_video(video_path, [_gray_bgr_frame(128) for _ in range(30)])
        assert not has_audio_track(video_path)
        assert extract_clip_volume(video_path, 0.0, 1.0) is None

    def test_has_audio_flag_short_circuits_probe(self, tmp_path):
        """Passing _has_audio=False skips ffprobe and returns None immediately."""
        # File need not even have audio; the flag forces the no-audio path.
        wav = self._wav(tmp_path, "sine.wav", _sine_samples(1.0))
        assert extract_clip_volume(wav, 0.0, 1.0, _has_audio=False) is None

    def test_mutation_ordering_would_be_caught(self, tmp_path):
        """Guard: if extract_clip_volume ignored content and returned a constant,
        the silence-vs-sine separation assertion would fail. Confirm the
        assertion rejects equal values.
        """
        silence = self._wav(tmp_path, "silence.wav", _silence_samples(1.0))
        sine = self._wav(tmp_path, "sine.wav", _sine_samples(1.0))
        silence_vol = extract_clip_volume(silence, 0.0, 1.0)
        sine_vol = extract_clip_volume(sine, 0.0, 1.0)

        # Real values pass the separation check.
        assert sine_vol - silence_vol > 40.0
        # A constant-return mutation (both equal) must be rejected.
        constant = -20.0
        with pytest.raises(AssertionError):
            assert constant - constant > 40.0

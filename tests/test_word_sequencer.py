"""Tests for ``core/remix/word_sequencer.py``.

Covers frame-math conventions (floor on in, ceil on out, handle-frame
clamping), validation of word-data presence, and zero-duration word
dropping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from core.remix.word_sequencer import MissingWordDataError, generate_word_sequence
from core.transcription import TranscriptSegment, WordTimestamp


@dataclass
class MockSource:
    id: str = "src1"
    fps: float = 24.0


@dataclass
class MockClip:
    id: str = "clip1"
    source_id: str = "src1"
    start_frame: int = 0
    end_frame: int = 240  # 10 s at 24 fps
    transcript: Optional[list[TranscriptSegment]] = None


def _seg(words: list[tuple[str, float, float]], language: str = "en") -> TranscriptSegment:
    word_list = [WordTimestamp(start=s, end=e, text=t) for (t, s, e) in words]
    return TranscriptSegment(
        start_time=words[0][1] if words else 0.0,
        end_time=words[-1][2] if words else 0.0,
        text=" ".join(t for t, _, _ in words),
        confidence=1.0,
        words=word_list,
        language=language,
    )


# ---------------------------------------------------------------------------
# Happy-path framing
# ---------------------------------------------------------------------------


class TestAlphabeticalSequence:
    def test_basic_alphabetical_sequence(self):
        clip = MockClip(
            transcript=[
                _seg([("zoo", 0.0, 0.5), ("apple", 0.6, 1.0)])
            ]
        )
        result = generate_word_sequence(
            clips=[(clip, MockSource())],
            mode="alphabetical",
            mode_params={},
            handle_frames=0,
        )
        # Two words, alphabetical order: apple → zoo
        assert len(result) == 2
        assert result[0].source_clip_id == clip.id
        assert result[0].source_id == clip.source_id

    def test_in_point_floor_out_point_ceil(self):
        # Word at 0.5–0.9 s @ 24 fps → in_point=floor(0.5*24)=12, out_point=ceil(0.9*24)=ceil(21.6)=22
        clip = MockClip(
            transcript=[_seg([("hello", 0.5, 0.9)])]
        )
        result = generate_word_sequence(
            clips=[(clip, MockSource(fps=24.0))],
            mode="alphabetical",
            mode_params={},
            handle_frames=0,
        )
        assert len(result) == 1
        assert result[0].in_point == 12
        assert result[0].out_point == 22


# ---------------------------------------------------------------------------
# Handle-frame padding & clamping
# ---------------------------------------------------------------------------


class TestHandleFrameClamping:
    def test_handle_clamps_at_clip_start(self):
        # Word starts at 0.0 with handle=2 → would push in_point negative; clamps to 0.
        clip = MockClip(
            start_frame=0,
            end_frame=240,
            transcript=[_seg([("hi", 0.0, 0.4)])],
        )
        result = generate_word_sequence(
            clips=[(clip, MockSource(fps=24.0))],
            mode="alphabetical",
            mode_params={},
            handle_frames=2,
        )
        assert result[0].in_point == 0

    def test_handle_clamps_at_clip_end(self):
        # Clip duration = 240 frames @ 24fps = 10 s.  Word at 9.9–10.0 with handle=4 → end clamps to 240.
        clip = MockClip(
            start_frame=0,
            end_frame=240,
            transcript=[_seg([("end", 9.9, 10.0)])],
        )
        result = generate_word_sequence(
            clips=[(clip, MockSource(fps=24.0))],
            mode="alphabetical",
            mode_params={},
            handle_frames=4,
        )
        assert result[0].out_point == 240

    def test_handle_widens_window(self):
        # Word at 1.0–2.0 with handle_frames=2 (at 24 fps = 0.0833 s):
        # in_seconds  = 1.0  - 2/24  ≈ 0.9167  → in_point  = floor(0.9167*24)  = 22
        # out_seconds = 2.0  + 2/24  ≈ 2.0833  → out_point = ceil(2.0833*24)   = 50
        clip = MockClip(transcript=[_seg([("mid", 1.0, 2.0)])])
        result = generate_word_sequence(
            clips=[(clip, MockSource(fps=24.0))],
            mode="alphabetical",
            mode_params={},
            handle_frames=2,
        )
        assert result[0].in_point == 22
        assert result[0].out_point == 50


# ---------------------------------------------------------------------------
# Zero-duration words
# ---------------------------------------------------------------------------


class TestZeroDurationDropped:
    def test_zero_duration_word_dropped(self):
        clip = MockClip(
            transcript=[
                _seg([("real", 0.0, 0.5), ("ghost", 1.0, 1.0), ("end", 2.0, 2.5)])
            ]
        )
        result = generate_word_sequence(
            clips=[(clip, MockSource())],
            mode="alphabetical",
            mode_params={},
            handle_frames=0,
        )
        # "ghost" (start == end) is dropped.
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestMissingWordData:
    def test_clip_with_none_words_raises(self):
        clip = MockClip(
            transcript=[
                TranscriptSegment(
                    start_time=0.0,
                    end_time=1.0,
                    text="x y",
                    confidence=1.0,
                    words=None,
                    language="en",
                )
            ]
        )
        with pytest.raises(MissingWordDataError) as exc:
            generate_word_sequence(
                clips=[(clip, MockSource())],
                mode="alphabetical",
                mode_params={},
            )
        assert clip.id in exc.value.clip_ids

    def test_aggregates_multiple_missing(self):
        bad_seg = TranscriptSegment(
            start_time=0.0, end_time=1.0, text="x", confidence=1.0, words=None, language="en"
        )
        clip_a = MockClip(id="A", transcript=[bad_seg])
        clip_b = MockClip(id="B", transcript=[bad_seg])
        with pytest.raises(MissingWordDataError) as exc:
            generate_word_sequence(
                clips=[(clip_a, MockSource()), (clip_b, MockSource())],
                mode="alphabetical",
                mode_params={},
            )
        assert "A" in exc.value.clip_ids
        assert "B" in exc.value.clip_ids


class TestMissingFps:
    def test_none_fps_raises(self):
        clip = MockClip(transcript=[_seg([("a", 0.0, 0.5)])])
        src = MockSource(fps=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="fps"):
            generate_word_sequence(
                clips=[(clip, src)],
                mode="alphabetical",
                mode_params={},
            )


# ---------------------------------------------------------------------------
# Mode plumbing
# ---------------------------------------------------------------------------


class TestModeDispatch:
    def test_by_frequency_descending(self):
        clip = MockClip(
            transcript=[
                _seg(
                    [
                        ("the", 0.0, 0.2),
                        ("dog", 0.3, 0.5),
                        ("the", 0.6, 0.8),
                    ]
                )
            ]
        )
        result = generate_word_sequence(
            clips=[(clip, MockSource())],
            mode="by_frequency",
            mode_params={"order": "descending"},
        )
        assert len(result) == 3  # the, the, dog

    def test_by_chosen_words(self):
        clip = MockClip(
            transcript=[
                _seg(
                    [
                        ("never", 0.0, 0.3),
                        ("always", 0.4, 0.7),
                        ("never", 0.8, 1.1),
                    ]
                )
            ]
        )
        result = generate_word_sequence(
            clips=[(clip, MockSource())],
            mode="by_chosen_words",
            mode_params={"include": ["never"]},
        )
        assert len(result) == 2

    def test_from_word_list(self):
        clip = MockClip(
            transcript=[_seg([("the", 0.0, 0.2), ("sky", 0.3, 0.6)])]
        )
        result = generate_word_sequence(
            clips=[(clip, MockSource())],
            mode="from_word_list",
            mode_params={"sequence": ["the", "the", "sky"]},
        )
        assert len(result) == 3

    def test_by_property(self):
        clip = MockClip(
            transcript=[
                _seg([("a", 0.0, 0.1), ("bb", 0.2, 0.5), ("ccc", 0.6, 0.7)])
            ]
        )
        result = generate_word_sequence(
            clips=[(clip, MockSource())],
            mode="by_property",
            mode_params={"key": "length"},
        )
        assert len(result) == 3

    def test_unknown_mode_raises(self):
        clip = MockClip(transcript=[_seg([("a", 0.0, 0.5)])])
        with pytest.raises(ValueError, match="mode"):
            generate_word_sequence(
                clips=[(clip, MockSource())],
                mode="nope",
                mode_params={},
            )


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_clips_returns_empty(self):
        result = generate_word_sequence(
            clips=[],
            mode="alphabetical",
            mode_params={},
        )
        assert result == []


# ---------------------------------------------------------------------------
# Fraction-fps handling (per the cut-tab-fractional-duration learning)
# ---------------------------------------------------------------------------


class TestFractionFps:
    def test_fraction_fps_converted_to_float(self):
        from fractions import Fraction

        clip = MockClip(transcript=[_seg([("hi", 0.5, 0.9)])])
        src = MockSource(fps=Fraction(24000, 1001))  # type: ignore[arg-type]
        result = generate_word_sequence(
            clips=[(clip, src)],
            mode="alphabetical",
            mode_params={},
        )
        assert len(result) == 1
        # Frame numbers are ints.
        assert isinstance(result[0].in_point, int)
        assert isinstance(result[0].out_point, int)

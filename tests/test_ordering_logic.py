"""Tests for the shared order_matched_clips function from core/analysis/faces.py.

Tests cover:
- All 6 ordering strategies (original, duration, color, brightness, confidence, random)
- Both display-name and agent-key variants produce identical results
- Edge cases: empty list, single clip
- Color ordering with empty dominant_colors (should not crash)
- Brightness ordering with None values (falls back to 0.5)
"""

from pathlib import Path

import pytest

from core.analysis.faces import order_matched_clips
from models.clip import Clip, Source


# -- Helpers ------------------------------------------------------------------


def _make_source(
    source_id: str = "src1",
    fps: float = 30.0,
    file_path: str = "/videos/video_a.mp4",
) -> Source:
    return Source(id=source_id, file_path=Path(file_path), fps=fps)


def _make_clip(
    clip_id: str,
    source_id: str = "src1",
    start_frame: int = 0,
    end_frame: int = 90,
    **kwargs,
) -> Clip:
    return Clip(
        id=clip_id,
        source_id=source_id,
        start_frame=start_frame,
        end_frame=end_frame,
        **kwargs,
    )


# -- Original Order -----------------------------------------------------------


class TestOriginalOrder:
    """Clips sorted by (source.file_path.name, clip.start_frame)."""

    def test_sorts_by_source_name_then_start_frame(self):
        src_a = _make_source("s1", file_path="/videos/alpha.mp4")
        src_b = _make_source("s2", file_path="/videos/beta.mp4")

        c1 = _make_clip("c1", source_id="s2", start_frame=100, end_frame=200)
        c2 = _make_clip("c2", source_id="s1", start_frame=50, end_frame=100)
        c3 = _make_clip("c3", source_id="s1", start_frame=0, end_frame=50)

        matched = [(c1, src_b, 0.9), (c2, src_a, 0.8), (c3, src_a, 0.7)]
        result = order_matched_clips(matched, "original")

        ids = [clip.id for clip, _ in result]
        # alpha.mp4 clips first (start_frame 0, then 50), then beta.mp4
        assert ids == ["c3", "c2", "c1"]

    def test_display_name_variant(self):
        src = _make_source("s1")
        c1 = _make_clip("c1", start_frame=90, end_frame=120)
        c2 = _make_clip("c2", start_frame=0, end_frame=30)
        c3 = _make_clip("c3", start_frame=30, end_frame=60)

        matched = [(c1, src, 0.9), (c2, src, 0.8), (c3, src, 0.7)]
        result = order_matched_clips(matched, "Original Order")

        ids = [clip.id for clip, _ in result]
        assert ids == ["c2", "c3", "c1"]


# -- By Duration --------------------------------------------------------------


class TestDurationOrder:
    """Clips sorted by duration_seconds ascending."""

    def test_sorts_by_duration(self):
        src = _make_source("s1", fps=30.0)

        c1 = _make_clip("c1", start_frame=0, end_frame=90)    # 3.0s
        c2 = _make_clip("c2", start_frame=0, end_frame=30)    # 1.0s
        c3 = _make_clip("c3", start_frame=0, end_frame=150)   # 5.0s

        matched = [(c1, src, 0.9), (c2, src, 0.8), (c3, src, 0.7)]
        result = order_matched_clips(matched, "duration")

        ids = [clip.id for clip, _ in result]
        assert ids == ["c2", "c1", "c3"]

    def test_display_name_matches_agent_key(self):
        src = _make_source("s1", fps=30.0)
        c1 = _make_clip("c1", start_frame=0, end_frame=90)
        c2 = _make_clip("c2", start_frame=0, end_frame=30)
        c3 = _make_clip("c3", start_frame=0, end_frame=150)

        matched_a = [(c1, src, 0.9), (c2, src, 0.8), (c3, src, 0.7)]
        matched_b = [(c1, src, 0.9), (c2, src, 0.8), (c3, src, 0.7)]

        result_agent = order_matched_clips(matched_a, "duration")
        result_display = order_matched_clips(matched_b, "By Duration")

        assert [c.id for c, _ in result_agent] == [c.id for c, _ in result_display]


# -- By Color -----------------------------------------------------------------


class TestColorOrder:
    """Clips sorted by hue of first dominant color."""

    def test_sorts_by_hue(self):
        src = _make_source("s1")

        # Red hue ~0.0, Green hue ~0.33, Blue hue ~0.67
        c_red = _make_clip("red", dominant_colors=[(255, 0, 0)])
        c_green = _make_clip("green", dominant_colors=[(0, 255, 0)])
        c_blue = _make_clip("blue", dominant_colors=[(0, 0, 255)])

        matched = [(c_blue, src, 0.9), (c_red, src, 0.8), (c_green, src, 0.7)]
        result = order_matched_clips(matched, "color")

        ids = [clip.id for clip, _ in result]
        assert ids == ["red", "green", "blue"]

    def test_empty_dominant_colors_does_not_crash(self):
        """Clips with empty or None dominant_colors should sort without error."""
        src = _make_source("s1")

        c_none = _make_clip("none_colors", dominant_colors=None)
        c_empty = _make_clip("empty_colors", dominant_colors=[])
        c_red = _make_clip("red", dominant_colors=[(255, 0, 0)])

        matched = [
            (c_none, src, 0.9),
            (c_empty, src, 0.8),
            (c_red, src, 0.7),
        ]
        # Should not raise
        result = order_matched_clips(matched, "color")
        assert len(result) == 3

    def test_empty_colors_get_default_hue(self):
        """Clips without colors use 0.5 as fallback hue, sorting between green and blue."""
        src = _make_source("s1")

        # Red hue ~0.0, Green hue ~0.33, Blue hue ~0.67
        c_red = _make_clip("red", dominant_colors=[(255, 0, 0)])
        c_green = _make_clip("green", dominant_colors=[(0, 255, 0)])
        c_blue = _make_clip("blue", dominant_colors=[(0, 0, 255)])
        c_no_color = _make_clip("no_color", dominant_colors=None)

        matched = [
            (c_blue, src, 0.9),
            (c_no_color, src, 0.8),
            (c_red, src, 0.7),
            (c_green, src, 0.6),
        ]
        result = order_matched_clips(matched, "color")

        ids = [clip.id for clip, _ in result]
        # red(~0.0) < green(~0.33) < no_color(0.5) < blue(~0.67)
        assert ids == ["red", "green", "no_color", "blue"]

    def test_display_name_matches_agent_key(self):
        src = _make_source("s1")
        c1 = _make_clip("c1", dominant_colors=[(255, 0, 0)])
        c2 = _make_clip("c2", dominant_colors=[(0, 0, 255)])
        c3 = _make_clip("c3", dominant_colors=[(0, 255, 0)])

        matched_a = [(c1, src, 0.9), (c2, src, 0.8), (c3, src, 0.7)]
        matched_b = [(c1, src, 0.9), (c2, src, 0.8), (c3, src, 0.7)]

        result_agent = order_matched_clips(matched_a, "color")
        result_display = order_matched_clips(matched_b, "By Color")

        assert [c.id for c, _ in result_agent] == [c.id for c, _ in result_display]


# -- By Brightness ------------------------------------------------------------


class TestBrightnessOrder:
    """Clips sorted by average_brightness ascending."""

    def test_sorts_by_brightness(self):
        src = _make_source("s1")

        c_dark = _make_clip("dark", average_brightness=0.1)
        c_mid = _make_clip("mid", average_brightness=0.5)
        c_bright = _make_clip("bright", average_brightness=0.9)

        matched = [(c_bright, src, 0.9), (c_dark, src, 0.8), (c_mid, src, 0.7)]
        result = order_matched_clips(matched, "brightness")

        ids = [clip.id for clip, _ in result]
        assert ids == ["dark", "mid", "bright"]

    def test_none_brightness_defaults_to_half(self):
        """Clips with None average_brightness should sort as 0.5."""
        src = _make_source("s1")

        c_low = _make_clip("low", average_brightness=0.2)
        c_none = _make_clip("none_val", average_brightness=None)
        c_high = _make_clip("high", average_brightness=0.8)

        matched = [(c_high, src, 0.9), (c_none, src, 0.8), (c_low, src, 0.7)]
        result = order_matched_clips(matched, "brightness")

        ids = [clip.id for clip, _ in result]
        # low(0.2) < none_val(0.5) < high(0.8)
        assert ids == ["low", "none_val", "high"]

    def test_all_none_brightness_does_not_crash(self):
        """All clips with None brightness should sort without error."""
        src = _make_source("s1")

        c1 = _make_clip("c1", average_brightness=None)
        c2 = _make_clip("c2", average_brightness=None)
        c3 = _make_clip("c3", average_brightness=None)

        matched = [(c1, src, 0.9), (c2, src, 0.8), (c3, src, 0.7)]
        result = order_matched_clips(matched, "brightness")
        assert len(result) == 3

    def test_display_name_matches_agent_key(self):
        src = _make_source("s1")
        c1 = _make_clip("c1", average_brightness=0.1)
        c2 = _make_clip("c2", average_brightness=0.9)
        c3 = _make_clip("c3", average_brightness=0.5)

        matched_a = [(c1, src, 0.9), (c2, src, 0.8), (c3, src, 0.7)]
        matched_b = [(c1, src, 0.9), (c2, src, 0.8), (c3, src, 0.7)]

        result_agent = order_matched_clips(matched_a, "brightness")
        result_display = order_matched_clips(matched_b, "By Brightness")

        assert [c.id for c, _ in result_agent] == [c.id for c, _ in result_display]


# -- By Confidence ------------------------------------------------------------


class TestConfidenceOrder:
    """Clips sorted by confidence descending."""

    def test_sorts_by_confidence_descending(self):
        src = _make_source("s1")

        c1 = _make_clip("c1")
        c2 = _make_clip("c2")
        c3 = _make_clip("c3")

        matched = [(c1, src, 0.5), (c2, src, 0.95), (c3, src, 0.7)]
        result = order_matched_clips(matched, "confidence")

        ids = [clip.id for clip, _ in result]
        assert ids == ["c2", "c3", "c1"]

    def test_display_name_matches_agent_key(self):
        src = _make_source("s1")
        c1 = _make_clip("c1")
        c2 = _make_clip("c2")
        c3 = _make_clip("c3")

        matched_a = [(c1, src, 0.5), (c2, src, 0.95), (c3, src, 0.7)]
        matched_b = [(c1, src, 0.5), (c2, src, 0.95), (c3, src, 0.7)]

        result_agent = order_matched_clips(matched_a, "confidence")
        result_display = order_matched_clips(matched_b, "By Confidence")

        assert [c.id for c, _ in result_agent] == [c.id for c, _ in result_display]


# -- Random -------------------------------------------------------------------


class TestRandomOrder:
    """Random ordering shuffles and strips confidence."""

    def test_random_returns_all_clips(self):
        src = _make_source("s1")
        c1 = _make_clip("c1")
        c2 = _make_clip("c2")
        c3 = _make_clip("c3")

        matched = [(c1, src, 0.9), (c2, src, 0.8), (c3, src, 0.7)]
        result = order_matched_clips(matched, "random")

        result_ids = sorted(clip.id for clip, _ in result)
        assert result_ids == ["c1", "c2", "c3"]

    def test_random_strips_confidence(self):
        """Result tuples should be (Clip, Source), not (Clip, Source, confidence)."""
        src = _make_source("s1")
        c1 = _make_clip("c1")

        matched = [(c1, src, 0.9)]
        result = order_matched_clips(matched, "random")

        assert len(result) == 1
        assert len(result[0]) == 2  # (Clip, Source), no confidence

    def test_display_name_variant(self):
        src = _make_source("s1")
        c1 = _make_clip("c1")
        c2 = _make_clip("c2")
        c3 = _make_clip("c3")

        matched = [(c1, src, 0.9), (c2, src, 0.8), (c3, src, 0.7)]
        result = order_matched_clips(matched, "Random")

        result_ids = sorted(clip.id for clip, _ in result)
        assert result_ids == ["c1", "c2", "c3"]


# -- Edge Cases ---------------------------------------------------------------


class TestEdgeCases:
    """Empty and single-clip inputs."""

    def test_empty_list_returns_empty(self):
        result = order_matched_clips([], "original")
        assert result == []

    def test_empty_list_all_strategies(self):
        for strategy in ["original", "duration", "color", "brightness", "confidence", "random"]:
            result = order_matched_clips([], strategy)
            assert result == [], f"Failed for strategy: {strategy}"

    def test_single_clip_returns_single(self):
        src = _make_source("s1")
        c1 = _make_clip("c1")

        matched = [(c1, src, 0.9)]

        for strategy in ["original", "duration", "color", "brightness", "confidence", "random"]:
            result = order_matched_clips(matched[:], strategy)
            assert len(result) == 1, f"Failed for strategy: {strategy}"
            assert result[0][0].id == "c1"
            assert result[0][1].id == "s1"

    def test_output_strips_confidence_for_all_strategies(self):
        """All strategies should return (Clip, Source) tuples, not triples."""
        src = _make_source("s1")
        c1 = _make_clip("c1")
        c2 = _make_clip("c2", start_frame=100, end_frame=200,
                         dominant_colors=[(128, 128, 128)],
                         average_brightness=0.5)

        for strategy in ["original", "duration", "color", "brightness", "confidence", "random"]:
            matched = [(c1, src, 0.9), (c2, src, 0.8)]
            result = order_matched_clips(matched, strategy)
            for pair in result:
                assert len(pair) == 2, (
                    f"Strategy '{strategy}' returned tuple of length {len(pair)}, expected 2"
                )


# -- Display Name / Agent Key Parity -----------------------------------------


class TestDisplayNameAgentKeyParity:
    """Both display-name and agent-key variants must produce identical ordering."""

    PAIRS = [
        ("Original Order", "original"),
        ("By Duration", "duration"),
        ("By Color", "color"),
        ("By Brightness", "brightness"),
        ("By Confidence", "confidence"),
    ]

    def test_all_deterministic_pairs_match(self):
        src_a = _make_source("s1", file_path="/videos/alpha.mp4")
        src_b = _make_source("s2", file_path="/videos/beta.mp4")

        c1 = _make_clip("c1", source_id="s1", start_frame=0, end_frame=90,
                         dominant_colors=[(255, 0, 0)], average_brightness=0.3)
        c2 = _make_clip("c2", source_id="s2", start_frame=50, end_frame=200,
                         dominant_colors=[(0, 255, 0)], average_brightness=0.7)
        c3 = _make_clip("c3", source_id="s1", start_frame=100, end_frame=130,
                         dominant_colors=[(0, 0, 255)], average_brightness=0.1)

        sources = {"s1": src_a, "s2": src_b}

        for display_name, agent_key in self.PAIRS:
            matched_a = [(c1, sources[c1.source_id], 0.9),
                         (c2, sources[c2.source_id], 0.5),
                         (c3, sources[c3.source_id], 0.7)]
            matched_b = [(c1, sources[c1.source_id], 0.9),
                         (c2, sources[c2.source_id], 0.5),
                         (c3, sources[c3.source_id], 0.7)]

            result_display = order_matched_clips(matched_a, display_name)
            result_agent = order_matched_clips(matched_b, agent_key)

            ids_display = [clip.id for clip, _ in result_display]
            ids_agent = [clip.id for clip, _ in result_agent]
            assert ids_display == ids_agent, (
                f"Mismatch for {display_name!r} vs {agent_key!r}: "
                f"{ids_display} != {ids_agent}"
            )

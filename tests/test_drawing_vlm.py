"""Tests for VLM interpretation mode in Signature Style drawing sequencer.

Covers:
- VLM JSON response parsing (_parse_vlm_response)
- VLM-to-segment mapping (_map_vlm_to_segment)
- Shot type closest-match logic (_closest_shot_type)
- Adaptive drawing slicing (slice_drawing_adaptive)
"""

import pytest

from core.remix.drawing_vlm import (
    _parse_vlm_response,
    _map_vlm_to_segment,
    _closest_shot_type,
    _equal_slices,
)
from core.remix.drawing_segment import DrawingSegment

# Guard Qt imports for CI environments without PySide6
try:
    from PySide6.QtGui import QImage, QPainter, QColor
    from PySide6.QtCore import Qt

    HAS_QT = True
except ImportError:
    HAS_QT = False

qt_required = pytest.mark.skipif(not HAS_QT, reason="PySide6 not available")


# ──────────────────────────────────────────────────────────────
# 1. VLM JSON parsing (_parse_vlm_response)
# ──────────────────────────────────────────────────────────────


class TestParseVlmResponse:
    """Tests for _parse_vlm_response JSON extraction logic."""

    def test_valid_json_string(self):
        """Pure JSON string parses directly."""
        raw = '{"shot_type": "close-up", "energy": 0.7}'
        result = _parse_vlm_response(raw)
        assert result == {"shot_type": "close-up", "energy": 0.7}

    def test_fenced_json_block(self):
        """JSON inside a fenced ```json block is extracted and parsed."""
        raw = (
            "Here is my analysis:\n"
            "```json\n"
            '{"shot_type": "wide shot", "pacing": "fast"}\n'
            "```\n"
            "That's my interpretation."
        )
        result = _parse_vlm_response(raw)
        assert result == {"shot_type": "wide shot", "pacing": "fast"}

    def test_bare_json_in_surrounding_text(self):
        """A bare JSON object embedded in prose is found and parsed."""
        raw = (
            "I see vibrant colors and dynamic movement. "
            '{"shot_type": "medium shot", "energy": 0.5, "brightness": "bright"} '
            "The composition suggests energy."
        )
        result = _parse_vlm_response(raw)
        assert result == {
            "shot_type": "medium shot",
            "energy": 0.5,
            "brightness": "bright",
        }

    def test_completely_invalid_text(self):
        """Text with no JSON at all returns empty dict."""
        raw = "This is just a plain text response with no JSON whatsoever."
        result = _parse_vlm_response(raw)
        assert result == {}

    def test_empty_string(self):
        """Empty string returns empty dict."""
        result = _parse_vlm_response("")
        assert result == {}

    def test_none_like_empty(self):
        """None input returns empty dict (guarded by 'if not response_text')."""
        result = _parse_vlm_response(None)
        assert result == {}

    def test_json_with_extra_whitespace(self):
        """JSON with leading/trailing whitespace and newlines parses fine."""
        raw = '  \n\n  {"shot_type": "full shot", "energy": 0.3}  \n  '
        result = _parse_vlm_response(raw)
        assert result == {"shot_type": "full shot", "energy": 0.3}

    def test_fenced_block_with_extra_whitespace(self):
        """Fenced block with whitespace around the JSON."""
        raw = (
            "Analysis:\n"
            "```json\n"
            "  \n"
            '  {"pacing": "slow", "color_mood": "cool"}  \n'
            "  \n"
            "```"
        )
        result = _parse_vlm_response(raw)
        assert result == {"pacing": "slow", "color_mood": "cool"}

    def test_json_array_returns_empty(self):
        """A JSON array (not dict) returns empty dict."""
        raw = '[1, 2, 3]'
        result = _parse_vlm_response(raw)
        assert result == {}

    def test_nested_json_picks_outer(self):
        """When multiple JSON-like patterns exist, the first valid one wins."""
        # Direct parse of the full string will fail (it's not pure JSON),
        # fenced block doesn't exist, so bare object regex finds the first {...}
        raw = 'Some text {"a": 1} and more text {"b": 2}'
        result = _parse_vlm_response(raw)
        assert result == {"a": 1}

    def test_fenced_block_preferred_over_bare(self):
        """Fenced block is tried before bare JSON extraction."""
        raw = (
            '{"wrong": true}\n'
            "```json\n"
            '{"correct": true}\n'
            "```"
        )
        # Direct parse of the full string fails (has extra text after first JSON),
        # so fenced block is tried next and wins.
        # Actually, the full string '{"wrong": true}\n```json...' won't parse as
        # valid JSON, so step 1 fails, step 2 finds the fenced block.
        result = _parse_vlm_response(raw)
        assert result.get("correct") is True


# ──────────────────────────────────────────────────────────────
# 2. Segment mapping (_map_vlm_to_segment)
# ──────────────────────────────────────────────────────────────


class TestMapVlmToSegment:
    """Tests for _map_vlm_to_segment VLM data to DrawingSegment conversion."""

    def test_valid_dict_all_fields(self):
        """VLM dict with all fields produces correct DrawingSegment."""
        vlm = {
            "shot_type": "close-up",
            "color_mood": "warm",
            "energy": 0.75,
            "pacing": "fast",
            "brightness": "bright",
        }
        seg = _map_vlm_to_segment(
            vlm, x_start=0, x_end=100,
            total_duration_seconds=60.0, canvas_width=200,
        )
        assert isinstance(seg, DrawingSegment)
        assert seg.x_start == 0
        assert seg.x_end == 100
        assert seg.shot_type == "close-up"
        assert seg.color_mood == "warm"
        assert seg.energy == 0.75
        assert seg.target_pacing == 0.8  # "fast" -> 0.8
        assert seg.brightness == 0.8  # "bright" -> 0.8
        # Duration: 60 * (100/200) = 30.0
        assert seg.target_duration_seconds == pytest.approx(30.0)

    def test_string_pacing_fast(self):
        """String pacing "fast" maps to 0.8."""
        seg = _map_vlm_to_segment(
            {"pacing": "fast"}, 0, 100, 10.0, 100,
        )
        assert seg.target_pacing == pytest.approx(0.8)

    def test_string_pacing_medium(self):
        """String pacing "medium" maps to 0.5."""
        seg = _map_vlm_to_segment(
            {"pacing": "medium"}, 0, 100, 10.0, 100,
        )
        assert seg.target_pacing == pytest.approx(0.5)

    def test_string_pacing_slow(self):
        """String pacing "slow" maps to 0.2."""
        seg = _map_vlm_to_segment(
            {"pacing": "slow"}, 0, 100, 10.0, 100,
        )
        assert seg.target_pacing == pytest.approx(0.2)

    def test_string_pacing_unknown_defaults(self):
        """Unknown pacing string defaults to 0.5."""
        seg = _map_vlm_to_segment(
            {"pacing": "hyperspeed"}, 0, 100, 10.0, 100,
        )
        assert seg.target_pacing == pytest.approx(0.5)

    def test_string_energy_high(self):
        """String energy "high" uses _PACING_MAP lookup -> 0.5 fallback.

        Note: energy string values also use _PACING_MAP, so "high" is not
        a key there and defaults to 0.5.  Only "fast"/"medium"/"slow" are
        valid string energy values that map to 0.8/0.5/0.2.
        """
        seg = _map_vlm_to_segment(
            {"energy": "high"}, 0, 100, 10.0, 100,
        )
        # "high" is not in _PACING_MAP, so it defaults to 0.5
        assert seg.energy == pytest.approx(0.5)

    def test_string_energy_fast(self):
        """String energy "fast" maps to 0.8 via _PACING_MAP."""
        seg = _map_vlm_to_segment(
            {"energy": "fast"}, 0, 100, 10.0, 100,
        )
        assert seg.energy == pytest.approx(0.8)

    def test_string_energy_medium(self):
        """String energy "medium" maps to 0.5 via _PACING_MAP."""
        seg = _map_vlm_to_segment(
            {"energy": "medium"}, 0, 100, 10.0, 100,
        )
        assert seg.energy == pytest.approx(0.5)

    def test_string_energy_slow(self):
        """String energy "slow" maps to 0.2 via _PACING_MAP."""
        seg = _map_vlm_to_segment(
            {"energy": "slow"}, 0, 100, 10.0, 100,
        )
        assert seg.energy == pytest.approx(0.2)

    def test_string_brightness_bright(self):
        """String brightness "bright" maps to 0.8."""
        seg = _map_vlm_to_segment(
            {"brightness": "bright"}, 0, 100, 10.0, 100,
        )
        assert seg.brightness == pytest.approx(0.8)

    def test_string_brightness_dark(self):
        """String brightness "dark" maps to 0.2."""
        seg = _map_vlm_to_segment(
            {"brightness": "dark"}, 0, 100, 10.0, 100,
        )
        assert seg.brightness == pytest.approx(0.2)

    def test_string_brightness_medium(self):
        """String brightness "medium" maps to 0.5."""
        seg = _map_vlm_to_segment(
            {"brightness": "medium"}, 0, 100, 10.0, 100,
        )
        assert seg.brightness == pytest.approx(0.5)

    def test_string_brightness_light(self):
        """String brightness "light" maps to 0.8 (alias for bright)."""
        seg = _map_vlm_to_segment(
            {"brightness": "light"}, 0, 100, 10.0, 100,
        )
        assert seg.brightness == pytest.approx(0.8)

    def test_string_brightness_normal(self):
        """String brightness "normal" maps to 0.5 (alias for medium)."""
        seg = _map_vlm_to_segment(
            {"brightness": "normal"}, 0, 100, 10.0, 100,
        )
        assert seg.brightness == pytest.approx(0.5)

    def test_string_brightness_unknown_defaults(self):
        """Unknown brightness string defaults to 0.5."""
        seg = _map_vlm_to_segment(
            {"brightness": "blinding"}, 0, 100, 10.0, 100,
        )
        assert seg.brightness == pytest.approx(0.5)

    def test_numeric_pacing_passthrough(self):
        """Numeric pacing value is passed through (clamped to 0-1)."""
        seg = _map_vlm_to_segment(
            {"pacing": 0.65}, 0, 100, 10.0, 100,
        )
        assert seg.target_pacing == pytest.approx(0.65)

    def test_numeric_energy_passthrough(self):
        """Numeric energy value is passed through (clamped to 0-1)."""
        seg = _map_vlm_to_segment(
            {"energy": 0.9}, 0, 100, 10.0, 100,
        )
        assert seg.energy == pytest.approx(0.9)

    def test_numeric_brightness_passthrough(self):
        """Numeric brightness value is passed through (clamped to 0-1)."""
        seg = _map_vlm_to_segment(
            {"brightness": 0.35}, 0, 100, 10.0, 100,
        )
        assert seg.brightness == pytest.approx(0.35)

    def test_numeric_values_clamped_high(self):
        """Numeric values above 1.0 are clamped to 1.0."""
        seg = _map_vlm_to_segment(
            {"pacing": 5.0, "energy": 2.5, "brightness": 1.5},
            0, 100, 10.0, 100,
        )
        assert seg.target_pacing == pytest.approx(1.0)
        assert seg.energy == pytest.approx(1.0)
        assert seg.brightness == pytest.approx(1.0)

    def test_numeric_values_clamped_low(self):
        """Numeric values below 0.0 are clamped to 0.0."""
        seg = _map_vlm_to_segment(
            {"pacing": -0.5, "energy": -1.0, "brightness": -0.1},
            0, 100, 10.0, 100,
        )
        assert seg.target_pacing == pytest.approx(0.0)
        assert seg.energy == pytest.approx(0.0)
        assert seg.brightness == pytest.approx(0.0)

    def test_unknown_shot_type_mapped(self):
        """Unknown shot_type is mapped to nearest known type."""
        seg = _map_vlm_to_segment(
            {"shot_type": "establishing shot"}, 0, 100, 10.0, 100,
        )
        # "establishing" triggers the "wide" keyword heuristic
        assert seg.shot_type == "wide shot"

    def test_missing_fields_uses_defaults(self):
        """Missing fields use defaults without crashing."""
        seg = _map_vlm_to_segment(
            {"color_mood": "vibrant"}, 0, 100, 10.0, 100,
        )
        assert seg.target_pacing == pytest.approx(0.5)  # default "medium"
        assert seg.energy == pytest.approx(0.5)  # _ENERGY_DEFAULT
        assert seg.brightness == pytest.approx(0.5)  # default "medium"
        assert seg.shot_type == "medium shot"  # default
        assert seg.color_mood == "vibrant"

    def test_empty_dict_valid_segment(self):
        """Empty dict produces a valid segment with all defaults."""
        seg = _map_vlm_to_segment({}, 0, 100, 10.0, 100)
        assert isinstance(seg, DrawingSegment)
        assert seg.x_start == 0
        assert seg.x_end == 100
        assert seg.target_pacing == pytest.approx(0.5)
        assert seg.energy == pytest.approx(0.5)
        assert seg.brightness == pytest.approx(0.5)
        assert seg.shot_type == "medium shot"
        assert seg.color_mood is None
        assert seg.target_color is None
        assert seg.is_bw is False

    def test_duration_proportional_to_slice_width(self):
        """Duration is proportional: total * (slice_width / canvas_width)."""
        seg = _map_vlm_to_segment(
            {}, x_start=50, x_end=150,
            total_duration_seconds=120.0, canvas_width=400,
        )
        # 120 * (100 / 400) = 30.0
        assert seg.target_duration_seconds == pytest.approx(30.0)

    def test_duration_full_canvas(self):
        """Slice covering the full canvas gets the full duration."""
        seg = _map_vlm_to_segment(
            {}, x_start=0, x_end=500,
            total_duration_seconds=60.0, canvas_width=500,
        )
        assert seg.target_duration_seconds == pytest.approx(60.0)

    def test_color_mood_normalized_to_lowercase(self):
        """Color mood is normalized to lowercase."""
        seg = _map_vlm_to_segment(
            {"color_mood": "Warm"}, 0, 100, 10.0, 100,
        )
        assert seg.color_mood == "warm"

    def test_color_mood_stripped(self):
        """Color mood whitespace is stripped."""
        seg = _map_vlm_to_segment(
            {"color_mood": "  cool  "}, 0, 100, 10.0, 100,
        )
        assert seg.color_mood == "cool"

    def test_pacing_string_case_insensitive(self):
        """Pacing string matching is case-insensitive."""
        seg = _map_vlm_to_segment(
            {"pacing": "FAST"}, 0, 100, 10.0, 100,
        )
        assert seg.target_pacing == pytest.approx(0.8)

    def test_brightness_string_case_insensitive(self):
        """Brightness string matching is case-insensitive."""
        seg = _map_vlm_to_segment(
            {"brightness": "DARK"}, 0, 100, 10.0, 100,
        )
        assert seg.brightness == pytest.approx(0.2)


# ──────────────────────────────────────────────────────────────
# 3. Shot type closest match (_closest_shot_type)
# ──────────────────────────────────────────────────────────────


class TestClosestShotType:
    """Tests for _closest_shot_type fuzzy matching."""

    def test_exact_match_close_up(self):
        """Exact match: "close-up" returns "close-up"."""
        assert _closest_shot_type("close-up") == "close-up"

    def test_exact_match_wide_shot(self):
        """Exact match: "wide shot" returns "wide shot"."""
        assert _closest_shot_type("wide shot") == "wide shot"

    def test_exact_match_medium_shot(self):
        """Exact match: "medium shot" returns "medium shot"."""
        assert _closest_shot_type("medium shot") == "medium shot"

    def test_exact_match_full_shot(self):
        """Exact match: "full shot" returns "full shot"."""
        assert _closest_shot_type("full shot") == "full shot"

    def test_exact_match_extreme_close_up(self):
        """Exact match: "extreme close-up" returns "extreme close-up"."""
        assert _closest_shot_type("extreme close-up") == "extreme close-up"

    def test_substring_extreme_close(self):
        """Substring match: "extreme close" is contained in "extreme close-up"."""
        assert _closest_shot_type("extreme close") == "extreme close-up"

    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        assert _closest_shot_type("Close-Up") == "close-up"
        assert _closest_shot_type("WIDE SHOT") == "wide shot"

    def test_keyword_establishing(self):
        """Keyword heuristic: "establishing" maps to "wide shot"."""
        assert _closest_shot_type("establishing shot") == "wide shot"

    def test_keyword_long(self):
        """Keyword heuristic: "long shot" maps to "wide shot"."""
        assert _closest_shot_type("long shot") == "wide shot"

    def test_keyword_tight(self):
        """Keyword heuristic: "tight shot" maps to "close-up"."""
        assert _closest_shot_type("tight shot") == "close-up"

    def test_keyword_cowboy(self):
        """Keyword heuristic: "cowboy shot" maps to "medium shot"."""
        assert _closest_shot_type("cowboy shot") == "medium shot"

    def test_keyword_mid(self):
        """Keyword heuristic: "mid shot" maps to "medium shot"."""
        assert _closest_shot_type("mid shot") == "medium shot"

    def test_keyword_ecu(self):
        """Keyword heuristic: "ECU" maps to "extreme close-up"."""
        assert _closest_shot_type("ECU") == "extreme close-up"

    def test_unknown_string_fallback(self):
        """Completely unknown string falls back to "medium shot"."""
        assert _closest_shot_type("something totally unrecognizable") == "medium shot"

    def test_empty_string_fallback(self):
        """Empty string falls back to "medium shot"."""
        assert _closest_shot_type("") == "medium shot"

    def test_none_like_empty(self):
        """None input falls back to "medium shot" (via 'if not value')."""
        assert _closest_shot_type(None) == "medium shot"

    def test_whitespace_trimmed(self):
        """Leading/trailing whitespace is trimmed before matching."""
        assert _closest_shot_type("  close-up  ") == "close-up"

    def test_substring_wide(self):
        """Substring: "wide" is contained in "wide shot"."""
        assert _closest_shot_type("wide") == "wide shot"

    def test_full_body_maps_to_full_shot(self):
        """Keyword heuristic: "full body" contains "full" -> "full shot"."""
        assert _closest_shot_type("full body") == "full shot"


# ──────────────────────────────────────────────────────────────
# 4. Equal slicing helper (_equal_slices)
# ──────────────────────────────────────────────────────────────


class TestEqualSlices:
    """Tests for _equal_slices helper function."""

    def test_basic_slicing(self):
        """Divides width into equal parts."""
        slices = _equal_slices(300, 3)
        assert len(slices) == 3
        assert slices[0] == (0, 100)
        assert slices[1] == (100, 200)
        assert slices[2] == (200, 300)

    def test_single_slice(self):
        """Count of 1 produces one slice spanning full width."""
        slices = _equal_slices(200, 1)
        assert len(slices) == 1
        assert slices[0] == (0, 200)

    def test_count_zero_clamped(self):
        """Count of 0 is clamped to 1."""
        slices = _equal_slices(100, 0)
        assert len(slices) == 1
        assert slices[0] == (0, 100)

    def test_count_negative_clamped(self):
        """Negative count is clamped to 1."""
        slices = _equal_slices(100, -5)
        assert len(slices) == 1

    def test_uneven_division(self):
        """Width not evenly divisible still covers the full width."""
        slices = _equal_slices(100, 3)
        assert len(slices) == 3
        # First slice starts at 0, last slice ends at or near 100
        assert slices[0][0] == 0
        assert slices[-1][1] == 100
        # No gaps
        for i in range(len(slices) - 1):
            assert slices[i][1] == slices[i + 1][0]


# ──────────────────────────────────────────────────────────────
# 5. Adaptive slicing (slice_drawing_adaptive) — requires Qt
# ──────────────────────────────────────────────────────────────


class TestSliceDrawingAdaptive:
    """Tests for slice_drawing_adaptive boundary detection."""

    @qt_required
    def test_solid_color_falls_back_to_equal(self):
        """Solid color image has no visual changes, falls back to equal slicing."""
        from core.remix.drawing_vlm import slice_drawing_adaptive

        img = QImage(300, 100, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(QColor(128, 128, 128))

        slices = slice_drawing_adaptive(img, min_slices=3, max_slices=20)

        # Should fall back to min_slices equal slices
        assert len(slices) == 3
        # Covers full width
        assert slices[0][0] == 0
        assert slices[-1][1] == 300

    @qt_required
    def test_two_colored_halves_boundary_near_center(self):
        """Image with two distinct colored halves detects boundary near center."""
        from core.remix.drawing_vlm import slice_drawing_adaptive

        width = 300
        img = QImage(width, 100, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(img)
        # Left half: solid red
        painter.fillRect(0, 0, 150, 100, QColor(255, 0, 0))
        # Right half: solid blue
        painter.fillRect(150, 0, 150, 100, QColor(0, 0, 255))
        painter.end()

        slices = slice_drawing_adaptive(img, min_slices=2, max_slices=20)

        # Should detect a boundary near x=150
        assert len(slices) >= 2

        # Find the boundary closest to center
        boundaries = [s[1] for s in slices[:-1]]
        center = width // 2
        closest_boundary = min(boundaries, key=lambda b: abs(b - center))
        # Boundary should be within 20 pixels of the actual transition
        assert abs(closest_boundary - center) <= 20, (
            f"Expected boundary near {center}, closest was {closest_boundary}"
        )

    @qt_required
    def test_min_slices_constraint(self):
        """Result always has at least min_slices slices."""
        from core.remix.drawing_vlm import slice_drawing_adaptive

        img = QImage(300, 100, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(QColor(200, 200, 200))  # uniform image

        slices = slice_drawing_adaptive(img, min_slices=5, max_slices=20)
        assert len(slices) >= 5

    @qt_required
    def test_max_slices_constraint(self):
        """Result never exceeds max_slices slices."""
        from core.remix.drawing_vlm import slice_drawing_adaptive

        # Create image with many color bands to trigger many boundaries
        width = 600
        img = QImage(width, 100, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(img)
        colors = [
            QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255),
            QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255),
            QColor(128, 0, 0), QColor(0, 128, 0), QColor(0, 0, 128),
            QColor(128, 128, 0), QColor(128, 0, 128), QColor(0, 128, 128),
        ]
        band_width = width // len(colors)
        for i, color in enumerate(colors):
            painter.fillRect(i * band_width, 0, band_width, 100, color)
        painter.end()

        max_s = 5
        slices = slice_drawing_adaptive(img, min_slices=2, max_slices=max_s)
        assert len(slices) <= max_s

    @qt_required
    def test_empty_image_returns_empty(self):
        """Zero-dimension image returns empty list."""
        from core.remix.drawing_vlm import slice_drawing_adaptive

        img = QImage(0, 0, QImage.Format.Format_ARGB32_Premultiplied)
        slices = slice_drawing_adaptive(img)
        assert slices == []

    @qt_required
    def test_zero_width_returns_empty(self):
        """Zero-width image returns empty list."""
        from core.remix.drawing_vlm import slice_drawing_adaptive

        img = QImage(0, 100, QImage.Format.Format_ARGB32_Premultiplied)
        slices = slice_drawing_adaptive(img)
        assert slices == []

    @qt_required
    def test_zero_height_returns_empty(self):
        """Zero-height image returns empty list."""
        from core.remix.drawing_vlm import slice_drawing_adaptive

        img = QImage(300, 0, QImage.Format.Format_ARGB32_Premultiplied)
        slices = slice_drawing_adaptive(img)
        assert slices == []

    @qt_required
    def test_slices_cover_full_width(self):
        """All slices together cover the full image width with no gaps."""
        from core.remix.drawing_vlm import slice_drawing_adaptive

        width = 400
        img = QImage(width, 100, QImage.Format.Format_ARGB32_Premultiplied)
        # Two-color split for a realistic test
        painter = QPainter(img)
        painter.fillRect(0, 0, 200, 100, QColor(255, 0, 0))
        painter.fillRect(200, 0, 200, 100, QColor(0, 0, 255))
        painter.end()

        slices = slice_drawing_adaptive(img, min_slices=2, max_slices=10)

        assert len(slices) >= 1
        # First slice starts at 0
        assert slices[0][0] == 0
        # Last slice ends at width
        assert slices[-1][1] == width
        # No gaps between adjacent slices
        for i in range(len(slices) - 1):
            assert slices[i][1] == slices[i + 1][0], (
                f"Gap between slice {i} end ({slices[i][1]}) "
                f"and slice {i+1} start ({slices[i+1][0]})"
            )

    @qt_required
    def test_three_color_bands(self):
        """Three distinct color bands should produce at least 3 slices."""
        from core.remix.drawing_vlm import slice_drawing_adaptive

        width = 300
        img = QImage(width, 100, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(img)
        painter.fillRect(0, 0, 100, 100, QColor(255, 0, 0))
        painter.fillRect(100, 0, 100, 100, QColor(0, 255, 0))
        painter.fillRect(200, 0, 100, 100, QColor(0, 0, 255))
        painter.end()

        slices = slice_drawing_adaptive(img, min_slices=2, max_slices=20)

        # With 3 sharp transitions, should detect at least 3 slices
        assert len(slices) >= 3

    @qt_required
    def test_narrow_image(self):
        """Very narrow image (e.g. 10px) still produces valid slices."""
        from core.remix.drawing_vlm import slice_drawing_adaptive

        img = QImage(10, 100, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(QColor(100, 100, 100))

        slices = slice_drawing_adaptive(img, min_slices=2, max_slices=5)
        assert len(slices) >= 1
        assert slices[0][0] == 0
        assert slices[-1][1] == 10

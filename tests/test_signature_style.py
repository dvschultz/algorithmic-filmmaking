"""Tests for Signature Style Phase 1 core algorithm.

Covers:
- DrawingSegment dataclass
- Parametric sampling from QImage
- Segment merging logic
- Clip matching (color, B&W, reuse)
- Duration trimming / sequence building
- Hue distance computation
"""

from pathlib import Path

import pytest

from core.remix.drawing_segment import DrawingSegment
from core.remix.signature_style import (
    _hue_distance,
    _merge_samples,
    _colors_similar,
    _PACING_MERGE_THRESHOLD,
    _COLOR_MERGE_THRESHOLD,
    match_clips_to_segments,
    build_sequence_from_matches,
    sample_drawing_parametric,
)
from models.clip import Clip, Source

# Guard Qt imports for CI environments without PySide6
try:
    from PySide6.QtGui import QImage, QPainter, QColor, QPen
    from PySide6.QtCore import Qt, QPoint

    HAS_QT = True
except ImportError:
    HAS_QT = False

qt_required = pytest.mark.skipif(not HAS_QT, reason="PySide6 not available")


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def color_source() -> Source:
    """A color video source."""
    return Source(
        id="src-color",
        file_path=Path("/fake/color_video.mp4"),
        fps=30.0,
        color_profile="color",
    )


@pytest.fixture
def bw_source() -> Source:
    """A grayscale video source."""
    return Source(
        id="src-bw",
        file_path=Path("/fake/bw_video.mp4"),
        fps=30.0,
        color_profile="grayscale",
    )


@pytest.fixture
def red_clip(color_source: Source) -> Clip:
    """A clip with dominant red color, 3 seconds at 30fps."""
    return Clip(
        id="clip-red",
        source_id=color_source.id,
        start_frame=0,
        end_frame=90,
        dominant_colors=[(255, 0, 0)],
    )


@pytest.fixture
def blue_clip(color_source: Source) -> Clip:
    """A clip with dominant blue color, 3 seconds at 30fps."""
    return Clip(
        id="clip-blue",
        source_id=color_source.id,
        start_frame=100,
        end_frame=190,
        dominant_colors=[(0, 0, 255)],
    )


@pytest.fixture
def green_clip(color_source: Source) -> Clip:
    """A clip with dominant green color, 2 seconds at 30fps."""
    return Clip(
        id="clip-green",
        source_id=color_source.id,
        start_frame=200,
        end_frame=260,
        dominant_colors=[(0, 255, 0)],
    )


@pytest.fixture
def bw_clip(bw_source: Source) -> Clip:
    """A clip from a B&W source, 3 seconds at 30fps."""
    return Clip(
        id="clip-bw",
        source_id=bw_source.id,
        start_frame=0,
        end_frame=90,
        dominant_colors=[(128, 128, 128)],
    )


@pytest.fixture
def long_clip(color_source: Source) -> Clip:
    """A long clip (10 seconds at 30fps = 300 frames)."""
    return Clip(
        id="clip-long",
        source_id=color_source.id,
        start_frame=0,
        end_frame=300,
        dominant_colors=[(255, 0, 0)],
    )


@pytest.fixture
def short_clip(color_source: Source) -> Clip:
    """A short clip (1 second at 30fps = 30 frames)."""
    return Clip(
        id="clip-short",
        source_id=color_source.id,
        start_frame=0,
        end_frame=30,
        dominant_colors=[(255, 0, 0)],
    )


# ──────────────────────────────────────────────────────────────
# 1. DrawingSegment dataclass
# ──────────────────────────────────────────────────────────────


class TestDrawingSegment:
    """Tests for DrawingSegment construction and properties."""

    def test_construction_required_fields(self):
        seg = DrawingSegment(
            x_start=0,
            x_end=100,
            target_duration_seconds=5.0,
            target_pacing=0.7,
        )
        assert seg.x_start == 0
        assert seg.x_end == 100
        assert seg.target_duration_seconds == 5.0
        assert seg.target_pacing == 0.7

    def test_width_property(self):
        seg = DrawingSegment(
            x_start=10,
            x_end=60,
            target_duration_seconds=2.0,
            target_pacing=0.5,
        )
        assert seg.width == 50

    def test_width_zero(self):
        seg = DrawingSegment(
            x_start=50,
            x_end=50,
            target_duration_seconds=0.0,
            target_pacing=0.5,
        )
        assert seg.width == 0

    def test_default_optional_fields(self):
        seg = DrawingSegment(
            x_start=0,
            x_end=100,
            target_duration_seconds=5.0,
            target_pacing=0.5,
        )
        assert seg.target_color is None
        assert seg.is_bw is False
        assert seg.shot_type is None
        assert seg.energy is None
        assert seg.brightness is None
        assert seg.color_mood is None

    def test_construction_with_color(self):
        seg = DrawingSegment(
            x_start=0,
            x_end=100,
            target_duration_seconds=5.0,
            target_pacing=0.5,
            target_color=(255, 0, 0),
        )
        assert seg.target_color == (255, 0, 0)
        assert seg.is_bw is False

    def test_construction_bw_segment(self):
        seg = DrawingSegment(
            x_start=0,
            x_end=100,
            target_duration_seconds=5.0,
            target_pacing=0.5,
            is_bw=True,
        )
        assert seg.is_bw is True
        assert seg.target_color is None

    def test_construction_vlm_fields(self):
        seg = DrawingSegment(
            x_start=0,
            x_end=100,
            target_duration_seconds=5.0,
            target_pacing=0.5,
            shot_type="close-up",
            energy=0.8,
            brightness=0.6,
            color_mood="warm",
        )
        assert seg.shot_type == "close-up"
        assert seg.energy == 0.8
        assert seg.brightness == 0.6
        assert seg.color_mood == "warm"

    def test_proportion_property(self):
        seg = DrawingSegment(
            x_start=0,
            x_end=50,
            target_duration_seconds=1.0,
            target_pacing=0.5,
        )
        # proportion returns width; caller divides by canvas_width
        assert seg.proportion == 50


# ──────────────────────────────────────────────────────────────
# 2. Parametric sampling (QImage-based)
# ──────────────────────────────────────────────────────────────


class TestParametricSampling:
    """Tests for sample_drawing_parametric with synthetic QImage inputs."""

    @qt_required
    def test_horizontal_red_line(self):
        """Horizontal red line at mid-height -> ~0.5 pacing, red color."""
        img = QImage(200, 100, QImage.Format_ARGB32_Premultiplied)
        img.fill(QColor(Qt.white))
        painter = QPainter(img)
        painter.setPen(QPen(QColor(255, 0, 0), 3))
        painter.drawLine(QPoint(0, 50), QPoint(199, 50))
        painter.end()

        segments = sample_drawing_parametric(img, total_duration_seconds=10.0, sample_count=4)

        assert len(segments) > 0

        # All segments should be roughly red and ~0.5 pacing
        for seg in segments:
            assert seg.target_color is not None, "Red line should produce color segments"
            assert seg.is_bw is False
            # Red channel should dominate
            r, g, b = seg.target_color
            assert r > g and r > b, f"Expected red-dominant color, got ({r},{g},{b})"
            # Pacing at middle height: approximately 0.5
            assert 0.3 <= seg.target_pacing <= 0.7, (
                f"Expected ~0.5 pacing for mid-height line, got {seg.target_pacing}"
            )

    @qt_required
    def test_diagonal_line(self):
        """Diagonal line from top-left to bottom-right -> varying pacing."""
        img = QImage(200, 100, QImage.Format_ARGB32_Premultiplied)
        img.fill(QColor(Qt.white))
        painter = QPainter(img)
        painter.setPen(QPen(QColor(0, 0, 0), 3))
        painter.drawLine(QPoint(0, 0), QPoint(199, 99))
        painter.end()

        segments = sample_drawing_parametric(img, total_duration_seconds=10.0, sample_count=8)

        assert len(segments) > 0

        # B&W line should have is_bw=True
        for seg in segments:
            assert seg.is_bw is True

        # If there are multiple segments, first should have higher pacing
        # (top of image = high pacing) and last should have lower pacing
        if len(segments) >= 2:
            assert segments[0].target_pacing > segments[-1].target_pacing, (
                f"First segment pacing ({segments[0].target_pacing}) should be higher "
                f"than last ({segments[-1].target_pacing}) for top-left to bottom-right diagonal"
            )

    @qt_required
    def test_blank_white_canvas(self):
        """Blank white canvas -> all segments default to 0.5 pacing, B&W."""
        img = QImage(200, 100, QImage.Format_ARGB32_Premultiplied)
        img.fill(QColor(Qt.white))

        segments = sample_drawing_parametric(img, total_duration_seconds=10.0, sample_count=4)

        # Blank canvas still produces segments (with default 0.5 pacing)
        # but they may all merge into one since they're identical
        assert len(segments) >= 1

        for seg in segments:
            assert seg.is_bw is True
            assert seg.target_color is None
            assert seg.target_pacing == pytest.approx(0.5, abs=0.01)

    @qt_required
    def test_black_drawing(self):
        """Full black drawing -> B&W segments with no target_color."""
        img = QImage(200, 100, QImage.Format_ARGB32_Premultiplied)
        img.fill(QColor(Qt.white))
        painter = QPainter(img)
        painter.setPen(QPen(QColor(0, 0, 0), 5))
        # Draw thick horizontal lines across the image
        for y in range(10, 90, 10):
            painter.drawLine(QPoint(0, y), QPoint(199, y))
        painter.end()

        segments = sample_drawing_parametric(img, total_duration_seconds=10.0, sample_count=4)

        assert len(segments) > 0
        for seg in segments:
            assert seg.is_bw is True
            assert seg.target_color is None

    @qt_required
    def test_segment_count_respects_sample_count(self):
        """Number of raw samples matches sample_count (before merging)."""
        img = QImage(400, 100, QImage.Format_ARGB32_Premultiplied)
        img.fill(QColor(Qt.white))
        painter = QPainter(img)
        # Draw distinct colored bands so they won't merge
        colors = [
            QColor(255, 0, 0),
            QColor(0, 255, 0),
            QColor(0, 0, 255),
            QColor(255, 255, 0),
        ]
        band_width = 100
        for i, color in enumerate(colors):
            painter.setPen(QPen(color, 5))
            x_start = i * band_width
            painter.drawLine(QPoint(x_start + 10, 50), QPoint(x_start + 90, 50))
        painter.end()

        # With 4 samples, one per band, we should get 4 distinct segments
        segments = sample_drawing_parametric(img, total_duration_seconds=10.0, sample_count=4)
        # Each sample falls on a different color band, so all 4 should be distinct
        assert len(segments) == 4

    @qt_required
    def test_sample_count_clamped_to_width(self):
        """sample_count larger than image width is clamped."""
        img = QImage(10, 10, QImage.Format_ARGB32_Premultiplied)
        img.fill(QColor(Qt.white))
        painter = QPainter(img)
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.drawLine(QPoint(0, 5), QPoint(9, 5))
        painter.end()

        # Request 1000 samples on a 10-pixel-wide image
        segments = sample_drawing_parametric(img, total_duration_seconds=5.0, sample_count=1000)
        # Should still work and produce segments (clamped to width=10)
        assert len(segments) >= 1

    @qt_required
    def test_duration_allocation_proportional(self):
        """Segment durations should sum to total_duration_seconds."""
        img = QImage(200, 100, QImage.Format_ARGB32_Premultiplied)
        img.fill(QColor(Qt.white))
        painter = QPainter(img)
        painter.setPen(QPen(QColor(255, 0, 0), 3))
        painter.drawLine(QPoint(0, 50), QPoint(199, 50))
        painter.end()

        total_duration = 12.0
        segments = sample_drawing_parametric(img, total_duration_seconds=total_duration, sample_count=8)

        duration_sum = sum(seg.target_duration_seconds for seg in segments)
        assert duration_sum == pytest.approx(total_duration, abs=0.01)

    @qt_required
    def test_empty_image_returns_empty(self):
        """A zero-dimension image returns no segments."""
        img = QImage(0, 0, QImage.Format_ARGB32_Premultiplied)
        segments = sample_drawing_parametric(img, total_duration_seconds=10.0)
        assert segments == []


# ──────────────────────────────────────────────────────────────
# 3. Segment merging (_merge_samples)
# ──────────────────────────────────────────────────────────────


class TestMergeSamples:
    """Tests for _merge_samples merging logic."""

    def test_identical_samples_merge(self):
        """Adjacent samples with same pacing and color merge into one."""
        samples = [
            {"x_start": 0, "x_end": 10, "pacing": 0.5, "color": (255, 0, 0), "is_bw": False},
            {"x_start": 10, "x_end": 20, "pacing": 0.5, "color": (255, 0, 0), "is_bw": False},
            {"x_start": 20, "x_end": 30, "pacing": 0.5, "color": (255, 0, 0), "is_bw": False},
        ]
        merged = _merge_samples(samples)
        assert len(merged) == 1
        assert merged[0]["x_start"] == 0
        assert merged[0]["x_end"] == 30

    def test_different_pacing_no_merge(self):
        """Samples with very different pacing stay separate."""
        samples = [
            {"x_start": 0, "x_end": 10, "pacing": 0.1, "color": (255, 0, 0), "is_bw": False},
            {"x_start": 10, "x_end": 20, "pacing": 0.9, "color": (255, 0, 0), "is_bw": False},
        ]
        merged = _merge_samples(samples)
        assert len(merged) == 2

    def test_different_color_no_merge(self):
        """Samples with very different colors stay separate."""
        samples = [
            {"x_start": 0, "x_end": 10, "pacing": 0.5, "color": (255, 0, 0), "is_bw": False},
            {"x_start": 10, "x_end": 20, "pacing": 0.5, "color": (0, 0, 255), "is_bw": False},
        ]
        merged = _merge_samples(samples)
        assert len(merged) == 2

    def test_bw_and_color_no_merge(self):
        """B&W and color samples don't merge even if pacing matches."""
        samples = [
            {"x_start": 0, "x_end": 10, "pacing": 0.5, "color": None, "is_bw": True},
            {"x_start": 10, "x_end": 20, "pacing": 0.5, "color": (255, 0, 0), "is_bw": False},
        ]
        merged = _merge_samples(samples)
        assert len(merged) == 2

    def test_bw_samples_merge(self):
        """Two B&W samples with similar pacing merge."""
        samples = [
            {"x_start": 0, "x_end": 10, "pacing": 0.5, "color": None, "is_bw": True},
            {"x_start": 10, "x_end": 20, "pacing": 0.5, "color": None, "is_bw": True},
        ]
        merged = _merge_samples(samples)
        assert len(merged) == 1
        assert merged[0]["is_bw"] is True

    def test_pacing_within_threshold_merges(self):
        """Samples within _PACING_MERGE_THRESHOLD merge."""
        small_diff = _PACING_MERGE_THRESHOLD * 0.5  # well within threshold
        samples = [
            {"x_start": 0, "x_end": 10, "pacing": 0.5, "color": (255, 0, 0), "is_bw": False},
            {"x_start": 10, "x_end": 20, "pacing": 0.5 + small_diff, "color": (255, 0, 0), "is_bw": False},
        ]
        merged = _merge_samples(samples)
        assert len(merged) == 1

    def test_pacing_beyond_threshold_no_merge(self):
        """Samples beyond _PACING_MERGE_THRESHOLD don't merge."""
        big_diff = _PACING_MERGE_THRESHOLD * 2.0  # well beyond threshold
        samples = [
            {"x_start": 0, "x_end": 10, "pacing": 0.5, "color": (255, 0, 0), "is_bw": False},
            {"x_start": 10, "x_end": 20, "pacing": 0.5 + big_diff, "color": (255, 0, 0), "is_bw": False},
        ]
        merged = _merge_samples(samples)
        assert len(merged) == 2

    def test_empty_input(self):
        """Empty input returns empty output."""
        assert _merge_samples([]) == []

    def test_single_sample(self):
        """Single sample returns single item."""
        samples = [
            {"x_start": 0, "x_end": 10, "pacing": 0.5, "color": (255, 0, 0), "is_bw": False},
        ]
        merged = _merge_samples(samples)
        assert len(merged) == 1

    def test_merge_averages_pacing(self):
        """Merged samples produce averaged pacing."""
        samples = [
            {"x_start": 0, "x_end": 10, "pacing": 0.50, "color": None, "is_bw": True},
            {"x_start": 10, "x_end": 20, "pacing": 0.54, "color": None, "is_bw": True},
        ]
        merged = _merge_samples(samples)
        assert len(merged) == 1
        # Running average of 0.50 and 0.54 = 0.52
        assert merged[0]["pacing"] == pytest.approx(0.52, abs=0.01)

    def test_partial_merge_sequence(self):
        """A sequence of samples where only some adjacent pairs merge."""
        samples = [
            {"x_start": 0, "x_end": 10, "pacing": 0.5, "color": (255, 0, 0), "is_bw": False},
            {"x_start": 10, "x_end": 20, "pacing": 0.5, "color": (255, 0, 0), "is_bw": False},
            {"x_start": 20, "x_end": 30, "pacing": 0.9, "color": (0, 0, 255), "is_bw": False},
            {"x_start": 30, "x_end": 40, "pacing": 0.9, "color": (0, 0, 255), "is_bw": False},
        ]
        merged = _merge_samples(samples)
        assert len(merged) == 2
        # First merged segment covers 0-20
        assert merged[0]["x_start"] == 0
        assert merged[0]["x_end"] == 20
        # Second merged segment covers 20-40
        assert merged[1]["x_start"] == 20
        assert merged[1]["x_end"] == 40


class TestColorsSimilar:
    """Tests for the _colors_similar helper."""

    def test_both_none(self):
        assert _colors_similar(None, None) is True

    def test_one_none(self):
        assert _colors_similar(None, (255, 0, 0)) is False
        assert _colors_similar((255, 0, 0), None) is False

    def test_identical_colors(self):
        assert _colors_similar((255, 0, 0), (255, 0, 0)) is True

    def test_similar_colors(self):
        # Distance = sqrt(10^2 + 10^2 + 10^2) ~ 17.3 < default threshold 30
        assert _colors_similar((255, 0, 0), (245, 10, 10)) is True

    def test_very_different_colors(self):
        # Red vs Blue: distance = sqrt(255^2 + 0 + 255^2) ~ 360 > 30
        assert _colors_similar((255, 0, 0), (0, 0, 255)) is False


# ──────────────────────────────────────────────────────────────
# 4. Clip matching
# ──────────────────────────────────────────────────────────────


class TestClipMatching:
    """Tests for match_clips_to_segments."""

    def test_red_segment_matches_red_clip(
        self, color_source, red_clip, blue_clip
    ):
        """Red segment should prefer a red clip over a blue clip."""
        segment = DrawingSegment(
            x_start=0,
            x_end=100,
            target_duration_seconds=3.0,
            target_pacing=0.5,
            target_color=(255, 0, 0),
            is_bw=False,
        )

        clips = [(red_clip, color_source), (blue_clip, color_source)]
        matches = match_clips_to_segments([segment], clips)

        assert len(matches) == 1
        matched_clip, matched_source, matched_seg = matches[0]
        assert matched_clip.id == "clip-red"

    def test_blue_segment_matches_blue_clip(
        self, color_source, red_clip, blue_clip
    ):
        """Blue segment should prefer a blue clip over a red clip."""
        segment = DrawingSegment(
            x_start=0,
            x_end=100,
            target_duration_seconds=3.0,
            target_pacing=0.5,
            target_color=(0, 0, 255),
            is_bw=False,
        )

        clips = [(red_clip, color_source), (blue_clip, color_source)]
        matches = match_clips_to_segments([segment], clips)

        assert len(matches) == 1
        matched_clip, _source, _seg = matches[0]
        assert matched_clip.id == "clip-blue"

    def test_bw_segment_prefers_bw_source(
        self, color_source, bw_source, red_clip, bw_clip
    ):
        """B&W segment should prefer clip from grayscale source."""
        segment = DrawingSegment(
            x_start=0,
            x_end=100,
            target_duration_seconds=3.0,
            target_pacing=0.5,
            is_bw=True,
        )

        clips = [(red_clip, color_source), (bw_clip, bw_source)]
        matches = match_clips_to_segments([segment], clips)

        assert len(matches) == 1
        matched_clip, matched_source, _seg = matches[0]
        assert matched_clip.id == "clip-bw"
        assert matched_source.color_profile == "grayscale"

    def test_clip_reuse_when_pool_small(self, color_source, red_clip):
        """With allow_reuse=True, same clip can match multiple segments."""
        segments = [
            DrawingSegment(
                x_start=0, x_end=50,
                target_duration_seconds=2.0, target_pacing=0.5,
                target_color=(255, 0, 0), is_bw=False,
            ),
            DrawingSegment(
                x_start=50, x_end=100,
                target_duration_seconds=2.0, target_pacing=0.5,
                target_color=(255, 0, 0), is_bw=False,
            ),
            DrawingSegment(
                x_start=100, x_end=150,
                target_duration_seconds=2.0, target_pacing=0.5,
                target_color=(255, 0, 0), is_bw=False,
            ),
        ]

        clips = [(red_clip, color_source)]  # Only one clip available
        matches = match_clips_to_segments(segments, clips, allow_reuse=True)

        # All 3 segments should be matched (reusing the single clip)
        assert len(matches) == 3
        for clip, _source, _seg in matches:
            assert clip.id == "clip-red"

    def test_no_reuse_exhausts_pool(self, color_source, red_clip, blue_clip):
        """With allow_reuse=False, each clip is used at most once."""
        segments = [
            DrawingSegment(
                x_start=0, x_end=50,
                target_duration_seconds=3.0, target_pacing=0.5,
                target_color=(255, 0, 0), is_bw=False,
            ),
            DrawingSegment(
                x_start=50, x_end=100,
                target_duration_seconds=3.0, target_pacing=0.5,
                target_color=(255, 0, 0), is_bw=False,
            ),
            DrawingSegment(
                x_start=100, x_end=150,
                target_duration_seconds=3.0, target_pacing=0.5,
                target_color=(255, 0, 0), is_bw=False,
            ),
        ]

        clips = [(red_clip, color_source), (blue_clip, color_source)]
        matches = match_clips_to_segments(segments, clips, allow_reuse=False)

        # Only 2 clips available, so only 2 segments matched
        assert len(matches) == 2
        used_ids = {m[0].id for m in matches}
        assert len(used_ids) == 2  # No duplicates

    def test_empty_segments(self, color_source, red_clip):
        """No segments -> no matches."""
        clips = [(red_clip, color_source)]
        matches = match_clips_to_segments([], clips)
        assert matches == []

    def test_empty_clips(self):
        """No clips -> no matches."""
        segment = DrawingSegment(
            x_start=0, x_end=100,
            target_duration_seconds=3.0, target_pacing=0.5,
        )
        matches = match_clips_to_segments([segment], [])
        assert matches == []

    def test_multiple_segments_multiple_colors(
        self, color_source, red_clip, blue_clip, green_clip
    ):
        """Each segment should match the clip with the closest color."""
        segments = [
            DrawingSegment(
                x_start=0, x_end=50,
                target_duration_seconds=2.0, target_pacing=0.5,
                target_color=(250, 10, 10), is_bw=False,  # near-red
            ),
            DrawingSegment(
                x_start=50, x_end=100,
                target_duration_seconds=2.0, target_pacing=0.5,
                target_color=(10, 10, 250), is_bw=False,  # near-blue
            ),
            DrawingSegment(
                x_start=100, x_end=150,
                target_duration_seconds=2.0, target_pacing=0.5,
                target_color=(10, 250, 10), is_bw=False,  # near-green
            ),
        ]

        clips = [
            (red_clip, color_source),
            (blue_clip, color_source),
            (green_clip, color_source),
        ]
        matches = match_clips_to_segments(segments, clips, allow_reuse=True)

        assert len(matches) == 3
        assert matches[0][0].id == "clip-red"
        assert matches[1][0].id == "clip-blue"
        assert matches[2][0].id == "clip-green"


# ──────────────────────────────────────────────────────────────
# 5. Duration trimming (build_sequence_from_matches)
# ──────────────────────────────────────────────────────────────


class TestBuildSequenceFromMatches:
    """Tests for build_sequence_from_matches trimming behavior."""

    def test_clip_longer_than_target_is_trimmed(self, color_source, long_clip):
        """Clip with 300 frames, target 2 seconds (60 frames) -> trimmed."""
        segment = DrawingSegment(
            x_start=0, x_end=100,
            target_duration_seconds=2.0,
            target_pacing=0.5,
        )
        matches = [(long_clip, color_source, segment)]
        result = build_sequence_from_matches(matches, fps=30.0)

        assert len(result) == 1
        clip, source, in_point, out_point = result[0]
        trimmed_frames = out_point - in_point
        assert trimmed_frames == 60  # 2.0 seconds * 30fps
        # Verify clip is shorter than original
        assert trimmed_frames < long_clip.duration_frames

    def test_clip_shorter_than_target_used_in_full(self, color_source, short_clip):
        """Clip with 30 frames, target 5 seconds (150 frames) -> use full clip."""
        segment = DrawingSegment(
            x_start=0, x_end=100,
            target_duration_seconds=5.0,
            target_pacing=0.5,
        )
        matches = [(short_clip, color_source, segment)]
        result = build_sequence_from_matches(matches, fps=30.0)

        assert len(result) == 1
        clip, source, in_point, out_point = result[0]
        assert in_point == 0
        assert out_point == short_clip.duration_frames  # Full clip

    def test_trim_is_centered(self, color_source, long_clip):
        """When trimming, the cut should be centered within the clip."""
        segment = DrawingSegment(
            x_start=0, x_end=100,
            target_duration_seconds=2.0,
            target_pacing=0.5,
        )
        matches = [(long_clip, color_source, segment)]
        result = build_sequence_from_matches(matches, fps=30.0)

        clip, source, in_point, out_point = result[0]
        target_frames = 60  # 2.0s * 30fps
        excess = long_clip.duration_frames - target_frames  # 300 - 60 = 240
        expected_in = excess // 2  # 120

        assert in_point == expected_in
        assert out_point == expected_in + target_frames

    def test_exact_duration_match(self, color_source):
        """Clip exactly matching target duration -> use full clip, no trim."""
        clip = Clip(
            id="clip-exact",
            source_id=color_source.id,
            start_frame=0,
            end_frame=150,  # 5 seconds at 30fps
        )
        segment = DrawingSegment(
            x_start=0, x_end=100,
            target_duration_seconds=5.0,
            target_pacing=0.5,
        )
        matches = [(clip, color_source, segment)]
        result = build_sequence_from_matches(matches, fps=30.0)

        _, _, in_point, out_point = result[0]
        assert in_point == 0
        assert out_point == 150

    def test_empty_matches(self):
        """Empty input produces empty output."""
        result = build_sequence_from_matches([], fps=30.0)
        assert result == []

    def test_multiple_matches(self, color_source, long_clip, short_clip):
        """Multiple matches are all processed."""
        seg1 = DrawingSegment(
            x_start=0, x_end=50,
            target_duration_seconds=2.0, target_pacing=0.5,
        )
        seg2 = DrawingSegment(
            x_start=50, x_end=100,
            target_duration_seconds=5.0, target_pacing=0.5,
        )
        matches = [
            (long_clip, color_source, seg1),
            (short_clip, color_source, seg2),
        ]
        result = build_sequence_from_matches(matches, fps=30.0)
        assert len(result) == 2

        # First: long clip trimmed to 2 seconds
        _, _, in1, out1 = result[0]
        assert out1 - in1 == 60

        # Second: short clip used in full (30 frames < 150 target frames)
        _, _, in2, out2 = result[1]
        assert in2 == 0
        assert out2 == 30


# ──────────────────────────────────────────────────────────────
# 6. Hue distance (_hue_distance)
# ──────────────────────────────────────────────────────────────


class TestHueDistance:
    """Tests for _hue_distance perceptual color distance."""

    def test_identical_colors_zero(self):
        """Same color -> 0.0 distance."""
        assert _hue_distance((255, 0, 0), (255, 0, 0)) == pytest.approx(0.0)

    def test_red_vs_blue_high(self):
        """Red vs blue -> high distance.

        Red hue ~0, Blue hue ~240 -> diff=120 degrees -> 120/180=0.667 hue_dist.
        Weighted: 0.7*0.667 + 0.15*0 + 0.15*0 = ~0.467
        """
        dist = _hue_distance((255, 0, 0), (0, 0, 255))
        assert dist > 0.4, f"Red vs Blue should be > 0.4, got {dist}"

    def test_red_vs_orange_low(self):
        """Red vs orange -> low distance."""
        dist = _hue_distance((255, 0, 0), (255, 128, 0))
        assert dist < 0.2, f"Red vs Orange should be < 0.2, got {dist}"

    def test_symmetric(self):
        """Distance should be symmetric: d(a,b) == d(b,a)."""
        d1 = _hue_distance((255, 0, 0), (0, 255, 0))
        d2 = _hue_distance((0, 255, 0), (255, 0, 0))
        assert d1 == pytest.approx(d2)

    def test_complementary_colors_high(self):
        """Complementary colors (opposite on wheel) -> high distance."""
        # Red and Cyan are complementary
        dist = _hue_distance((255, 0, 0), (0, 255, 255))
        assert dist > 0.6, f"Complementary colors should be > 0.6, got {dist}"

    def test_similar_blues(self):
        """Two shades of blue -> low distance."""
        dist = _hue_distance((0, 0, 255), (30, 30, 220))
        assert dist < 0.15, f"Similar blues should be < 0.15, got {dist}"

    def test_returns_between_zero_and_one(self):
        """Distance is always in [0.0, 1.0] range."""
        test_pairs = [
            ((255, 0, 0), (0, 255, 0)),
            ((0, 0, 0), (255, 255, 255)),
            ((128, 64, 32), (32, 64, 128)),
            ((255, 255, 0), (0, 255, 255)),
        ]
        for c1, c2 in test_pairs:
            dist = _hue_distance(c1, c2)
            assert 0.0 <= dist <= 1.0, f"Distance {dist} out of range for {c1} vs {c2}"

    def test_black_vs_white(self):
        """Black vs white should be max value distance (hue undefined for achromatics)."""
        dist = _hue_distance((0, 0, 0), (255, 255, 255))
        # Both are achromatic; hue distance is 0 but value distance is 1.0
        # Result = 0.7*0 + 0.15*0 + 0.15*1.0 = 0.15
        assert dist == pytest.approx(0.15, abs=0.02)

    def test_pure_red_hue_wrap_around(self):
        """Test hue wrapping: red (hue~0) vs magenta (hue~300) -> 60 degree diff."""
        # Pure red = (255, 0, 0) hue ~0
        # Magenta = (255, 0, 255) hue ~300
        dist = _hue_distance((255, 0, 0), (255, 0, 255))
        # Hue diff = min(300, 360-300) = 60 degrees -> 60/180 = 0.333
        # With saturation/value weighting, should be moderate
        assert 0.15 < dist < 0.45

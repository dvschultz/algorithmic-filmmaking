"""Tests for core/remix/gaze.py — Eyes Without a Face sequencing algorithms."""

from dataclasses import dataclass
from typing import Optional

import pytest

from core.remix.gaze import eyeline_match, gaze_filter, gaze_rotation


@dataclass
class FakeClip:
    """Minimal clip stub with gaze fields."""
    id: str = "c1"
    gaze_yaw: Optional[float] = None
    gaze_pitch: Optional[float] = None
    gaze_category: Optional[str] = None


@dataclass
class FakeSource:
    """Minimal source stub."""
    id: str = "s1"


def _make_clip(clip_id, yaw=None, pitch=None, category=None):
    return FakeClip(id=clip_id, gaze_yaw=yaw, gaze_pitch=pitch, gaze_category=category)


def _make_pair(clip_id, yaw=None, pitch=None, category=None, source_id="s1"):
    return (_make_clip(clip_id, yaw, pitch, category), FakeSource(id=source_id))


# ─── Eyeline Match ─────────────────────────────────────────

class TestEyelineMatch:
    def test_pairs_left_looking_with_right_looking(self):
        """Clips with negated yaw should be paired together."""
        clips = [
            _make_pair("left1", yaw=-25.0),
            _make_pair("right1", yaw=25.0),
            _make_pair("left2", yaw=-15.0),
            _make_pair("right2", yaw=18.0),
        ]
        result = eyeline_match(clips, tolerance=20.0)
        assert len(result) == 4

        # Should be paired: the result should alternate paired clips
        # Check that paired clips have roughly negated yaw
        ids = [clip.id for clip, source in result]
        # left2 (-15) and right2 (18) pair: abs(-15 + 18) = 3 <= 20
        # left1 (-25) and right1 (25) pair: abs(-25 + 25) = 0 <= 20
        # All clips should be present
        assert set(ids) == {"left1", "right1", "left2", "right2"}

    def test_no_complementary_pairs_returns_all_appended(self):
        """When no clips can be paired, all are returned (as unmatched)."""
        clips = [
            _make_pair("a", yaw=10.0),
            _make_pair("b", yaw=15.0),
            _make_pair("c", yaw=20.0),
        ]
        # All positive yaw — sum of any two is > 20, so no pairs
        result = eyeline_match(clips, tolerance=5.0)
        assert len(result) == 3
        result_ids = {clip.id for clip, _ in result}
        assert result_ids == {"a", "b", "c"}

    def test_odd_clips_has_unpaired_appended(self):
        """With odd number of clips, unpaired one goes to the end."""
        clips = [
            _make_pair("left", yaw=-20.0),
            _make_pair("right", yaw=20.0),
            _make_pair("extra", yaw=5.0),
        ]
        result = eyeline_match(clips, tolerance=20.0)
        assert len(result) == 3
        # The paired clips come first (2 clips), then the extra
        result_ids = [clip.id for clip, _ in result]
        assert set(result_ids) == {"left", "right", "extra"}

    def test_without_gaze_appended_at_end(self):
        """Clips without gaze data should appear at the end."""
        clips = [
            _make_pair("left", yaw=-20.0),
            _make_pair("right", yaw=20.0),
            _make_pair("no_gaze"),  # No gaze_yaw
        ]
        result = eyeline_match(clips, tolerance=20.0)
        assert len(result) == 3
        # Last clip should be the one without gaze data
        assert result[-1][0].id == "no_gaze"

    def test_empty_input(self):
        """Empty input returns empty output."""
        result = eyeline_match([], tolerance=20.0)
        assert result == []


# ─── Gaze Filter ───────────────────────────────────────────

class TestGazeFilter:
    def test_returns_only_matching_category(self):
        """Only clips matching the category should appear first."""
        clips = [
            _make_pair("a", category="at_camera"),
            _make_pair("b", category="looking_left"),
            _make_pair("c", category="at_camera"),
            _make_pair("d", category="looking_right"),
        ]
        result = gaze_filter(clips, category="at_camera")
        matching_ids = [clip.id for clip, _ in result[:2]]
        assert set(matching_ids) == {"a", "c"}
        # Non-matching are appended
        assert len(result) == 4

    def test_no_matches_returns_empty_plus_appended(self):
        """When no clips match, matching section is empty but all clips are returned."""
        clips = [
            _make_pair("a", category="looking_left"),
            _make_pair("b", category="looking_right"),
        ]
        result = gaze_filter(clips, category="at_camera")
        # No matching clips, so first items are the non-matching
        assert len(result) == 2
        # All clips should be in the non-matching section
        result_ids = {clip.id for clip, _ in result}
        assert result_ids == {"a", "b"}

    def test_all_match(self):
        """When all clips match, all appear in the matching section."""
        clips = [
            _make_pair("a", category="looking_up"),
            _make_pair("b", category="looking_up"),
        ]
        result = gaze_filter(clips, category="looking_up")
        assert len(result) == 2
        assert all(clip.gaze_category == "looking_up" for clip, _ in result)

    def test_clips_without_category(self):
        """Clips without gaze_category go to non-matching."""
        clips = [
            _make_pair("a", category="at_camera"),
            _make_pair("b"),  # No category
        ]
        result = gaze_filter(clips, category="at_camera")
        assert result[0][0].id == "a"
        assert result[1][0].id == "b"


# ─── Gaze Rotation ─────────────────────────────────────────

class TestGazeRotation:
    def test_produces_monotonically_increasing_angles(self):
        """Result should have clips sorted by ascending angle."""
        clips = [
            _make_pair("a", yaw=20.0),
            _make_pair("b", yaw=-20.0),
            _make_pair("c", yaw=0.0),
            _make_pair("d", yaw=-10.0),
            _make_pair("e", yaw=10.0),
        ]
        result = gaze_rotation(
            clips, axis="yaw", range_start=-30.0, range_end=30.0, ascending=True,
        )
        # Extract yaw values for clips with gaze data
        yaws = [clip.gaze_yaw for clip, _ in result if clip.gaze_yaw is not None]
        # Should be monotonically non-decreasing
        for i in range(len(yaws) - 1):
            assert yaws[i] <= yaws[i + 1], f"Not monotonic at index {i}: {yaws}"

    def test_sparse_data_produces_shorter_sequence(self):
        """With fewer clips than range would suggest, sequence uses all available."""
        clips = [
            _make_pair("a", yaw=-5.0),
            _make_pair("b", yaw=5.0),
        ]
        result = gaze_rotation(
            clips, axis="yaw", range_start=-30.0, range_end=30.0, ascending=True,
        )
        assert len(result) == 2
        # The clip closer to -30 should come first
        assert result[0][0].gaze_yaw <= result[1][0].gaze_yaw

    def test_descending_reverses_order(self):
        """ascending=False should reverse the angle progression."""
        clips = [
            _make_pair("a", yaw=-20.0),
            _make_pair("b", yaw=0.0),
            _make_pair("c", yaw=20.0),
        ]
        result_asc = gaze_rotation(
            clips, axis="yaw", range_start=-30.0, range_end=30.0, ascending=True,
        )
        result_desc = gaze_rotation(
            clips, axis="yaw", range_start=-30.0, range_end=30.0, ascending=False,
        )
        # Descending should be the reverse of ascending (for clips with gaze)
        asc_ids = [clip.id for clip, _ in result_asc if clip.gaze_yaw is not None]
        desc_ids = [clip.id for clip, _ in result_desc if clip.gaze_yaw is not None]
        assert asc_ids == list(reversed(desc_ids))

    def test_without_gaze_appended_at_end(self):
        """Clips without gaze data should appear at the end."""
        clips = [
            _make_pair("a", yaw=10.0),
            _make_pair("no_gaze"),
            _make_pair("b", yaw=-10.0),
        ]
        result = gaze_rotation(
            clips, axis="yaw", range_start=-30.0, range_end=30.0, ascending=True,
        )
        assert len(result) == 3
        # Last clip should be the one without gaze
        assert result[-1][0].id == "no_gaze"

    def test_pitch_axis(self):
        """Rotation works on the pitch axis too."""
        clips = [
            _make_pair("a", pitch=15.0),
            _make_pair("b", pitch=-15.0),
            _make_pair("c", pitch=0.0),
        ]
        result = gaze_rotation(
            clips, axis="pitch", range_start=-20.0, range_end=20.0, ascending=True,
        )
        pitches = [clip.gaze_pitch for clip, _ in result if clip.gaze_pitch is not None]
        for i in range(len(pitches) - 1):
            assert pitches[i] <= pitches[i + 1], f"Not monotonic at index {i}: {pitches}"

    def test_empty_input(self):
        """Empty input returns empty output."""
        result = gaze_rotation([], axis="yaw", range_start=-30.0, range_end=30.0)
        assert result == []

    def test_single_clip(self):
        """Single clip with gaze data is returned as-is."""
        clips = [_make_pair("a", yaw=5.0)]
        result = gaze_rotation(clips, axis="yaw", range_start=-30.0, range_end=30.0)
        assert len(result) == 1
        assert result[0][0].id == "a"

    def test_all_without_gaze_returns_original_list(self):
        """When no clips have gaze data, returns all clips unchanged."""
        clips = [_make_pair("a"), _make_pair("b"), _make_pair("c")]
        result = gaze_rotation(
            clips, axis="yaw", range_start=-30.0, range_end=30.0, ascending=True,
        )
        assert len(result) == 3
        ids = [clip.id for clip, _ in result]
        assert ids == ["a", "b", "c"]

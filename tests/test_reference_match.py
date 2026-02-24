"""Tests for reference-guided matching algorithm."""

from pathlib import Path

import pytest

from models.clip import Source, Clip
from models.sequence import Sequence
from core.remix.reference_match import (
    extract_feature_vector,
    compute_normalizers,
    weighted_distance,
    reference_guided_match,
    get_active_dimensions_for_clips,
)


# --- Helpers ---

def _make_source(source_id="src-1", fps=30.0):
    return Source(id=source_id, file_path=Path("/test/video.mp4"), fps=fps)


def _make_clip(
    clip_id="clip-1",
    source_id="src-1",
    start_frame=0,
    end_frame=90,
    dominant_colors=None,
    average_brightness=None,
    shot_type=None,
    rms_volume=None,
    embedding=None,
):
    return Clip(
        id=clip_id,
        source_id=source_id,
        start_frame=start_frame,
        end_frame=end_frame,
        dominant_colors=dominant_colors,
        average_brightness=average_brightness,
        shot_type=shot_type,
        rms_volume=rms_volume,
        embedding=embedding,
    )


# --- Feature Vector Extraction ---

class TestExtractFeatureVector:
    def test_extracts_duration(self):
        clip = _make_clip(start_frame=0, end_frame=90)
        source = _make_source(fps=30.0)
        vec = extract_feature_vector(clip, source, ["duration"])
        assert vec["duration"] == pytest.approx(3.0)

    def test_extracts_brightness(self):
        clip = _make_clip(average_brightness=0.7)
        source = _make_source()
        vec = extract_feature_vector(clip, source, ["brightness"])
        assert vec["brightness"] == pytest.approx(0.7)

    def test_skips_missing_data(self):
        clip = _make_clip()  # No colors, no brightness
        source = _make_source()
        vec = extract_feature_vector(clip, source, ["color", "brightness"])
        assert "color" not in vec
        assert "brightness" not in vec

    def test_extracts_shot_scale_from_shot_type(self):
        clip = _make_clip(shot_type="close-up")
        source = _make_source()
        vec = extract_feature_vector(clip, source, ["shot_scale"])
        assert vec["shot_scale"] == pytest.approx(0.7)  # 7.0/10.0

    def test_extracts_embedding(self):
        emb = [0.1] * 768
        clip = _make_clip(embedding=emb)
        source = _make_source()
        vec = extract_feature_vector(clip, source, ["embedding"])
        assert vec["embedding"] == emb


# --- Normalization ---

class TestComputeNormalizers:
    def test_min_max_range(self):
        vectors = [
            {"duration": 1.0, "brightness": 0.2},
            {"duration": 5.0, "brightness": 0.8},
            {"duration": 3.0, "brightness": 0.5},
        ]
        normalizers = compute_normalizers(vectors, ["duration", "brightness"])
        assert normalizers["duration"] == (1.0, 5.0)
        assert normalizers["brightness"] == (0.2, 0.8)

    def test_single_value_dimension(self):
        vectors = [{"duration": 3.0}, {"duration": 3.0}]
        normalizers = compute_normalizers(vectors, ["duration"])
        assert normalizers["duration"] == (3.0, 3.0)

    def test_skips_embedding_dimensions(self):
        vectors = [{"embedding": [0.1, 0.2]}]
        normalizers = compute_normalizers(vectors, ["embedding"])
        assert "embedding" not in normalizers

    def test_skips_categorical_dimensions(self):
        vectors = [{"movement": "pan"}]
        normalizers = compute_normalizers(vectors, ["movement"])
        assert "movement" not in normalizers


# --- Weighted Distance ---

class TestWeightedDistance:
    def test_identical_vectors_zero_distance(self):
        vec = {"brightness": 0.5, "duration": 3.0}
        normalizers = {"brightness": (0.0, 1.0), "duration": (1.0, 5.0)}
        weights = {"brightness": 1.0, "duration": 1.0}
        dist = weighted_distance(vec, vec, weights, normalizers)
        assert dist == pytest.approx(0.0)

    def test_opposite_values_max_distance(self):
        ref = {"brightness": 0.0}
        user = {"brightness": 1.0}
        normalizers = {"brightness": (0.0, 1.0)}
        weights = {"brightness": 1.0}
        dist = weighted_distance(ref, user, weights, normalizers)
        assert dist == pytest.approx(1.0)

    def test_equal_weights_equal_influence(self):
        """Two dimensions with equal weights should contribute equally."""
        ref = {"brightness": 0.0, "duration": 1.0}
        user_bright_match = {"brightness": 0.1, "duration": 5.0}
        user_dur_match = {"brightness": 1.0, "duration": 1.5}

        normalizers = {"brightness": (0.0, 1.0), "duration": (1.0, 5.0)}
        weights = {"brightness": 1.0, "duration": 1.0}

        dist_bright = weighted_distance(ref, user_bright_match, weights, normalizers)
        dist_dur = weighted_distance(ref, user_dur_match, weights, normalizers)

        # bright_match: brightness dist=0.1, duration dist=1.0 -> avg=0.55
        # dur_match: brightness dist=1.0, duration dist=0.125 -> avg=0.5625
        # Both should be in similar range, neither dominated
        assert dist_bright < 1.0
        assert dist_dur < 1.0

    def test_zero_weight_ignored(self):
        ref = {"brightness": 0.0, "duration": 1.0}
        user = {"brightness": 1.0, "duration": 1.0}
        normalizers = {"brightness": (0.0, 1.0), "duration": (1.0, 5.0)}
        weights = {"brightness": 0.0, "duration": 1.0}
        dist = weighted_distance(ref, user, weights, normalizers)
        # Only duration matters, and it's identical
        assert dist == pytest.approx(0.0)

    def test_categorical_exact_match(self):
        ref = {"movement": "pan"}
        user = {"movement": "pan"}
        weights = {"movement": 1.0}
        dist = weighted_distance(ref, user, weights, {})
        assert dist == pytest.approx(0.0)

    def test_categorical_mismatch(self):
        ref = {"movement": "pan"}
        user = {"movement": "tilt"}
        weights = {"movement": 1.0}
        dist = weighted_distance(ref, user, weights, {})
        assert dist == pytest.approx(1.0)

    def test_missing_dimension_in_one_vector(self):
        ref = {"brightness": 0.5, "color": 0.3}
        user = {"brightness": 0.5}  # No color data
        normalizers = {"brightness": (0.0, 1.0), "color": (0.0, 1.0)}
        weights = {"brightness": 1.0, "color": 1.0}
        dist = weighted_distance(ref, user, weights, normalizers)
        # color is skipped, only brightness counted
        assert dist == pytest.approx(0.0)

    def test_all_weights_zero_returns_inf(self):
        ref = {"brightness": 0.5}
        user = {"brightness": 0.5}
        weights = {"brightness": 0.0}
        dist = weighted_distance(ref, user, weights, {})
        assert dist == float("inf")


# --- Reference-Guided Match ---

class TestReferenceGuidedMatch:
    def test_basic_matching_by_brightness(self):
        source = _make_source()

        # Reference clips: dark, medium, bright
        ref_clips = [
            (_make_clip("r1", average_brightness=0.2), source),
            (_make_clip("r2", average_brightness=0.5), source),
            (_make_clip("r3", average_brightness=0.8), source),
        ]

        # User clips: bright, dark, medium (different order)
        user_source = _make_source("src-2")
        user_clips = [
            (_make_clip("u1", source_id="src-2", average_brightness=0.85), user_source),
            (_make_clip("u2", source_id="src-2", average_brightness=0.15), user_source),
            (_make_clip("u3", source_id="src-2", average_brightness=0.55), user_source),
        ]

        weights = {"brightness": 1.0}
        result = reference_guided_match(ref_clips, user_clips, weights)

        # Should match: r1(0.2)->u2(0.15), r2(0.5)->u3(0.55), r3(0.8)->u1(0.85)
        result_ids = [clip.id for clip, _ in result]
        assert result_ids == ["u2", "u3", "u1"]

    def test_no_repeats_uses_each_clip_once(self):
        source = _make_source()
        user_source = _make_source("src-2")

        ref_clips = [
            (_make_clip("r1", average_brightness=0.5), source),
            (_make_clip("r2", average_brightness=0.5), source),
        ]
        # Only one user clip matches well
        user_clips = [
            (_make_clip("u1", source_id="src-2", average_brightness=0.5), user_source),
            (_make_clip("u2", source_id="src-2", average_brightness=0.1), user_source),
        ]

        weights = {"brightness": 1.0}
        result = reference_guided_match(ref_clips, user_clips, weights, allow_repeats=False)

        result_ids = [clip.id for clip, _ in result]
        # u1 matched first, u2 gets second position
        assert len(result_ids) == 2
        assert len(set(result_ids)) == 2  # No duplicates

    def test_allow_repeats(self):
        source = _make_source()
        user_source = _make_source("src-2")

        ref_clips = [
            (_make_clip("r1", average_brightness=0.5), source),
            (_make_clip("r2", average_brightness=0.5), source),
        ]
        user_clips = [
            (_make_clip("u1", source_id="src-2", average_brightness=0.5), user_source),
            (_make_clip("u2", source_id="src-2", average_brightness=0.1), user_source),
        ]

        weights = {"brightness": 1.0}
        result = reference_guided_match(ref_clips, user_clips, weights, allow_repeats=True)

        result_ids = [clip.id for clip, _ in result]
        assert result_ids == ["u1", "u1"]  # Same clip used twice

    def test_more_ref_than_user_clips(self):
        source = _make_source()
        user_source = _make_source("src-2")

        ref_clips = [
            (_make_clip("r1", average_brightness=0.3), source),
            (_make_clip("r2", average_brightness=0.5), source),
            (_make_clip("r3", average_brightness=0.7), source),
        ]
        user_clips = [
            (_make_clip("u1", source_id="src-2", average_brightness=0.5), user_source),
        ]

        weights = {"brightness": 1.0}
        result = reference_guided_match(ref_clips, user_clips, weights, allow_repeats=False)

        # Only 1 user clip available, so only 1 position filled
        assert len(result) == 1

    def test_empty_clips_returns_empty(self):
        result = reference_guided_match([], [], {"brightness": 1.0})
        assert result == []

    def test_all_weights_zero_returns_empty(self):
        source = _make_source()
        ref_clips = [(_make_clip("r1"), source)]
        user_clips = [(_make_clip("u1", source_id="src-2"), _make_source("src-2"))]

        result = reference_guided_match(ref_clips, user_clips, {"brightness": 0.0})
        assert result == []

    def test_multi_dimension_matching(self):
        source = _make_source()
        user_source = _make_source("src-2")

        ref_clips = [
            (_make_clip("r1", average_brightness=0.8, end_frame=150), source),  # 5 sec bright
        ]
        # u1: bright but short, u2: dim but right duration, u3: bright and right duration
        user_clips = [
            (_make_clip("u1", source_id="src-2", average_brightness=0.85, end_frame=30), user_source),
            (_make_clip("u2", source_id="src-2", average_brightness=0.2, end_frame=150), user_source),
            (_make_clip("u3", source_id="src-2", average_brightness=0.75, end_frame=150), user_source),
        ]

        # Equal weight to brightness and duration
        weights = {"brightness": 1.0, "duration": 1.0}
        result = reference_guided_match(ref_clips, user_clips, weights)

        # u3 is best: close brightness AND close duration
        assert result[0][0].id == "u3"


# --- Sequence Model Serialization ---

class TestSequenceModelExtensions:
    def test_to_dict_includes_reference_fields(self):
        seq = Sequence(
            algorithm="reference_guided",
            reference_source_id="src-ref",
            dimension_weights={"brightness": 0.8, "color": 0.5},
            allow_repeats=True,
        )
        data = seq.to_dict()
        assert data["algorithm"] == "reference_guided"
        assert data["reference_source_id"] == "src-ref"
        assert data["dimension_weights"] == {"brightness": 0.8, "color": 0.5}
        assert data["allow_repeats"] is True

    def test_from_dict_round_trip(self):
        seq = Sequence(
            algorithm="reference_guided",
            reference_source_id="src-ref",
            dimension_weights={"brightness": 0.8},
            allow_repeats=True,
        )
        data = seq.to_dict()
        restored = Sequence.from_dict(data)

        assert restored.algorithm == "reference_guided"
        assert restored.reference_source_id == "src-ref"
        assert restored.dimension_weights == {"brightness": 0.8}
        assert restored.allow_repeats is True

    def test_from_dict_backward_compatible(self):
        """Old project files without reference fields should load fine."""
        data = {
            "id": "seq-1",
            "name": "Test",
            "fps": 30.0,
            "tracks": [],
            "algorithm": "color",
        }
        seq = Sequence.from_dict(data)
        assert seq.algorithm == "color"
        assert seq.reference_source_id is None
        assert seq.dimension_weights is None
        assert seq.allow_repeats is False


# --- Active Dimensions Detection ---

class TestGetActiveDimensions:
    def test_duration_always_available(self):
        clips = [_make_clip()]
        dims = get_active_dimensions_for_clips(clips)
        assert "duration" in dims

    def test_color_available_when_clips_have_colors(self):
        clips = [_make_clip(dominant_colors=[(255, 0, 0)])]
        dims = get_active_dimensions_for_clips(clips)
        assert "color" in dims

    def test_color_not_available_without_data(self):
        clips = [_make_clip()]
        dims = get_active_dimensions_for_clips(clips)
        assert "color" not in dims

    def test_brightness_available_when_analyzed(self):
        clips = [_make_clip(average_brightness=0.5)]
        dims = get_active_dimensions_for_clips(clips)
        assert "brightness" in dims


# --- Cost Estimation Override ---

class TestCostEstimationOverride:
    def test_override_required_bypasses_config(self):
        """override_required should use explicit ops, not algorithm config."""
        from core.cost_estimates import estimate_sequence_cost

        # reference_guided has required_analysis=[] in config
        # Without override, should return nothing
        clip = _make_clip(average_brightness=None)
        result = estimate_sequence_cost("reference_guided", [clip])
        assert result == []

        # With override, should detect clips needing brightness analysis
        result = estimate_sequence_cost(
            "reference_guided",
            [clip],
            override_required=["brightness"],
        )
        assert len(result) == 1
        assert result[0].operation == "brightness"
        assert result[0].clips_needing == 1

    def test_override_required_skips_already_analyzed(self):
        """Clips with existing data should not count as needing analysis."""
        from core.cost_estimates import estimate_sequence_cost

        clip = _make_clip(average_brightness=0.5)
        result = estimate_sequence_cost(
            "reference_guided",
            [clip],
            override_required=["brightness"],
        )
        assert result == []  # Already has brightness

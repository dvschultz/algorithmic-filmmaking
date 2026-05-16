"""Tests for the 6 new sequencer algorithms.

Tests cover:
- Brightness gradient sorting
- Volume gradient sorting (with exclusion)
- Proximity sorting (dual mapping)
- Similarity chain (greedy nearest-neighbor)
- Match cut chain (boundary frame similarity + 2-opt)
- Color complementary/rainbow (purity filter + hue sort)
- Auto-analysis helpers
- Clip model field serialization
"""

from pathlib import Path

import pytest

from models.clip import Clip, Source
from models.cinematography import CinematographyAnalysis


# -- Helpers ------------------------------------------------------------------

def _make_source(source_id: str = "src1", fps: float = 30.0) -> Source:
    return Source(id=source_id, file_path=Path("/video.mp4"), fps=fps)


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


# -- Clip Model Serialization ------------------------------------------------

class TestClipModelFields:
    """New sequencer fields serialize/deserialize correctly."""

    def test_average_brightness_round_trip(self):
        clip = _make_clip("c1", average_brightness=0.75)
        data = clip.to_dict()
        assert data["average_brightness"] == 0.75
        restored = Clip.from_dict(data)
        assert restored.average_brightness == 0.75

    def test_rms_volume_round_trip(self):
        clip = _make_clip("c1", rms_volume=-23.4)
        data = clip.to_dict()
        assert data["rms_volume"] == -23.4
        restored = Clip.from_dict(data)
        assert restored.rms_volume == -23.4

    def test_embedding_round_trip(self):
        emb = [0.1] * 512
        clip = _make_clip("c1", embedding=emb)
        data = clip.to_dict()
        assert len(data["embedding"]) == 512
        restored = Clip.from_dict(data)
        assert len(restored.embedding) == 512
        assert restored.embedding[0] == pytest.approx(0.1)

    def test_boundary_embeddings_round_trip(self):
        first_emb = [0.2] * 512
        last_emb = [0.3] * 512
        clip = _make_clip("c1", first_frame_embedding=first_emb, last_frame_embedding=last_emb)
        data = clip.to_dict()
        assert "first_frame_embedding" in data
        assert "last_frame_embedding" in data
        restored = Clip.from_dict(data)
        assert restored.first_frame_embedding[0] == pytest.approx(0.2)
        assert restored.last_frame_embedding[0] == pytest.approx(0.3)

    def test_none_fields_not_serialized(self):
        clip = _make_clip("c1")
        data = clip.to_dict()
        assert "average_brightness" not in data
        assert "rms_volume" not in data
        assert "embedding" not in data
        assert "first_frame_embedding" not in data
        assert "last_frame_embedding" not in data


# -- Brightness Algorithm ---------------------------------------------------

class TestBrightnessAlgorithm:
    """Brightness gradient sorting."""

    def test_bright_to_dark_order(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("dark", average_brightness=0.1), source),
            (_make_clip("mid", average_brightness=0.5), source),
            (_make_clip("bright", average_brightness=0.9), source),
        ]
        result = generate_sequence("brightness", clips, 3, direction="bright_to_dark")
        ids = [c.id for c, _ in result]
        assert ids == ["bright", "mid", "dark"]

    def test_dark_to_bright_order(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("bright", average_brightness=0.9), source),
            (_make_clip("dark", average_brightness=0.1), source),
            (_make_clip("mid", average_brightness=0.5), source),
        ]
        result = generate_sequence("brightness", clips, 3, direction="dark_to_bright")
        ids = [c.id for c, _ in result]
        assert ids == ["dark", "mid", "bright"]

    def test_default_brightness_for_missing(self):
        """Clips without brightness get 0.5 default (sorted to middle)."""
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("dark", average_brightness=0.1), source),
            (_make_clip("unknown", average_brightness=None), source),
            (_make_clip("bright", average_brightness=0.9), source),
        ]
        # Need to mock auto-compute since there's no real video
        # Set the value manually to simulate auto-compute fallback
        clips[1][0].average_brightness = 0.5
        result = generate_sequence("brightness", clips, 3, direction="bright_to_dark")
        ids = [c.id for c, _ in result]
        assert ids == ["bright", "unknown", "dark"]


# -- Volume Algorithm -------------------------------------------------------

class TestVolumeAlgorithm:
    """Volume gradient sorting with clip exclusion."""

    def test_quiet_to_loud_order(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("loud", rms_volume=-5.0), source),
            (_make_clip("quiet", rms_volume=-40.0), source),
            (_make_clip("mid", rms_volume=-20.0), source),
        ]
        result = generate_sequence("volume", clips, 3, direction="quiet_to_loud")
        ids = [c.id for c, _ in result]
        assert ids == ["quiet", "mid", "loud"]

    def test_loud_to_quiet_order(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("quiet", rms_volume=-40.0), source),
            (_make_clip("loud", rms_volume=-5.0), source),
        ]
        result = generate_sequence("volume", clips, 2, direction="loud_to_quiet")
        ids = [c.id for c, _ in result]
        assert ids == ["loud", "quiet"]

    def test_clips_without_volume_excluded(self):
        """Clips with rms_volume=None are excluded from the result."""
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("has_vol", rms_volume=-20.0), source),
            (_make_clip("no_vol", rms_volume=None), source),
            (_make_clip("has_vol2", rms_volume=-10.0), source),
        ]
        result = generate_sequence("volume", clips, 3, direction="quiet_to_loud")
        ids = [c.id for c, _ in result]
        assert "no_vol" not in ids
        assert len(ids) == 2


# -- Proximity Algorithm ----------------------------------------------------

class TestProximityAlgorithm:
    """Proximity sorting with dual mapping."""

    def test_far_to_close_with_shot_type(self):
        """Uses 5-class shot_type for proximity scoring."""
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("cu", shot_type="close-up"), source),
            (_make_clip("wide", shot_type="wide shot"), source),
            (_make_clip("med", shot_type="medium shot"), source),
        ]
        result = generate_sequence("proximity", clips, 3, direction="far_to_close")
        ids = [c.id for c, _ in result]
        assert ids == ["wide", "med", "cu"]

    def test_close_to_far_direction(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("wide", shot_type="wide shot"), source),
            (_make_clip("cu", shot_type="close-up"), source),
        ]
        result = generate_sequence("proximity", clips, 2, direction="close_to_far")
        ids = [c.id for c, _ in result]
        assert ids == ["cu", "wide"]

    def test_prefers_cinematography_over_shot_type(self):
        """10-class cinematography.shot_size takes priority."""
        from core.remix import generate_sequence
        source = _make_source()

        # Clip with both: cinematography says BCU (8.0), shot_type says medium (5.0)
        clip_both = _make_clip("both", shot_type="medium shot")
        clip_both.cinematography = CinematographyAnalysis(shot_size="BCU")

        clip_wide = _make_clip("wide", shot_type="wide shot")

        clips = [(clip_both, source), (clip_wide, source)]
        result = generate_sequence("proximity", clips, 2, direction="far_to_close")
        ids = [c.id for c, _ in result]
        # BCU (8.0) > wide shot (2.0), so wide first in far_to_close
        assert ids == ["wide", "both"]

    def test_unclassified_clips_sort_to_middle(self):
        """Clips without shot_type or cinematography get score 5.0."""
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("cu", shot_type="close-up"), source),   # 7.0
            (_make_clip("none"), source),                        # 5.0
            (_make_clip("wide", shot_type="wide shot"), source), # 2.0
        ]
        result = generate_sequence("proximity", clips, 3, direction="far_to_close")
        ids = [c.id for c, _ in result]
        assert ids == ["wide", "none", "cu"]


# -- Color Complementary/Rainbow Algorithm ----------------------------------

class TestColorComplementaryAlgorithm:
    """Color algorithm: purity filter + hue sorting (rainbow & complementary)."""

    def test_rainbow_sorts_by_hue(self):
        from core.remix import generate_sequence
        source = _make_source()
        # Pure red, pure green, pure blue
        clips = [
            (_make_clip("green", dominant_colors=[(0, 255, 0)]), source),
            (_make_clip("red", dominant_colors=[(255, 0, 0)]), source),
            (_make_clip("blue", dominant_colors=[(0, 0, 255)]), source),
        ]
        result = generate_sequence("color", clips, 3, direction="rainbow")
        ids = [c.id for c, _ in result]
        # Red hue ~0, Green hue ~120, Blue hue ~240
        assert ids == ["red", "green", "blue"]

    def test_low_purity_clips_included(self):
        """All clips with color data are included regardless of purity."""
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("pure", dominant_colors=[(255, 0, 0)]), source),
            # Gray has zero saturation but still has color data
            (_make_clip("gray", dominant_colors=[(128, 128, 128)]), source),
        ]
        result = generate_sequence("color", clips, 2, direction="rainbow", no_color_handling="append_end")
        ids = [c.id for c, _ in result]
        assert "gray" in ids
        assert "pure" in ids

    def test_complementary_interleaves(self):
        """Complementary direction alternates from opposite ends."""
        from core.remix import generate_sequence
        source = _make_source()
        # 4 clips with distinct hues
        clips = [
            (_make_clip("red", dominant_colors=[(255, 0, 0)]), source),    # ~0°
            (_make_clip("yellow", dominant_colors=[(255, 255, 0)]), source), # ~60°
            (_make_clip("cyan", dominant_colors=[(0, 255, 255)]), source),   # ~180°
            (_make_clip("blue", dominant_colors=[(0, 0, 255)]), source),    # ~240°
        ]
        result = generate_sequence("color", clips, 4, direction="complementary", no_color_handling="append_end")
        ids = [c.id for c, _ in result]
        # Sorted by hue: [red, yellow, cyan, blue]
        # Interleaved: red(lo), blue(hi), yellow(lo), cyan(hi)
        assert ids == ["red", "blue", "yellow", "cyan"]

    def test_clips_without_colors_appended_at_end(self):
        """Clips without color data are appended after sorted clips."""
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("has_color", dominant_colors=[(255, 0, 0)]), source),
            (_make_clip("no_color"), source),
        ]
        result = generate_sequence("color", clips, 2, direction="rainbow", no_color_handling="append_end")
        ids = [c.id for c, _ in result]
        assert ids == ["has_color", "no_color"]


# -- Color Purity Computation -----------------------------------------------

class TestColorPurity:
    """compute_color_purity function."""

    def test_pure_red_high_purity(self):
        from core.analysis.color import compute_color_purity
        # Single pure red → high saturation, no variance
        purity = compute_color_purity([(255, 0, 0)])
        assert purity > 0.8

    def test_gray_zero_purity(self):
        from core.analysis.color import compute_color_purity
        purity = compute_color_purity([(128, 128, 128)])
        assert purity < 0.1

    def test_mixed_colors_lower_purity(self):
        from core.analysis.color import compute_color_purity
        # Red + cyan = high hue variance
        purity = compute_color_purity([(255, 0, 0), (0, 255, 255)])
        pure_purity = compute_color_purity([(255, 0, 0), (200, 0, 0)])
        assert purity < pure_purity

    def test_empty_returns_zero(self):
        from core.analysis.color import compute_color_purity
        assert compute_color_purity([]) == 0.0


# -- Similarity Chain Algorithm ---------------------------------------------

class TestSimilarityChain:
    """Greedy nearest-neighbor visual similarity chain."""

    def _make_embedding(self, value: float) -> list[float]:
        """Create a simple 512-dim embedding with a dominant value."""
        import numpy as np
        emb = np.zeros(512, dtype=np.float32)
        emb[0] = value
        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.tolist()

    def test_chain_follows_similarity(self):
        from core.remix.similarity_chain import similarity_chain
        source = _make_source()

        # Create 3 clips with embeddings that form a clear chain
        # A→B is closer than A→C
        import numpy as np
        emb_a = np.zeros(512, dtype=np.float32)
        emb_a[0] = 1.0
        emb_b = np.zeros(512, dtype=np.float32)
        emb_b[0] = 0.9
        emb_b[1] = 0.1
        emb_c = np.zeros(512, dtype=np.float32)
        emb_c[1] = 1.0

        # Normalize
        for emb in [emb_a, emb_b, emb_c]:
            emb /= np.linalg.norm(emb)

        clip_a = _make_clip("a", embedding=emb_a.tolist())
        clip_b = _make_clip("b", embedding=emb_b.tolist())
        clip_c = _make_clip("c", embedding=emb_c.tolist())

        clips = [(clip_a, source), (clip_c, source), (clip_b, source)]
        result = similarity_chain(clips, start_clip_id="a")
        ids = [c.id for c, _ in result]
        # Starting from A, B is most similar to A, then C
        assert ids[0] == "a"
        assert ids[1] == "b"
        assert ids[2] == "c"

    def test_chain_uses_all_clips(self):
        from core.remix.similarity_chain import similarity_chain
        source = _make_source()
        import numpy as np

        clips = []
        for i in range(5):
            emb = np.random.randn(512).astype(np.float32)
            emb /= np.linalg.norm(emb)
            clips.append((_make_clip(f"c{i}", embedding=emb.tolist()), source))

        result = similarity_chain(clips)
        assert len(result) == 5

    def test_clips_without_embeddings_appended(self):
        from core.remix.similarity_chain import similarity_chain
        source = _make_source()
        import numpy as np

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        clips = [
            (_make_clip("with_emb", embedding=emb.tolist()), source),
            (_make_clip("no_emb"), source),
        ]
        result = similarity_chain(clips)
        assert len(result) == 2
        # Clip without embedding should be at the end
        assert result[-1][0].id == "no_emb"

    def test_single_clip_returns_same(self):
        from core.remix.similarity_chain import similarity_chain
        source = _make_source()
        clips = [(_make_clip("only"), source)]
        result = similarity_chain(clips)
        assert len(result) == 1
        assert result[0][0].id == "only"


# -- Match Cut Algorithm ----------------------------------------------------

class TestMatchCut:
    """Match cut chain with boundary frame embeddings."""

    def test_chain_uses_boundary_embeddings(self):
        from core.remix.match_cut import match_cut_chain
        source = _make_source()
        import numpy as np

        # Clip A's last frame is similar to clip B's first frame
        emb_a_first = np.zeros(512, dtype=np.float32)
        emb_a_first[0] = 1.0
        emb_a_last = np.zeros(512, dtype=np.float32)
        emb_a_last[1] = 1.0  # Similar to B's first

        emb_b_first = np.zeros(512, dtype=np.float32)
        emb_b_first[1] = 0.95
        emb_b_first[2] = 0.05
        emb_b_last = np.zeros(512, dtype=np.float32)
        emb_b_last[2] = 1.0  # Similar to C's first

        emb_c_first = np.zeros(512, dtype=np.float32)
        emb_c_first[2] = 0.9
        emb_c_first[3] = 0.1
        emb_c_last = np.zeros(512, dtype=np.float32)
        emb_c_last[3] = 1.0

        # Normalize all
        for emb in [emb_a_first, emb_a_last, emb_b_first, emb_b_last, emb_c_first, emb_c_last]:
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm

        clip_a = _make_clip("a",
                            first_frame_embedding=emb_a_first.tolist(),
                            last_frame_embedding=emb_a_last.tolist())
        clip_b = _make_clip("b",
                            first_frame_embedding=emb_b_first.tolist(),
                            last_frame_embedding=emb_b_last.tolist())
        clip_c = _make_clip("c",
                            first_frame_embedding=emb_c_first.tolist(),
                            last_frame_embedding=emb_c_last.tolist())

        clips = [(clip_a, source), (clip_c, source), (clip_b, source)]
        result = match_cut_chain(clips, start_clip_id="a", refine_iterations=0)
        ids = [c.id for c, _ in result]
        # A's last frame → B's first frame (most similar), then B → C
        assert ids == ["a", "b", "c"]

    def test_two_opt_improves_or_maintains_cost(self):
        """2-opt refinement should never increase total chain cost."""
        from core.remix.match_cut import match_cut_chain
        source = _make_source()
        import numpy as np

        np.random.seed(42)
        clips = []
        for i in range(10):
            first_emb = np.random.randn(512).astype(np.float32)
            first_emb /= np.linalg.norm(first_emb)
            last_emb = np.random.randn(512).astype(np.float32)
            last_emb /= np.linalg.norm(last_emb)
            clips.append((_make_clip(f"c{i}",
                                     first_frame_embedding=first_emb.tolist(),
                                     last_frame_embedding=last_emb.tolist()), source))

        # Run without refinement
        result_greedy = match_cut_chain(clips, refine_iterations=0)
        # Run with refinement
        result_refined = match_cut_chain(clips, refine_iterations=100)

        # Both should use all clips
        assert len(result_greedy) == 10
        assert len(result_refined) == 10

        # Compute transition costs: sum of cosine distance(last_frame[i], first_frame[j])
        def chain_cost(result):
            total = 0.0
            for k in range(len(result) - 1):
                last_emb = np.array(result[k][0].last_frame_embedding)
                first_emb = np.array(result[k + 1][0].first_frame_embedding)
                sim = np.dot(last_emb, first_emb)
                total += 1.0 - np.clip(sim, -1.0, 1.0)
            return total

        greedy_cost = chain_cost(result_greedy)
        refined_cost = chain_cost(result_refined)
        assert refined_cost <= greedy_cost + 1e-6

    def test_clips_without_boundary_embeddings_appended(self):
        from core.remix.match_cut import match_cut_chain
        source = _make_source()
        import numpy as np

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        clips = [
            (_make_clip("with_emb",
                        first_frame_embedding=emb.tolist(),
                        last_frame_embedding=emb.tolist()), source),
            (_make_clip("no_emb"), source),
        ]
        result = match_cut_chain(clips)
        assert len(result) == 2
        assert result[-1][0].id == "no_emb"


# -- Generate Sequence Integration ------------------------------------------

class TestGenerateSequenceIntegration:
    """Test generate_sequence() with all new algorithm keys."""

    def test_brightness_algorithm_key(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("a", average_brightness=0.9), source),
            (_make_clip("b", average_brightness=0.1), source),
        ]
        result = generate_sequence("brightness", clips, 2)
        assert len(result) == 2

    def test_volume_algorithm_key(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("a", rms_volume=-20.0), source),
            (_make_clip("b", rms_volume=-40.0), source),
        ]
        result = generate_sequence("volume", clips, 2)
        assert len(result) == 2

    def test_proximity_algorithm_key(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("a", shot_type="wide shot"), source),
            (_make_clip("b", shot_type="close-up"), source),
        ]
        result = generate_sequence("proximity", clips, 2)
        assert len(result) == 2

    def test_color_complementary_algorithm_key(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("a", dominant_colors=[(255, 0, 0)]), source),
            (_make_clip("b", dominant_colors=[(0, 0, 255)]), source),
        ]
        result = generate_sequence("color", clips, 2, direction="complementary")
        assert len(result) == 2

    def test_similarity_chain_algorithm_key(self):
        from core.remix import generate_sequence
        source = _make_source()
        import numpy as np

        clips = []
        for i in range(3):
            emb = np.random.randn(512).astype(np.float32)
            emb /= np.linalg.norm(emb)
            clips.append((_make_clip(f"c{i}", embedding=emb.tolist()), source))

        result = generate_sequence("similarity_chain", clips, 3)
        assert len(result) == 3

    def test_match_cut_algorithm_key(self):
        from core.remix import generate_sequence
        source = _make_source()
        import numpy as np

        clips = []
        for i in range(3):
            first_emb = np.random.randn(512).astype(np.float32)
            first_emb /= np.linalg.norm(first_emb)
            last_emb = np.random.randn(512).astype(np.float32)
            last_emb /= np.linalg.norm(last_emb)
            clips.append((_make_clip(
                f"c{i}",
                first_frame_embedding=first_emb.tolist(),
                last_frame_embedding=last_emb.tolist(),
            ), source))

        result = generate_sequence("match_cut", clips, 3)
        assert len(result) == 3

    def test_dialog_only_algorithm_key_raises_not_implemented(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("a"), source),
            (_make_clip("b"), source),
        ]

        with pytest.raises(NotImplementedError, match="dialog-only"):
            generate_sequence("storyteller", clips, 2)

    def test_unknown_algorithm_still_returns_clipped_input_order(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("a"), source),
            (_make_clip("b"), source),
            (_make_clip("c"), source),
        ]

        result = generate_sequence("unknown_algorithm", clips, 2)
        ids = [c.id for c, _ in result]
        assert ids == ["a", "b"]


# -- Deserialization Validation ------------------------------------------------

class TestClipDeserializationValidation:
    """Test that Clip.from_dict() validates new cache fields."""

    def test_valid_embedding_roundtrips(self):
        from models.clip import Clip
        emb = [0.1] * 512
        clip = Clip(embedding=emb, first_frame_embedding=emb, last_frame_embedding=emb)
        data = clip.to_dict()
        restored = Clip.from_dict(data)
        assert restored.embedding == emb
        assert restored.first_frame_embedding == emb
        assert restored.last_frame_embedding == emb

    def test_wrong_dimension_embedding_discarded(self):
        from models.clip import Clip
        data = {"embedding": [0.1] * 256}  # Wrong dimension
        clip = Clip.from_dict(data)
        assert clip.embedding is None

    def test_non_list_embedding_discarded(self):
        from models.clip import Clip
        data = {"embedding": "not_a_list"}
        clip = Clip.from_dict(data)
        assert clip.embedding is None

    def test_valid_float_fields_roundtrip(self):
        from models.clip import Clip
        clip = Clip(average_brightness=0.75, rms_volume=-20.5)
        data = clip.to_dict()
        restored = Clip.from_dict(data)
        assert restored.average_brightness == 0.75
        assert restored.rms_volume == -20.5

    def test_non_numeric_brightness_discarded(self):
        from models.clip import Clip
        data = {"average_brightness": "bright"}
        clip = Clip.from_dict(data)
        assert clip.average_brightness is None

    def test_non_numeric_volume_discarded(self):
        from models.clip import Clip
        data = {"rms_volume": [1, 2, 3]}
        clip = Clip.from_dict(data)
        assert clip.rms_volume is None

    def test_int_brightness_coerced_to_float(self):
        from models.clip import Clip
        data = {"average_brightness": 1}
        clip = Clip.from_dict(data)
        assert clip.average_brightness == 1.0
        assert isinstance(clip.average_brightness, float)


# -- Gaze Sort Algorithm -------------------------------------------------------

class TestGazeSortAlgorithm:
    """Gaze Sort: arrange clips by gaze direction."""

    def test_left_to_right_sorts_by_ascending_yaw(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("right", gaze_yaw=25.0, gaze_pitch=0.0, gaze_category="looking_right"), source),
            (_make_clip("left", gaze_yaw=-20.0, gaze_pitch=0.0, gaze_category="looking_left"), source),
            (_make_clip("center", gaze_yaw=2.0, gaze_pitch=0.0, gaze_category="at_camera"), source),
        ]
        result = generate_sequence("gaze_sort", clips, 3, direction="left_to_right")
        ids = [c.id for c, _ in result]
        assert ids == ["left", "center", "right"]

    def test_right_to_left_sorts_by_descending_yaw(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("left", gaze_yaw=-20.0, gaze_pitch=0.0, gaze_category="looking_left"), source),
            (_make_clip("right", gaze_yaw=25.0, gaze_pitch=0.0, gaze_category="looking_right"), source),
            (_make_clip("center", gaze_yaw=2.0, gaze_pitch=0.0, gaze_category="at_camera"), source),
        ]
        result = generate_sequence("gaze_sort", clips, 3, direction="right_to_left")
        ids = [c.id for c, _ in result]
        assert ids == ["right", "center", "left"]

    def test_up_to_down_sorts_by_ascending_pitch(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("down", gaze_yaw=0.0, gaze_pitch=15.0, gaze_category="looking_down"), source),
            (_make_clip("up", gaze_yaw=0.0, gaze_pitch=-18.0, gaze_category="looking_up"), source),
            (_make_clip("center", gaze_yaw=0.0, gaze_pitch=1.0, gaze_category="at_camera"), source),
        ]
        result = generate_sequence("gaze_sort", clips, 3, direction="up_to_down")
        ids = [c.id for c, _ in result]
        assert ids == ["up", "center", "down"]

    def test_down_to_up_sorts_by_descending_pitch(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("up", gaze_yaw=0.0, gaze_pitch=-18.0, gaze_category="looking_up"), source),
            (_make_clip("down", gaze_yaw=0.0, gaze_pitch=15.0, gaze_category="looking_down"), source),
        ]
        result = generate_sequence("gaze_sort", clips, 2, direction="down_to_up")
        ids = [c.id for c, _ in result]
        assert ids == ["down", "up"]

    def test_clips_without_gaze_appended_at_end(self):
        """Clips without gaze data are appended after sorted clips."""
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("no_gaze"), source),
            (_make_clip("right", gaze_yaw=25.0, gaze_pitch=0.0, gaze_category="looking_right"), source),
            (_make_clip("left", gaze_yaw=-20.0, gaze_pitch=0.0, gaze_category="looking_left"), source),
        ]
        result = generate_sequence("gaze_sort", clips, 3, direction="left_to_right")
        ids = [c.id for c, _ in result]
        assert ids == ["left", "right", "no_gaze"]

    def test_no_clips_have_gaze_returns_original_order(self):
        """When no clips have gaze data, return original order."""
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("a"), source),
            (_make_clip("b"), source),
            (_make_clip("c"), source),
        ]
        result = generate_sequence("gaze_sort", clips, 3, direction="left_to_right")
        ids = [c.id for c, _ in result]
        assert ids == ["a", "b", "c"]

    def test_default_direction_is_left_to_right(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("right", gaze_yaw=25.0, gaze_pitch=0.0, gaze_category="looking_right"), source),
            (_make_clip("left", gaze_yaw=-20.0, gaze_pitch=0.0, gaze_category="looking_left"), source),
        ]
        result = generate_sequence("gaze_sort", clips, 2)
        ids = [c.id for c, _ in result]
        assert ids == ["left", "right"]


# -- Gaze Consistency Algorithm ------------------------------------------------

class TestGazeConsistencyAlgorithm:
    """Gaze Consistency: group clips by matching gaze direction."""

    def test_groups_by_category_largest_first(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("r1", gaze_yaw=20.0, gaze_pitch=0.0, gaze_category="looking_right"), source),
            (_make_clip("l1", gaze_yaw=-15.0, gaze_pitch=0.0, gaze_category="looking_left"), source),
            (_make_clip("r2", gaze_yaw=30.0, gaze_pitch=0.0, gaze_category="looking_right"), source),
            (_make_clip("l2", gaze_yaw=-25.0, gaze_pitch=0.0, gaze_category="looking_left"), source),
            (_make_clip("r3", gaze_yaw=10.0, gaze_pitch=0.0, gaze_category="looking_right"), source),
        ]
        result = generate_sequence("gaze_consistency", clips, 5)
        ids = [c.id for c, _ in result]
        # 3 right clips come first (largest group), then 2 left clips
        assert ids[:3] == ["r3", "r1", "r2"]  # sorted by ascending yaw within group
        assert ids[3:] == ["l2", "l1"]  # sorted by ascending yaw within group

    def test_within_group_sorted_by_relevant_angle(self):
        """Left/right categories sort by yaw, up/down by pitch."""
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("d1", gaze_yaw=0.0, gaze_pitch=20.0, gaze_category="looking_down"), source),
            (_make_clip("d2", gaze_yaw=0.0, gaze_pitch=10.0, gaze_category="looking_down"), source),
            (_make_clip("u1", gaze_yaw=0.0, gaze_pitch=-15.0, gaze_category="looking_up"), source),
        ]
        result = generate_sequence("gaze_consistency", clips, 3)
        ids = [c.id for c, _ in result]
        # 2 down clips (largest group) sorted by ascending pitch, then 1 up clip
        assert ids == ["d2", "d1", "u1"]

    def test_clips_without_gaze_appended_at_end(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("no_gaze"), source),
            (_make_clip("r1", gaze_yaw=20.0, gaze_pitch=0.0, gaze_category="looking_right"), source),
        ]
        result = generate_sequence("gaze_consistency", clips, 2)
        ids = [c.id for c, _ in result]
        assert ids == ["r1", "no_gaze"]

    def test_no_clips_have_gaze_returns_original_order(self):
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("a"), source),
            (_make_clip("b"), source),
        ]
        result = generate_sequence("gaze_consistency", clips, 2)
        ids = [c.id for c, _ in result]
        assert ids == ["a", "b"]

    def test_all_same_category_stable_order_by_angle(self):
        """When all clips share a category, order by angle within that category."""
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("c", gaze_yaw=5.0, gaze_pitch=0.0, gaze_category="at_camera"), source),
            (_make_clip("a", gaze_yaw=-3.0, gaze_pitch=0.0, gaze_category="at_camera"), source),
            (_make_clip("b", gaze_yaw=1.0, gaze_pitch=0.0, gaze_category="at_camera"), source),
        ]
        result = generate_sequence("gaze_consistency", clips, 3)
        ids = [c.id for c, _ in result]
        # All at_camera, sorted by ascending yaw
        assert ids == ["a", "b", "c"]

    def test_mixed_with_and_without_gaze(self):
        """Gaze clips sorted by category, no-gaze clips appended."""
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("no1"), source),
            (_make_clip("r1", gaze_yaw=10.0, gaze_pitch=0.0, gaze_category="looking_right"), source),
            (_make_clip("no2"), source),
            (_make_clip("l1", gaze_yaw=-10.0, gaze_pitch=0.0, gaze_category="looking_left"), source),
        ]
        result = generate_sequence("gaze_consistency", clips, 4)
        ids = [c.id for c, _ in result]
        # Both categories have 1 clip each (equal size, sorted by first-encountered)
        # no-gaze clips at end
        assert ids[-2:] == ["no1", "no2"]
        assert set(ids[:2]) == {"r1", "l1"}

    def test_gaze_consistency_groups_by_category_largest_first(self):
        """gaze_consistency groups clips by category with the largest group first."""
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("r1", gaze_yaw=10.0, gaze_category="looking_right"), source),
            (_make_clip("l1", gaze_yaw=-20.0, gaze_category="looking_left"), source),
            (_make_clip("l2", gaze_yaw=-10.0, gaze_category="looking_left"), source),
            (_make_clip("r2", gaze_yaw=5.0, gaze_category="looking_right"), source),
            (_make_clip("l3", gaze_yaw=-5.0, gaze_category="looking_left"), source),
        ]
        result = generate_sequence("gaze_consistency", clips, 5)
        ids = [c.id for c, _ in result]
        # looking_left has 3 clips (largest group), so they come first
        assert ids[:3] == ["l1", "l2", "l3"]  # sorted by yaw ascending
        # looking_right has 2 clips
        assert ids[3:] == ["r2", "r1"]  # sorted by yaw ascending (5.0, 10.0)

    def test_gaze_consistency_without_gaze_appended(self):
        """Clips without gaze data are appended at the end."""
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("no_gaze"), source),
            (_make_clip("a1", gaze_yaw=5.0, gaze_category="at_camera"), source),
        ]
        result = generate_sequence("gaze_consistency", clips, 2)
        ids = [c.id for c, _ in result]
        assert ids == ["a1", "no_gaze"]

    def test_gaze_consistency_intra_group_sort(self):
        """Within a category group, clips are sorted by the relevant angle."""
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("u1", gaze_pitch=-15.0, gaze_category="looking_up"), source),
            (_make_clip("u2", gaze_pitch=-5.0, gaze_category="looking_up"), source),
            (_make_clip("u3", gaze_pitch=-25.0, gaze_category="looking_up"), source),
        ]
        result = generate_sequence("gaze_consistency", clips, 3)
        pitches = [c.gaze_pitch for c, _ in result]
        assert pitches == [-25.0, -15.0, -5.0]  # sorted ascending by pitch

    def test_gaze_sort_invalid_direction_raises(self):
        """gaze_sort raises ValueError for unrecognized direction strings."""
        from core.remix import generate_sequence
        source = _make_source()
        clips = [
            (_make_clip("a", gaze_yaw=5.0, gaze_category="at_camera"), source),
        ]
        with pytest.raises(ValueError, match="Unknown gaze_sort direction"):
            generate_sequence("gaze_sort", clips, 1, direction="bad_value")

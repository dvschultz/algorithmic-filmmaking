"""Tests for core.cost_estimates module."""

import pytest
from dataclasses import dataclass
from typing import Optional

from core.cost_estimates import (
    estimate_sequence_cost,
    estimate_intention_cost,
    TIERED_OPERATIONS,
    METADATA_CHECKS,
    OperationEstimate,
)


@dataclass
class MockClip:
    """Minimal clip mock with metadata fields used by METADATA_CHECKS."""

    dominant_colors: Optional[list] = None
    shot_type: Optional[str] = None
    cinematography: object = None
    extracted_texts: Optional[list] = None
    description: Optional[str] = None
    average_brightness: Optional[float] = None
    rms_volume: Optional[float] = None
    embedding: Optional[list] = None
    first_frame_embedding: Optional[list] = None
    last_frame_embedding: Optional[list] = None
    transcript: Optional[list] = None


@dataclass
class MockSettings:
    """Minimal settings mock for tier and parallelism fields."""

    description_model_tier: str = "cpu"
    shot_classifier_tier: str = "cpu"
    text_extraction_method: str = "tesseract"
    color_analysis_parallelism: int = 4
    description_parallelism: int = 3
    transcription_parallelism: int = 2
    local_model_parallelism: int = 1
    cinematography_batch_parallelism: int = 2


# --- No-analysis algorithms ---

class TestNoAnalysisAlgorithms:
    """Algorithms that require no analysis return empty estimates."""

    @pytest.mark.parametrize("algorithm", ["shuffle", "sequential", "duration"])
    def test_returns_empty(self, algorithm):
        clips = [MockClip() for _ in range(5)]
        result = estimate_sequence_cost(algorithm, clips)
        assert result == []

    def test_unknown_algorithm_returns_empty(self):
        clips = [MockClip() for _ in range(5)]
        result = estimate_sequence_cost("nonexistent_algo", clips)
        assert result == []

    def test_empty_clips_returns_empty(self):
        result = estimate_sequence_cost("storyteller", [])
        assert result == []


# --- Algorithms with required analysis ---

class TestStoryteller:
    """Storyteller requires 'describe' analysis."""

    def test_all_clips_need_description(self):
        clips = [MockClip() for _ in range(10)]
        result = estimate_sequence_cost("storyteller", clips)
        assert len(result) == 1
        est = result[0]
        assert est.operation == "describe"
        assert est.clips_needing == 10
        assert est.clips_total == 10

    def test_some_clips_have_description(self):
        clips = [MockClip() for _ in range(10)]
        clips[0].description = "A dog"
        clips[1].description = "A cat"
        clips[2].description = "A bird"
        result = estimate_sequence_cost("storyteller", clips)
        assert len(result) == 1
        assert result[0].clips_needing == 7

    def test_all_clips_have_description(self):
        clips = [MockClip(description=f"Desc {i}") for i in range(5)]
        result = estimate_sequence_cost("storyteller", clips)
        assert result == []

    def test_local_tier_is_free(self):
        clips = [MockClip() for _ in range(10)]
        result = estimate_sequence_cost("storyteller", clips, tier_overrides={"describe": "local"})
        assert result[0].cost_dollars == 0.0
        assert result[0].tier == "local"

    def test_cloud_tier_has_cost(self):
        clips = [MockClip() for _ in range(10)]
        result = estimate_sequence_cost("storyteller", clips, tier_overrides={"describe": "cloud"})
        assert result[0].cost_dollars > 0.0
        assert result[0].tier == "cloud"


class TestShotType:
    """shot_type requires 'shots' analysis."""

    def test_clips_with_shot_type_are_ready(self):
        clips = [MockClip(shot_type="wide") for _ in range(5)]
        result = estimate_sequence_cost("shot_type", clips)
        assert result == []

    def test_clips_with_cinematography_are_ready(self):
        """Cinematography analysis satisfies the 'shots' check."""
        clips = [MockClip(cinematography=object()) for _ in range(5)]
        result = estimate_sequence_cost("shot_type", clips)
        assert result == []

    def test_mixed_clips(self):
        clips = [
            MockClip(shot_type="wide"),
            MockClip(),
            MockClip(cinematography=object()),
            MockClip(),
        ]
        result = estimate_sequence_cost("shot_type", clips)
        assert len(result) == 1
        assert result[0].clips_needing == 2
        assert result[0].clips_total == 4


class TestColor:
    """Color algorithms require 'colors' analysis."""

    def test_color_needs_dominant_colors(self):
        clips = [MockClip() for _ in range(5)]
        result = estimate_sequence_cost("color", clips)
        assert len(result) == 1
        assert result[0].operation == "colors"
        assert result[0].clips_needing == 5

    def test_color_cycle_same_requirement(self):
        clips = [MockClip(dominant_colors=[(255, 0, 0)])] * 3 + [MockClip()] * 2
        result = estimate_sequence_cost("color_cycle", clips)
        assert len(result) == 1
        assert result[0].clips_needing == 2


class TestExquisiteCorpus:
    """Exquisite Corpus requires 'extract_text' analysis."""

    def test_needs_extracted_texts(self):
        clips = [MockClip() for _ in range(8)]
        result = estimate_sequence_cost("exquisite_corpus", clips)
        assert len(result) == 1
        assert result[0].operation == "extract_text"
        assert result[0].clips_needing == 8


class TestEmbeddings:
    """similarity_chain and match_cut require embedding operations."""

    def test_similarity_chain_needs_embeddings(self):
        clips = [MockClip() for _ in range(5)]
        result = estimate_sequence_cost("similarity_chain", clips)
        assert len(result) == 1
        assert result[0].operation == "embeddings"

    def test_match_cut_needs_boundary_embeddings(self):
        clips = [MockClip() for _ in range(5)]
        result = estimate_sequence_cost("match_cut", clips)
        assert len(result) == 1
        assert result[0].operation == "boundary_embeddings"


# --- Tier override behavior ---

class TestTierOverrides:
    """Tier overrides change cost and time estimates."""

    def test_override_local_to_cloud(self):
        clips = [MockClip() for _ in range(10)]
        local = estimate_sequence_cost("storyteller", clips, tier_overrides={"describe": "local"})
        cloud = estimate_sequence_cost("storyteller", clips, tier_overrides={"describe": "cloud"})
        assert local[0].tier == "local"
        assert cloud[0].tier == "cloud"
        assert local[0].cost_dollars == 0.0
        assert cloud[0].cost_dollars > 0.0

    def test_settings_tier_used_as_default(self):
        clips = [MockClip() for _ in range(5)]
        settings = MockSettings(description_model_tier="cloud")
        result = estimate_sequence_cost("storyteller", clips, settings=settings)
        assert result[0].tier == "cloud"
        assert result[0].cost_dollars > 0.0

    def test_override_takes_precedence_over_settings(self):
        clips = [MockClip() for _ in range(5)]
        settings = MockSettings(description_model_tier="cloud")
        result = estimate_sequence_cost(
            "storyteller", clips,
            tier_overrides={"describe": "local"},
            settings=settings,
        )
        assert result[0].tier == "local"


# --- Time estimation ---

class TestTimeEstimation:
    """Time estimates account for parallelism."""

    def test_parallelism_reduces_wall_time(self):
        clips = [MockClip() for _ in range(12)]
        settings_1 = MockSettings(description_parallelism=1)
        settings_3 = MockSettings(description_parallelism=3)
        result_1 = estimate_sequence_cost("storyteller", clips, settings=settings_1)
        result_3 = estimate_sequence_cost("storyteller", clips, settings=settings_3)
        assert result_1[0].time_seconds == pytest.approx(result_3[0].time_seconds * 3)

    def test_time_is_positive_for_needing_clips(self):
        clips = [MockClip() for _ in range(5)]
        result = estimate_sequence_cost("storyteller", clips)
        assert result[0].time_seconds > 0.0


# --- METADATA_CHECKS coverage ---

class TestMetadataChecks:
    """Each metadata check correctly detects presence/absence."""

    def test_colors_check(self):
        assert not METADATA_CHECKS["colors"](MockClip())
        assert METADATA_CHECKS["colors"](MockClip(dominant_colors=[(255, 0, 0)]))

    def test_shots_check(self):
        assert not METADATA_CHECKS["shots"](MockClip())
        assert METADATA_CHECKS["shots"](MockClip(shot_type="wide"))
        assert METADATA_CHECKS["shots"](MockClip(cinematography=object()))

    def test_extract_text_check(self):
        assert not METADATA_CHECKS["extract_text"](MockClip())
        assert METADATA_CHECKS["extract_text"](MockClip(extracted_texts=[object()]))

    def test_describe_check(self):
        assert not METADATA_CHECKS["describe"](MockClip())
        assert METADATA_CHECKS["describe"](MockClip(description="A dog"))

    def test_brightness_check(self):
        assert not METADATA_CHECKS["brightness"](MockClip())
        assert METADATA_CHECKS["brightness"](MockClip(average_brightness=0.5))

    def test_volume_check(self):
        assert not METADATA_CHECKS["volume"](MockClip())
        assert METADATA_CHECKS["volume"](MockClip(rms_volume=-20.0))

    def test_embeddings_check(self):
        assert not METADATA_CHECKS["embeddings"](MockClip())
        assert METADATA_CHECKS["embeddings"](MockClip(embedding=[0.1] * 512))

    def test_boundary_embeddings_check(self):
        assert not METADATA_CHECKS["boundary_embeddings"](MockClip())
        assert METADATA_CHECKS["boundary_embeddings"](
            MockClip(first_frame_embedding=[0.1] * 512)
        )

    def test_transcribe_check(self):
        assert not METADATA_CHECKS["transcribe"](MockClip())
        assert METADATA_CHECKS["transcribe"](MockClip(transcript=[object()]))


# --- Intention flow ---

class TestIntentionEstimate:
    """Intention flow assumes all clips need analysis."""

    def test_all_clips_need_analysis(self):
        result = estimate_intention_cost("storyteller", 20)
        assert len(result) == 1
        assert result[0].clips_needing == 20
        assert result[0].clips_total == 20

    def test_zero_clips_returns_empty(self):
        result = estimate_intention_cost("storyteller", 0)
        assert result == []

    def test_no_analysis_algorithm_returns_empty(self):
        result = estimate_intention_cost("shuffle", 20)
        assert result == []

    def test_tier_overrides_work(self):
        result = estimate_intention_cost(
            "storyteller", 10,
            tier_overrides={"describe": "cloud"},
        )
        assert result[0].tier == "cloud"
        assert result[0].cost_dollars > 0.0

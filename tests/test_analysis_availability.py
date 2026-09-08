"""Tests for operation availability checks used by analysis UI."""

from tests.conftest import make_test_clip
from core.analysis_availability import (
    clear_operation_results,
    compute_disabled_operations,
    compute_operation_need_counts,
)


def test_legacy_colors_remain_available_for_verified_recomputation():
    clip_a = make_test_clip(
        "c1",
        dominant_colors=[(1, 2, 3)],
        shot_type="wide",
        transcript_text="hello",
    )
    clip_b = make_test_clip(
        "c2",
        dominant_colors=[(4, 5, 6)],
        shot_type="close-up",
        transcript_text="world",
    )

    disabled = compute_disabled_operations(
        [clip_a, clip_b], ["colors", "shots", "transcribe"]
    )
    assert disabled == {"shots", "transcribe"}


def test_compute_disabled_operations_mixed_clips_keeps_option_enabled():
    clip_done = make_test_clip("done", dominant_colors=[(1, 2, 3)])
    clip_missing = make_test_clip("missing", dominant_colors=None)

    disabled = compute_disabled_operations([clip_done, clip_missing], ["colors"])
    assert disabled == set()

    counts = compute_operation_need_counts([clip_done, clip_missing], ["colors"])
    assert counts["colors"] == 2


def test_extract_text_empty_list_is_a_completed_observation():
    clip = make_test_clip("clip-1")
    clip.extracted_texts = []

    disabled = compute_disabled_operations([clip], ["extract_text"])
    assert disabled == {"extract_text"}


def test_embedding_projection_alone_does_not_establish_completion():
    clip_with = make_test_clip("with")
    clip_with.embedding = [0.1] * 768
    clip_without = make_test_clip("without")
    clip_without.embedding = None

    assert compute_disabled_operations([clip_with], ["embeddings"]) == set()
    assert compute_disabled_operations([clip_without], ["embeddings"]) == set()
    assert compute_disabled_operations(
        [clip_with, clip_without], ["embeddings"]
    ) == set()


def test_clear_operation_results_resets_selected_analysis_fields():
    clip = make_test_clip(
        "c1",
        dominant_colors=[(1, 2, 3)],
        shot_type="wide",
        transcript_text="hello",
    )
    clip.description = "old description"
    clip.description_model = "test-model"
    clip.description_frames = 1
    clip.embedding = [0.1] * 768
    clip.first_frame_embedding = [0.2] * 768
    clip.last_frame_embedding = [0.3] * 768
    clip.embedding_model = "dinov2-vit-b-14"

    cleared = clear_operation_results(
        [clip],
        ["colors", "describe", "embeddings"],
    )

    assert cleared == 3
    assert clip.dominant_colors is None
    assert clip.description is None
    assert clip.description_model is None
    assert clip.description_frames is None
    assert clip.embedding is None
    assert clip.first_frame_embedding is None
    assert clip.last_frame_embedding is None
    assert clip.embedding_model is None
    assert clip.shot_type == "wide"
    assert clip.transcript is not None


def test_clear_operation_results_ignores_custom_query_results():
    clip = make_test_clip("c1")
    clip.custom_queries = [{"query": "blue flower", "match": True}]

    cleared = clear_operation_results([clip], ["custom_query"])

    assert cleared == 0
    assert clip.custom_queries == [{"query": "blue flower", "match": True}]

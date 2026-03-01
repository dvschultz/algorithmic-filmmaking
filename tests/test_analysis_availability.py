"""Tests for operation availability checks used by analysis UI."""

from tests.conftest import make_test_clip
from core.analysis_availability import (
    compute_disabled_operations,
    compute_operation_need_counts,
)


def test_compute_disabled_operations_all_clips_complete():
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
    assert disabled == {"colors", "shots", "transcribe"}


def test_compute_disabled_operations_mixed_clips_keeps_option_enabled():
    clip_done = make_test_clip("done", dominant_colors=[(1, 2, 3)])
    clip_missing = make_test_clip("missing", dominant_colors=None)

    disabled = compute_disabled_operations([clip_done, clip_missing], ["colors"])
    assert disabled == set()

    counts = compute_operation_need_counts([clip_done, clip_missing], ["colors"])
    assert counts["colors"] == 1


def test_extract_text_empty_list_still_counts_as_needing_analysis():
    clip = make_test_clip("clip-1")
    clip.extracted_texts = []

    disabled = compute_disabled_operations([clip], ["extract_text"])
    assert disabled == set()

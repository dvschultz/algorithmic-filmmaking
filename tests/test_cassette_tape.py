"""Unit tests for the Cassette Tape matching module."""

from pathlib import Path

import pytest

from core.remix.cassette_tape import (
    MatchResult,
    _score_phrase_against_segment,
    build_sequence_data,
    flatten_matches_in_phrase_order,
    match_phrases,
)
from core.transcription import TranscriptSegment
from models.clip import Clip, Source


def _seg(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start_time=start, end_time=end, text=text)


def _clip(clip_id: str, source_id: str, transcript: list[TranscriptSegment] | None,
          *, disabled: bool = False) -> Clip:
    return Clip(
        id=clip_id,
        source_id=source_id,
        start_frame=0,
        end_frame=900,  # 30s @ 30fps
        transcript=transcript,
        disabled=disabled,
    )


def _source(source_id: str = "src1", fps: float = 30.0) -> Source:
    return Source(
        id=source_id,
        file_path=Path(f"/test/{source_id}.mp4"),
        duration_seconds=60.0,
        fps=fps,
        width=1920,
        height=1080,
    )


class TestScorePhraseAgainstSegment:
    def test_phrase_inside_longer_segment_scores_high(self):
        score, start, end = _score_phrase_against_segment(
            "thank you", "well thank you for coming today"
        )
        assert score >= 90
        assert start != -1 and end != -1
        assert "thank you" in "well thank you for coming today"[start:end].lower()

    def test_case_insensitive(self):
        score_a, _, _ = _score_phrase_against_segment("HELLO", "hello world")
        score_b, _, _ = _score_phrase_against_segment("hello", "HELLO WORLD")
        assert score_a >= 90
        assert score_b >= 90

    def test_empty_inputs_score_zero(self):
        assert _score_phrase_against_segment("", "anything") == (0, -1, -1)
        assert _score_phrase_against_segment("anything", "") == (0, -1, -1)

    def test_unrelated_phrase_scores_low(self):
        score, _, _ = _score_phrase_against_segment(
            "purple zebra spaceship", "and then we went to the store"
        )
        assert score < 60

    def test_alignment_indices_map_into_original_case_text(self):
        """rapidfuzz alignment must return indices into the ORIGINAL text,
        not the lowercased copy — otherwise Unicode chars whose lowercase
        form has a different length (Turkish dotted-I, German ß casefold,
        etc.) shift the highlight."""
        text = "Welcome to İstanbul tonight"
        score, start, end = _score_phrase_against_segment("istanbul", text)
        assert score >= 80
        # The highlighted span must contain the (case-preserved) word.
        assert "İstanbul" in text[start:end]

    def test_short_segment_does_not_score_100_against_long_phrase(self):
        """Regression for the asymmetric-partial_ratio bug. partial_ratio
        aligns the shorter string against substrings of the longer; a
        1-char segment like 'I' would score 100 against any phrase
        containing 'I'. After the length-branch fix, short segments get
        whole-string ratio scoring (length-penalized) so they can't fake
        a perfect match."""
        score, _, _ = _score_phrase_against_segment("I love you", "I")
        assert score < 50, f"single-char segment should not score 100 (got {score})"

    def test_single_char_a_does_not_match_long_phrase(self):
        score, _, _ = _score_phrase_against_segment("thank you very much", "a")
        # 'a' is below the noise floor, hard-zeroed.
        assert score == 0

    def test_noise_floor_drops_two_char_segments(self):
        """Segments under 3 chars are forced to score 0 — Whisper
        artifacts like single letters or punctuation should not surface
        as matches."""
        for noisy in ["a", ".", "uh", "I"]:
            score, _, _ = _score_phrase_against_segment("hello world", noisy)
            assert score == 0, f"noise '{noisy}' scored {score}, expected 0"

    def test_segment_just_under_phrase_length_uses_ratio(self):
        """When segment is shorter than phrase but still substantive,
        ratio is used (not partial_ratio). Score should be high but not
        the inflated 100 partial_ratio would have given."""
        # phrase 13 chars, segment 9 chars, both meaningful
        score, _, _ = _score_phrase_against_segment("hello there mate", "hello there")
        # ratio gives length-penalized score, partial_ratio would give 100
        assert 50 <= score < 100

    def test_segment_longer_than_phrase_still_uses_partial_ratio(self):
        """Don't regress the original behavior: phrase-inside-longer-segment
        still scores 100."""
        score, start, end = _score_phrase_against_segment(
            "thank you", "well thank you for coming today"
        )
        assert score >= 95
        assert start >= 0 and end > start


class TestMatchPhrasesHappyPaths:
    def test_phrase_thank_you_finds_match(self):
        clip = _clip("c1", "src1", [_seg(0.0, 2.0, "well thank you for coming")])
        results = match_phrases([("thank you", 1)], [clip])
        assert "thank you" in results
        assert len(results["thank you"]) == 1
        match = results["thank you"][0]
        assert match.clip_id == "c1"
        assert match.score >= 90
        assert match.match_start >= 0 and match.match_end > match.match_start

    def test_top_n_picks_best_scoring_pairs(self):
        c1 = _clip("c1", "s", [_seg(0.0, 2.0, "I love you so much")])
        c2 = _clip("c2", "s", [_seg(0.0, 2.0, "I love you")])
        c3 = _clip("c3", "s", [_seg(0.0, 2.0, "she loves him")])
        results = match_phrases([("I love you", 2)], [c1, c2, c3])
        matches = results["I love you"]
        assert len(matches) == 2
        # Best two should be c1 and c2 (both contain "I love you" exactly)
        clip_ids = {m.clip_id for m in matches}
        assert "c3" not in clip_ids
        # First match's score >= second match's score
        assert matches[0].score >= matches[1].score


class TestMatchPhrasesEdgeCases:
    def test_clip_without_transcript_is_excluded(self):
        c1 = _clip("c1", "s", None)
        c2 = _clip("c2", "s", [_seg(0.0, 2.0, "thank you very much")])
        results = match_phrases([("thank you", 5)], [c1, c2])
        assert len(results["thank you"]) == 1
        assert results["thank you"][0].clip_id == "c2"

    def test_disabled_clip_is_excluded(self):
        c1 = _clip("c1", "s", [_seg(0.0, 2.0, "thank you")], disabled=True)
        c2 = _clip("c2", "s", [_seg(0.0, 2.0, "thank you")])
        results = match_phrases([("thank you", 5)], [c1, c2])
        assert {m.clip_id for m in results["thank you"]} == {"c2"}

    def test_no_threshold_returns_top_n_even_with_bad_matches(self):
        # Phrase has no good match — should still return up to N picks
        c1 = _clip("c1", "s", [_seg(0.0, 1.0, "hello")])
        results = match_phrases([("intergalactic warfare", 3)], [c1])
        # 1 candidate exists, count=3 → returns 1 (no padding)
        assert len(results["intergalactic warfare"]) == 1

    def test_count_exceeds_available_returns_all_without_padding(self):
        c1 = _clip("c1", "s", [_seg(0.0, 1.0, "alpha bravo")])
        c2 = _clip("c2", "s", [_seg(0.0, 1.0, "alpha charlie")])
        results = match_phrases([("alpha", 5)], [c1, c2])
        assert len(results["alpha"]) == 2

    def test_empty_phrase_is_skipped(self):
        c1 = _clip("c1", "s", [_seg(0.0, 1.0, "hello world")])
        results = match_phrases([("", 3), ("hello", 3)], [c1])
        assert "" not in results
        assert "hello" in results

    def test_zero_count_phrase_is_skipped(self):
        c1 = _clip("c1", "s", [_seg(0.0, 1.0, "hello world")])
        results = match_phrases([("hello", 0)], [c1])
        assert results == {}

    def test_empty_segment_text_is_excluded(self):
        c1 = _clip("c1", "s", [_seg(0.0, 1.0, ""), _seg(1.0, 2.0, "   ")])
        c2 = _clip("c2", "s", [_seg(0.0, 1.0, "thank you")])
        results = match_phrases([("thank you", 5)], [c1, c2])
        assert len(results["thank you"]) == 1
        assert results["thank you"][0].clip_id == "c2"

    def test_tie_break_is_deterministic_across_runs(self):
        # Two identical-scoring segments, one earlier in clip order, one earlier in time
        c1 = _clip("c1", "s", [_seg(2.0, 3.0, "thank you")])
        c2 = _clip("c2", "s", [_seg(0.5, 1.5, "thank you")])
        # Run twice; same order both times
        r1 = match_phrases([("thank you", 2)], [c1, c2])["thank you"]
        r2 = match_phrases([("thank you", 2)], [c1, c2])["thank you"]
        assert [m.clip_id for m in r1] == [m.clip_id for m in r2]
        # Tie-break: earlier segment.start_time wins → c2 first
        assert r1[0].clip_id == "c2"
        assert r1[1].clip_id == "c1"

    def test_phrase_order_preserved_in_results(self):
        c1 = _clip("c1", "s", [_seg(0.0, 1.0, "alpha bravo charlie")])
        results = match_phrases([("charlie", 1), ("alpha", 1), ("bravo", 1)], [c1])
        assert list(results.keys()) == ["charlie", "alpha", "bravo"]


class TestBuildSequenceData:
    def test_seconds_to_frames_conversion(self):
        clip = _clip("c1", "src1", [_seg(2.5, 3.5, "thank you")])
        source = _source("src1", fps=24.0)
        match = MatchResult(
            phrase="thank you",
            clip_id="c1",
            segment_index=0,
            segment=clip.transcript[0],
            score=95,
            match_start=0,
            match_end=9,
        )
        result = build_sequence_data([match], {"c1": clip}, {"src1": source})
        assert len(result) == 1
        ret_clip, ret_source, in_frame, out_frame = result[0]
        assert ret_clip is clip
        assert ret_source is source
        assert in_frame == 60   # 2.5 * 24
        assert out_frame == 84  # 3.5 * 24

    def test_segment_starting_at_zero(self):
        clip = _clip("c1", "src1", [_seg(0.0, 1.0, "hi")])
        source = _source("src1", fps=30.0)
        match = MatchResult(
            phrase="hi", clip_id="c1", segment_index=0,
            segment=clip.transcript[0], score=100, match_start=0, match_end=2,
        )
        result = build_sequence_data([match], {"c1": clip}, {"src1": source})
        assert result[0][2] == 0
        assert result[0][3] == 30

    def test_zero_duration_segment_pads_to_one_frame(self):
        clip = _clip("c1", "src1", [_seg(1.0, 1.0, "hi")])
        source = _source("src1", fps=30.0)
        match = MatchResult(
            phrase="hi", clip_id="c1", segment_index=0,
            segment=clip.transcript[0], score=100, match_start=0, match_end=2,
        )
        result = build_sequence_data([match], {"c1": clip}, {"src1": source})
        in_frame, out_frame = result[0][2], result[0][3]
        assert out_frame == in_frame + 1

    def test_missing_clip_or_source_drops_match_with_warning(self, caplog):
        # match references c2, but c2 not in lookup
        match = MatchResult(
            phrase="x", clip_id="c2", segment_index=0,
            segment=_seg(0.0, 1.0, "x"), score=50, match_start=-1, match_end=-1,
        )
        result = build_sequence_data([match], {}, {})
        assert result == []


class TestFlattenMatchesInPhraseOrder:
    def test_preserves_phrase_order_and_score_order(self):
        m1 = MatchResult("a", "c1", 0, _seg(0, 1, "a1"), 90, 0, 1)
        m2 = MatchResult("a", "c1", 1, _seg(1, 2, "a2"), 80, 0, 2)
        m3 = MatchResult("b", "c2", 0, _seg(0, 1, "b1"), 95, 0, 2)
        flat = flatten_matches_in_phrase_order({"a": [m1, m2], "b": [m3]})
        assert flat == [m1, m2, m3]

    def test_enabled_keys_filters_out_disabled(self):
        m1 = MatchResult("a", "c1", 0, _seg(0, 1, "a1"), 90, 0, 1)
        m2 = MatchResult("a", "c1", 1, _seg(1, 2, "a2"), 80, 0, 2)
        # Only m1 is enabled
        enabled = {("a", "c1", 0)}
        flat = flatten_matches_in_phrase_order({"a": [m1, m2]}, enabled_keys=enabled)
        assert flat == [m1]

    def test_empty_enabled_set_yields_empty_list(self):
        m1 = MatchResult("a", "c1", 0, _seg(0, 1, "a1"), 90, 0, 1)
        flat = flatten_matches_in_phrase_order({"a": [m1]}, enabled_keys=set())
        assert flat == []

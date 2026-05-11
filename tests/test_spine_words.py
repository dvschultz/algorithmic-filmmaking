"""Tests for ``core/spine/words.py``: word inventory + 5 preset ordering modes.

These tests pin the contract the U4 word-sequencer feature relies on:
- Word normalization (lowercase + strip surrounding ASCII punctuation;
  contractions kept whole).
- ``WordInventory`` with both ``by_word`` and ``by_clip`` indices.
- Five mode functions: ``alphabetical``, ``by_chosen_words``, ``by_frequency``,
  ``by_property``, ``from_word_list`` — with the keyword-name and default
  semantics documented in the plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from core.spine.words import (
    MissingWordsError,
    WordInstance,
    WordInventory,
    alphabetical,
    build_inventory,
    by_chosen_words,
    by_frequency,
    by_property,
    from_word_list,
)
from core.transcription import TranscriptSegment, WordTimestamp


# ---------------------------------------------------------------------------
# Mocks: lightweight stand-ins so the spine tests can stay decoupled from
# the full Clip/Source dataclasses.
# ---------------------------------------------------------------------------


@dataclass
class MockSource:
    id: str = "src1"
    fps: float = 24.0


@dataclass
class MockClip:
    id: str = "clip1"
    source_id: str = "src1"
    start_frame: int = 0
    end_frame: int = 240  # 10 s at 24 fps
    transcript: Optional[list[TranscriptSegment]] = None


def _seg(words: list[tuple[str, float, float]], language: str = "en") -> TranscriptSegment:
    """Build a TranscriptSegment with a list of (text, start, end) word tuples."""
    word_list = [WordTimestamp(start=s, end=e, text=t) for (t, s, e) in words]
    return TranscriptSegment(
        start_time=words[0][1] if words else 0.0,
        end_time=words[-1][2] if words else 0.0,
        text=" ".join(t for t, _, _ in words),
        confidence=1.0,
        words=word_list,
        language=language,
    )


# ---------------------------------------------------------------------------
# build_inventory
# ---------------------------------------------------------------------------


class TestBuildInventory:
    def test_empty_clips_returns_empty_inventory(self):
        inv = build_inventory([])
        assert isinstance(inv, WordInventory)
        assert inv.by_word == {}
        assert inv.by_clip == {}

    def test_clip_without_transcript_skipped_cleanly(self):
        clip = MockClip(transcript=None)
        inv = build_inventory([(clip, MockSource())])
        assert inv.by_word == {}
        assert inv.by_clip == {}

    def test_clip_with_empty_segment_words_skipped(self):
        clip = MockClip(
            transcript=[
                TranscriptSegment(
                    start_time=0.0,
                    end_time=1.0,
                    text="",
                    confidence=1.0,
                    words=[],
                    language="en",
                )
            ]
        )
        inv = build_inventory([(clip, MockSource())])
        assert inv.by_word == {}
        # An entry in by_clip is fine but must not invent words.
        assert inv.by_clip.get(clip.id, []) == []

    def test_normalization_lowercase_and_punctuation(self):
        clip = MockClip(
            transcript=[
                _seg(
                    [
                        ("The", 0.0, 0.3),
                        ("Sky,", 0.4, 0.8),
                        ("blue!", 0.9, 1.2),
                        ("the", 1.3, 1.5),
                    ]
                )
            ]
        )
        inv = build_inventory([(clip, MockSource())])
        assert set(inv.by_word.keys()) == {"the", "sky", "blue"}
        # Original text is preserved on the instance.
        the_instances = inv.by_word["the"]
        assert {w.text for w in the_instances} == {"The", "the"}

    def test_contractions_stay_whole(self):
        clip = MockClip(
            transcript=[_seg([("don't", 0.0, 0.4), ("don't", 0.5, 0.9)])]
        )
        inv = build_inventory([(clip, MockSource())])
        assert list(inv.by_word.keys()) == ["don't"]
        assert len(inv.by_word["don't"]) == 2

    def test_by_clip_groups_words_by_clip_id(self):
        clip_a = MockClip(id="A", transcript=[_seg([("alpha", 0.0, 0.5), ("beta", 0.6, 1.0)])])
        clip_b = MockClip(id="B", transcript=[_seg([("alpha", 0.0, 0.5)])])
        inv = build_inventory([(clip_a, MockSource()), (clip_b, MockSource())])
        assert len(inv.by_clip["A"]) == 2
        assert len(inv.by_clip["B"]) == 1
        # Cross-check by_word
        assert len(inv.by_word["alpha"]) == 2
        assert len(inv.by_word["beta"]) == 1

    def test_instance_records_indices_and_ids(self):
        clip = MockClip(
            id="clip_x",
            source_id="src_x",
            transcript=[_seg([("hello", 0.0, 0.5), ("world", 0.6, 1.0)])],
        )
        inv = build_inventory([(clip, MockSource(id="src_x"))])
        hello = inv.by_word["hello"][0]
        assert hello.source_id == "src_x"
        assert hello.clip_id == "clip_x"
        assert hello.segment_index == 0
        assert hello.word_index == 0
        world = inv.by_word["world"][0]
        assert world.word_index == 1


# ---------------------------------------------------------------------------
# alphabetical
# ---------------------------------------------------------------------------


class TestAlphabetical:
    def test_lexical_order(self):
        clip = MockClip(
            transcript=[
                _seg([("zoo", 0.0, 0.3), ("apple", 0.4, 0.7), ("mango", 0.8, 1.1)])
            ]
        )
        inv = build_inventory([(clip, MockSource())])
        result = alphabetical(inv)
        assert [w.text.lower() for w in result] == ["apple", "mango", "zoo"]

    def test_encounter_order_within_word(self):
        clip = MockClip(
            transcript=[
                _seg(
                    [
                        ("the", 0.0, 0.2),  # first
                        ("apple", 0.3, 0.6),
                        ("the", 0.7, 0.9),  # second
                    ]
                )
            ]
        )
        inv = build_inventory([(clip, MockSource())])
        result = alphabetical(inv)
        # apple sorts before the
        assert result[0].text == "apple"
        # Both "the" come in encounter order (by start time).
        the_starts = [w.start for w in result[1:]]
        assert the_starts == sorted(the_starts)

    def test_empty_inventory(self):
        assert alphabetical(WordInventory(by_word={}, by_clip={})) == []


# ---------------------------------------------------------------------------
# by_frequency
# ---------------------------------------------------------------------------


class TestByFrequency:
    def _inv(self) -> WordInventory:
        clip = MockClip(
            transcript=[
                _seg(
                    [
                        ("the", 0.0, 0.2),
                        ("dog", 0.3, 0.5),
                        ("the", 0.6, 0.8),
                        ("cat", 0.9, 1.1),
                        ("the", 1.2, 1.4),
                        ("dog", 1.5, 1.7),
                    ]
                )
            ]
        )
        return build_inventory([(clip, MockSource())])

    def test_descending_default(self):
        inv = self._inv()
        result = by_frequency(inv)  # default descending
        # "the" (3) → "dog" (2) → "cat" (1)
        words = [w.text.lower() for w in result]
        assert words[:3] == ["the", "the", "the"]
        assert words[3:5] == ["dog", "dog"]
        assert words[5:] == ["cat"]

    def test_ascending(self):
        inv = self._inv()
        result = by_frequency(inv, order="ascending")
        words = [w.text.lower() for w in result]
        assert words[0] == "cat"
        assert words[1:3] == ["dog", "dog"]
        assert words[3:] == ["the", "the", "the"]


# ---------------------------------------------------------------------------
# by_chosen_words
# ---------------------------------------------------------------------------


class TestByChosenWords:
    def _inv(self) -> WordInventory:
        clip = MockClip(
            transcript=[
                _seg(
                    [
                        ("never", 0.0, 0.3),
                        ("always", 0.4, 0.7),
                        ("never", 0.8, 1.1),
                        ("silence", 1.2, 1.5),
                        ("always", 1.6, 1.9),
                    ]
                )
            ]
        )
        return build_inventory([(clip, MockSource())])

    def test_grouped_by_include_order(self):
        inv = self._inv()
        result = by_chosen_words(inv, include=["never", "always"])
        words = [w.text.lower() for w in result]
        # Every "never" first, then every "always".
        assert words == ["never", "never", "always", "always"]

    def test_case_insensitive(self):
        inv = self._inv()
        result = by_chosen_words(inv, include=["Never", "ALWAYS"])
        assert len(result) == 4

    def test_unknown_words_silently_dropped(self):
        inv = self._inv()
        result = by_chosen_words(inv, include=["never", "xyz"])
        words = [w.text.lower() for w in result]
        assert words == ["never", "never"]

    def test_all_unknown_returns_empty(self):
        inv = self._inv()
        assert by_chosen_words(inv, include=["xyz", "abc"]) == []


# ---------------------------------------------------------------------------
# by_property
# ---------------------------------------------------------------------------


class TestByProperty:
    def _inv(self) -> WordInventory:
        clip = MockClip(
            transcript=[
                _seg(
                    [
                        ("a", 0.0, 0.1),  # length 1, dur 0.1
                        ("bb", 0.2, 0.6),  # length 2, dur 0.4
                        ("ccc", 0.7, 0.9),  # length 3, dur 0.2
                        ("a", 1.0, 1.5),  # repeat: bumps "a" frequency to 2
                    ]
                )
            ]
        )
        return build_inventory([(clip, MockSource())])

    def test_length_ascending_default(self):
        inv = self._inv()
        result = by_property(inv, key="length")
        lengths = [len(w.text) for w in result]
        assert lengths == sorted(lengths)
        assert lengths[0] == 1
        assert lengths[-1] == 3

    def test_length_descending(self):
        inv = self._inv()
        result = by_property(inv, key="length", order="descending")
        lengths = [len(w.text) for w in result]
        assert lengths == sorted(lengths, reverse=True)

    def test_duration_ascending(self):
        inv = self._inv()
        result = by_property(inv, key="duration")
        durations = [w.end - w.start for w in result]
        assert durations == sorted(durations)

    def test_log_frequency_ascending(self):
        inv = self._inv()
        # "a" appears twice, "bb"/"ccc" once each → "bb"/"ccc" come first (lower freq).
        result = by_property(inv, key="log_frequency")
        # Two least-frequent words first (singletons), then the doubled "a"s.
        words = [w.text for w in result]
        assert words[-2:] == ["a", "a"]


# ---------------------------------------------------------------------------
# from_word_list
# ---------------------------------------------------------------------------


class TestFromWordList:
    def _inv(self) -> WordInventory:
        clip = MockClip(
            transcript=[
                _seg(
                    [
                        ("the", 0.0, 0.2),
                        ("sky", 0.3, 0.6),
                        ("the", 0.7, 0.9),
                    ]
                )
            ]
        )
        return build_inventory([(clip, MockSource())])

    def test_literal_materialization_with_repeats(self):
        inv = self._inv()
        result = from_word_list(inv, sequence=["the", "the", "the", "sky"])
        # Exactly 4 slots — repeats produce multiple instance picks.
        assert len(result) == 4
        # The non-"the" entry is "sky".
        assert result[3].text.lower() == "sky"

    def test_skip_missing_default(self):
        inv = self._inv()
        result = from_word_list(inv, sequence=["the", "missing", "sky"])
        assert [w.text.lower() for w in result] == ["the", "sky"]

    def test_raise_missing(self):
        inv = self._inv()
        with pytest.raises(MissingWordsError) as exc:
            from_word_list(inv, sequence=["the", "missing", "sky", "also_missing"], on_missing="raise")
        assert "missing" in exc.value.missing
        assert "also_missing" in exc.value.missing

    def test_case_insensitive_lookup(self):
        inv = self._inv()
        result = from_word_list(inv, sequence=["The", "SKY"])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# WordInventory dataclass
# ---------------------------------------------------------------------------


class TestWordInventory:
    def test_inventory_is_frozen(self):
        inv = WordInventory(by_word={}, by_clip={})
        with pytest.raises((AttributeError, TypeError)):
            inv.by_word = {}  # type: ignore[misc]

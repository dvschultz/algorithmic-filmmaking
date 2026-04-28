"""Cassette Tape: Phrase-driven sequencing of transcribed clips.

The user supplies a list of (phrase, count) pairs. For each phrase, this
module scores every transcript segment in the project and returns the top-N
best matches. The dialog renders matches with confidence + transcript snippet
so the user can prune, then a sequence of sub-clips (trimmed to the matched
segments) is built.

The match unit is the TranscriptSegment, scored against each phrase via
``rapidfuzz.fuzz.partial_ratio`` (case-insensitive). ``partial_ratio`` finds
the best-matching substring inside the segment, which mirrors the user's
mental model — "this clip says my phrase somewhere" — better than
``ratio`` (which penalises length differences) or ``token_sort_ratio``
(which is too forgiving on word order).

There is no minimum-score threshold. The user controls quality on the
review screen by toggling individual matches off.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from models.clip import Clip, Source
from core.transcription import TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """A single (phrase, clip, segment) match with its score."""

    phrase: str
    clip_id: str
    segment_index: int
    segment: TranscriptSegment
    score: int  # 0–100, rounded
    match_start: int  # substring start in segment.text (chars), or -1 if unknown
    match_end: int  # substring end in segment.text (chars), or -1 if unknown


# Slider range for matches-per-phrase. Both the dialog setup screen and
# the agent tool's input validator clamp to this range.
SLIDER_MIN = 1
SLIDER_MAX = 5
SLIDER_DEFAULT = 3


def clamp_count(count: int) -> int:
    """Clamp a phrase's match-count to [SLIDER_MIN, SLIDER_MAX]."""
    return max(SLIDER_MIN, min(SLIDER_MAX, count))


def safe_fps(source: Optional[Source]) -> float:
    """Return the source's fps, or 30.0 if missing / non-positive.

    A Source with fps<=0 (corrupted metadata, failed probe) would propagate
    divide-by-zero into seconds-to-frames math and the timeline scene. The
    UI apply path and build_sequence_data both need this guard, so the
    helper lives here next to the matcher.
    """
    if source is None:
        return 30.0
    fps = getattr(source, "fps", None)
    if fps and fps > 0:
        return float(fps)
    return 30.0


_MIN_SEGMENT_CHARS = 3
"""Floor for transcript segment length. Segments shorter than this are
treated as noise (Whisper often emits single-char artifacts on silent or
near-silent clips: "a", ".", "uh"). They aren't excluded from the
candidate set — empty-text exclusion still happens earlier — but their
score is forced to 0 so they don't surface as matches."""


def _score_phrase_against_segment(phrase: str, segment_text: str) -> tuple[int, int, int]:
    """Score a phrase against a segment's text.

    Returns ``(score_0_100, match_start, match_end)``. ``match_start``/``end``
    point into the **original** ``segment_text`` so the UI can highlight the
    matched substring without index drift on Unicode characters whose
    lowercase form has a different length (e.g., Turkish dotted-I, German ß
    in a casefold scenario).

    Scoring is length-aware to fix `partial_ratio`'s asymmetric bias:

    - When the segment is **at least** as long as the phrase, use
      ``partial_ratio`` so a phrase appearing inside a longer segment
      ("thank you" inside "well thank you for coming") scores high.
    - When the segment is **shorter** than the phrase, fall back to
      ``ratio`` (length-penalized whole-string Levenshtein) so a single-
      character segment like "a" or "the" cannot score 100 against a
      multi-word phrase that happens to contain that token. Without this
      branch, ``partial_ratio`` would invert the intent: it aligns the
      shorter string (the segment) against substrings of the longer (the
      phrase), so any segment that appears verbatim inside the phrase
      scores 100 — which is the opposite of what the user wants.

    Segments under :data:`_MIN_SEGMENT_CHARS` characters (after strip) are
    treated as noise and forced to score 0; Whisper emits these on silent
    clips as hallucination artifacts.

    We pass ``processor=str.lower`` to rapidfuzz so it lowercases
    internally for scoring, while alignment indices remain valid against
    the *original* strings. Returns ``(0, -1, -1)`` for empty inputs or
    when alignment fails.
    """
    if not phrase or not segment_text:
        return 0, -1, -1

    # Noise floor: very short segments cannot meaningfully "say" the phrase.
    if len(segment_text.strip()) < _MIN_SEGMENT_CHARS:
        return 0, -1, -1

    from rapidfuzz import fuzz

    score = 0
    match_start = -1
    match_end = -1

    # Length-branch: segments shorter than the phrase get whole-string ratio
    # (penalises length difference) instead of partial_ratio (which would
    # match the segment as a substring of the phrase — wrong direction).
    if len(segment_text) < len(phrase):
        try:
            score = int(round(fuzz.ratio(phrase, segment_text, processor=str.lower)))
        except Exception:
            logger.debug("rapidfuzz ratio failed", exc_info=True)
            score = 0
        # No meaningful sub-region to highlight when scoring whole-string;
        # let the UI render the full segment text.
        return score, 0, len(segment_text)

    # Segment is at least as long as the phrase — partial_ratio is correct.
    try:
        alignment = fuzz.partial_ratio_alignment(
            phrase, segment_text, processor=str.lower
        )
    except Exception:
        logger.debug(
            "rapidfuzz partial_ratio_alignment failed; falling back to score-only path",
            exc_info=True,
        )
        alignment = None

    if alignment is not None:
        score = int(round(alignment.score))
        match_start = int(alignment.dest_start)
        match_end = int(alignment.dest_end)
    else:
        try:
            score = int(round(fuzz.partial_ratio(phrase, segment_text, processor=str.lower)))
        except Exception:
            logger.debug("rapidfuzz partial_ratio also failed", exc_info=True)
            score = 0

    return score, match_start, match_end


def match_phrases(
    phrases_with_counts: list[tuple[str, int]],
    clips: list[Clip],
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> dict[str, list[MatchResult]]:
    """Score each phrase against every transcript segment and return top-N.

    Args:
        phrases_with_counts: ``[(phrase, count), ...]`` in user-entered order.
            Empty phrases are skipped (the dialog should already filter them,
            but we defend here too).
        clips: All clips in the candidate pool. Clips without a transcript or
            with ``disabled=True`` are silently excluded.
        is_cancelled: Optional callable returning ``True`` when the caller
            wants to abort. Checked once per phrase; partial results are
            discarded and an empty dict is returned. Lets a worker thread's
            cancel() actually unblock a long-running match instead of just
            suppressing the post-completion signal.

    Returns:
        Dict keyed by phrase (in input order, Python 3.7+ insertion-ordered)
        whose values are lists of ``MatchResult`` objects sorted best-first.
        Each list contains at most ``count`` results — fewer if the candidate
        pool has fewer segments. Returns ``{}`` if cancelled mid-loop.
    """
    candidates: list[tuple[int, Clip, int, TranscriptSegment]] = []
    for clip_index, clip in enumerate(clips):
        if clip.disabled:
            continue
        if not clip.transcript:
            continue
        for seg_index, segment in enumerate(clip.transcript):
            if not segment.text or not segment.text.strip():
                continue
            candidates.append((clip_index, clip, seg_index, segment))

    logger.debug(
        "cassette_tape: %d candidate segments from %d clips",
        len(candidates),
        len(clips),
    )

    results: dict[str, list[MatchResult]] = {}

    for phrase, count in phrases_with_counts:
        if is_cancelled is not None and is_cancelled():
            logger.info("cassette_tape: matching cancelled mid-loop")
            return {}

        phrase_clean = (phrase or "").strip()
        if not phrase_clean or count <= 0:
            continue

        # Score each candidate and keep only the lightweight tuple needed for
        # ranking + later MatchResult reconstruction. Deferring the dataclass
        # construction until after the sort+slice avoids ~99% throwaway
        # allocations at typical scale (only `count` of N candidates survive).
        # Tuple shape: (-score, clip_index, start_time, score, m_start, m_end,
        #               segment, clip_id, seg_index). The sort uses the first
        # three for deterministic tie-break (best score, earliest segment,
        # earliest clip in input order).
        scored: list[tuple] = []
        for clip_index, clip, seg_index, segment in candidates:
            score, m_start, m_end = _score_phrase_against_segment(phrase_clean, segment.text)
            scored.append((
                -score, clip_index, segment.start_time,
                score, m_start, m_end, segment, clip.id, seg_index,
            ))

        scored.sort(key=lambda r: (r[0], r[2], r[1]))

        results[phrase_clean] = [
            MatchResult(
                phrase=phrase_clean,
                clip_id=item[7],
                segment_index=item[8],
                segment=item[6],
                score=item[3],
                match_start=item[4],
                match_end=item[5],
            )
            for item in scored[:count]
        ]

        logger.debug(
            "cassette_tape: phrase=%r returned %d matches (best score=%d)",
            phrase_clean,
            len(results[phrase_clean]),
            results[phrase_clean][0].score if results[phrase_clean] else 0,
        )

    return results


def build_sequence_data(
    matches_in_order: list[MatchResult],
    clips_by_id: dict[str, Clip],
    sources_by_id: dict[str, Source],
) -> list[tuple[Clip, Source, int, int]]:
    """Convert a flat list of matches into ``(clip, source, in_frame, out_frame)`` tuples.

    The frame offsets are **relative to the clip's start_frame** — the sequence
    tab adds ``clip.start_frame`` when calling ``add_clip_to_track``, matching
    the pattern used by Signature Style. Frames are derived from the segment's
    seconds-based ``start_time``/``end_time`` via ``round(t * source.fps)``.

    Matches whose clip or source is missing from the lookup dicts are dropped
    with a warning — this happens when the project changes between matching
    and generation, which shouldn't normally occur in a single dialog run.
    """
    sequence: list[tuple[Clip, Source, int, int]] = []

    for match in matches_in_order:
        clip = clips_by_id.get(match.clip_id)
        if clip is None:
            logger.warning("cassette_tape: clip not found: %s", match.clip_id)
            continue
        source = sources_by_id.get(clip.source_id)
        if source is None:
            logger.warning("cassette_tape: source not found for clip %s", clip.id)
            continue

        fps = safe_fps(source)
        in_frame = int(round(match.segment.start_time * fps))
        out_frame = int(round(match.segment.end_time * fps))
        if out_frame <= in_frame:
            out_frame = in_frame + 1  # guarantee non-empty sub-clip

        sequence.append((clip, source, in_frame, out_frame))

    logger.info("cassette_tape: built sequence with %d sub-clips", len(sequence))
    return sequence


def flatten_matches_in_phrase_order(
    matches_by_phrase: dict[str, list[MatchResult]],
    enabled_keys: Optional[set[tuple[str, str, int]]] = None,
) -> list[MatchResult]:
    """Flatten the phrase-grouped matches into the final sequence order.

    Phrase order is preserved (dict insertion order); within each phrase,
    matches are kept in their already-sorted score order.

    Args:
        matches_by_phrase: Output of :func:`match_phrases`.
        enabled_keys: Optional set of ``(phrase, clip_id, segment_index)`` keys
            indicating which matches should appear in the final sequence. When
            ``None``, every match is included.
    """
    flat: list[MatchResult] = []
    for phrase, matches in matches_by_phrase.items():
        for match in matches:
            if enabled_keys is not None:
                key = (match.phrase, match.clip_id, match.segment_index)
                if key not in enabled_keys:
                    continue
            flat.append(match)
    return flat

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


def _score_phrase_against_segment(phrase: str, segment_text: str) -> tuple[int, int, int]:
    """Score a phrase against a segment's text.

    Returns ``(score_0_100, match_start, match_end)``. ``match_start``/``end``
    point into ``segment_text`` so the UI can highlight the matched substring.
    Returns ``(0, -1, -1)`` for empty inputs.
    """
    if not phrase or not segment_text:
        return 0, -1, -1

    from rapidfuzz import fuzz

    phrase_lc = phrase.lower()
    text_lc = segment_text.lower()

    score = int(round(fuzz.partial_ratio(phrase_lc, text_lc)))

    match_start = -1
    match_end = -1
    try:
        alignment = fuzz.partial_ratio_alignment(phrase_lc, text_lc)
    except Exception:
        alignment = None
    if alignment is not None:
        match_start = int(alignment.dest_start)
        match_end = int(alignment.dest_end)

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

        scored: list[tuple[int, int, float, MatchResult]] = []
        for clip_index, clip, seg_index, segment in candidates:
            score, m_start, m_end = _score_phrase_against_segment(phrase_clean, segment.text)
            mr = MatchResult(
                phrase=phrase_clean,
                clip_id=clip.id,
                segment_index=seg_index,
                segment=segment,
                score=score,
                match_start=m_start,
                match_end=m_end,
            )
            # Sort key projects (-score, start_time, clip_index) — MatchResult
            # itself is never compared because the lambda below picks indices
            # 0/2/1 only. Deterministic tie-break: best score wins, then
            # earliest segment in the source, then earliest clip in the
            # candidate list.
            scored.append((-score, clip_index, segment.start_time, mr))

        scored.sort(key=lambda r: (r[0], r[2], r[1]))
        results[phrase_clean] = [item[3] for item in scored[:count]]

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

        fps = source.fps if source.fps and source.fps > 0 else 30.0
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

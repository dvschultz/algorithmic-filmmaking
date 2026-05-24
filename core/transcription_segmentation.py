"""Transcript segment refinement policies."""

from __future__ import annotations

import re
from typing import Optional

from core.transcription_models import TranscriptSegment, WordTimestamp


def _words_for_range(
    words: Optional[list[WordTimestamp]],
    start_time: float,
    end_time: float,
) -> Optional[list[WordTimestamp]]:
    """Return word timestamps whose midpoint falls inside the refined segment."""
    if words is None:
        return None
    if not words:
        return []
    selected = [
        word
        for word in words
        if start_time <= ((word.start + word.end) / 2.0) <= end_time
    ]
    return selected


def _clone_segment(
    segment: TranscriptSegment,
    *,
    start_time: float,
    end_time: float,
    text: str,
) -> TranscriptSegment:
    """Return a segment copy with adjusted timing/text and preserved word data."""
    return TranscriptSegment(
        start_time=start_time,
        end_time=end_time,
        text=text.strip(),
        confidence=segment.confidence,
        words=_words_for_range(segment.words, start_time, end_time),
        language=segment.language,
    )


def _split_text_units(text: str, mode: str) -> list[str]:
    """Split text into phrase/sentence units while preserving punctuation."""
    stripped = text.strip()
    if not stripped:
        return []
    if mode == "sentence":
        units = re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", stripped)
    else:
        units = re.split(r"(?<=[,;:])\s+|(?<=[.!?])\s+", stripped)
    return [unit.strip() for unit in units if unit.strip()]


def _split_segment_by_text_units(
    segment: TranscriptSegment,
    units: list[str],
) -> list[TranscriptSegment]:
    """Split one segment using text-unit lengths to approximate boundaries."""
    if len(units) <= 1:
        return [segment]

    total_chars = sum(max(1, len(unit)) for unit in units)
    duration = max(0.0, segment.end_time - segment.start_time)
    cursor = segment.start_time
    result: list[TranscriptSegment] = []
    for index, unit in enumerate(units):
        if index == len(units) - 1:
            end_time = segment.end_time
        else:
            end_time = cursor + (duration * (max(1, len(unit)) / total_chars))
        result.append(_clone_segment(segment, start_time=cursor, end_time=end_time, text=unit))
        cursor = end_time
    return result


def _split_segment_by_fixed_seconds(
    segment: TranscriptSegment,
    max_seconds: float,
) -> list[TranscriptSegment]:
    """Split one segment into fixed-duration chunks with approximate text boundaries."""
    duration = segment.end_time - segment.start_time
    if max_seconds <= 0 or duration <= max_seconds:
        return [segment]

    words = [word for word in segment.text.split() if word]
    if not words:
        return [segment]

    chunk_count = max(1, int((duration + max_seconds - 0.001) // max_seconds))
    words_per_chunk = max(1, (len(words) + chunk_count - 1) // chunk_count)
    result: list[TranscriptSegment] = []
    for chunk_index in range(chunk_count):
        chunk_words = words[chunk_index * words_per_chunk:(chunk_index + 1) * words_per_chunk]
        if not chunk_words:
            continue
        start = segment.start_time + (duration * (chunk_index / chunk_count))
        end = segment.start_time + (duration * ((chunk_index + 1) / chunk_count))
        if chunk_index == chunk_count - 1:
            end = segment.end_time
        result.append(
            _clone_segment(
                segment,
                start_time=start,
                end_time=end,
                text=" ".join(chunk_words),
            )
        )
    return result or [segment]


def refine_transcript_segments(
    segments: list[TranscriptSegment],
    mode: str = "backend",
    max_seconds: float = 12.0,
) -> list[TranscriptSegment]:
    """Apply an explicit transcript segmentation policy to backend output."""
    normalized = (mode or "backend").strip().lower()
    if normalized in ("backend", "silence", "whisper", "none"):
        return segments

    refined: list[TranscriptSegment] = []
    for segment in segments:
        if normalized == "fixed":
            refined.extend(_split_segment_by_fixed_seconds(segment, max_seconds))
        elif normalized in ("sentence", "phrase"):
            refined.extend(_split_segment_by_text_units(segment, _split_text_units(segment.text, normalized)))
        else:
            refined.append(segment)
    return refined

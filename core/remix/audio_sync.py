"""Audio-driven sequence alignment for rhythm-based editing.

This module provides functions to align video clip cuts to music beats,
enabling rhythm-based editing for music videos, montages, and trailers.

Strategies:
- nearest: Cut at nearest beat to natural clip boundary
- on_beat: Force cuts exactly on beats (may trim clips)
- downbeat: Prefer strong beats (every 4th beat in 4/4)
- onset: Align to transients/hits instead of beats
"""

import logging
from dataclasses import dataclass
from typing import Optional

from core.analysis.audio import AudioAnalysis

logger = logging.getLogger(__name__)


@dataclass
class AlignmentSuggestion:
    """Suggested cut point adjustment for a clip.

    Attributes:
        clip_id: ID of the clip to adjust
        current_end: Current end time in sequence (seconds)
        suggested_end: Suggested end time aligned to beat
        beat_time: The beat timestamp being aligned to
        adjustment_seconds: Difference between current and suggested
    """
    clip_id: str
    current_end: float
    suggested_end: float
    beat_time: float
    adjustment_seconds: float

    def to_dict(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "current_end": round(self.current_end, 3),
            "suggested_end": round(self.suggested_end, 3),
            "beat_time": round(self.beat_time, 3),
            "adjustment_seconds": round(self.adjustment_seconds, 3),
        }


def suggest_beat_aligned_cuts(
    clip_end_times: list[tuple[str, float]],
    audio_analysis: AudioAnalysis,
    strategy: str = "nearest",
    max_adjustment: float = 0.5,
) -> list[AlignmentSuggestion]:
    """Suggest cut point adjustments to align with beats.

    Non-destructive: Returns suggestions for human review.

    Args:
        clip_end_times: List of (clip_id, end_time) tuples in sequence order
        audio_analysis: Beat analysis from music track
        strategy: Alignment strategy - "nearest", "on_beat", "downbeat", "onset"
        max_adjustment: Maximum time shift allowed (seconds)

    Returns:
        List of AlignmentSuggestion for each clip that would benefit from adjustment

    Raises:
        ValueError: If strategy is invalid or audio has no beats
    """
    if strategy not in ("nearest", "on_beat", "downbeat", "onset"):
        raise ValueError(f"Invalid strategy: {strategy}. Use: nearest, on_beat, downbeat, onset")

    # Select reference points based on strategy
    if strategy == "onset":
        reference_times = audio_analysis.onset_times
        if not reference_times:
            raise ValueError("No onsets detected in audio")
    elif strategy == "downbeat":
        reference_times = audio_analysis.downbeat_times
        if not reference_times:
            # Fall back to regular beats
            reference_times = audio_analysis.beat_times
    else:
        reference_times = audio_analysis.beat_times

    if not reference_times:
        raise ValueError("No beats detected in audio")

    suggestions = []

    for clip_id, current_end in clip_end_times:
        # Find nearest reference point
        nearest = min(reference_times, key=lambda t: abs(t - current_end))
        adjustment = nearest - current_end

        # Only suggest if within max_adjustment threshold
        if abs(adjustment) <= max_adjustment and abs(adjustment) > 0.01:
            suggestions.append(AlignmentSuggestion(
                clip_id=clip_id,
                current_end=current_end,
                suggested_end=nearest,
                beat_time=nearest,
                adjustment_seconds=adjustment,
            ))

    return suggestions


def align_times_to_beats(
    times: list[float],
    audio_analysis: AudioAnalysis,
    strategy: str = "nearest",
) -> list[float]:
    """Align a list of times to nearest beats.

    Args:
        times: List of times (seconds) to align
        audio_analysis: Beat analysis from music track
        strategy: "nearest" (beat), "downbeat", or "onset"

    Returns:
        List of aligned times (same length as input)
    """
    if strategy == "onset":
        reference_times = audio_analysis.onset_times or audio_analysis.beat_times
    elif strategy == "downbeat":
        reference_times = audio_analysis.downbeat_times or audio_analysis.beat_times
    else:
        reference_times = audio_analysis.beat_times

    if not reference_times:
        return times  # No beats, return unchanged

    aligned = []
    for t in times:
        nearest = min(reference_times, key=lambda b: abs(b - t))
        aligned.append(nearest)

    return aligned


def calculate_beat_intervals(audio_analysis: AudioAnalysis) -> list[float]:
    """Calculate intervals between consecutive beats.

    Useful for detecting tempo changes or irregular rhythms.

    Args:
        audio_analysis: Beat analysis from music track

    Returns:
        List of intervals in seconds (length = len(beats) - 1)
    """
    beats = audio_analysis.beat_times
    if len(beats) < 2:
        return []

    return [beats[i+1] - beats[i] for i in range(len(beats) - 1)]


def get_beats_in_range(
    audio_analysis: AudioAnalysis,
    start_time: float,
    end_time: float,
    include_downbeats_only: bool = False,
) -> list[float]:
    """Get all beats within a time range.

    Args:
        audio_analysis: Beat analysis from music track
        start_time: Range start (seconds)
        end_time: Range end (seconds)
        include_downbeats_only: If True, only return downbeats

    Returns:
        List of beat times within the range
    """
    if include_downbeats_only:
        beats = audio_analysis.downbeat_times
    else:
        beats = audio_analysis.beat_times

    return [b for b in beats if start_time <= b <= end_time]


def estimate_clip_count_for_duration(
    audio_analysis: AudioAnalysis,
    target_duration: float,
    beats_per_clip: int = 4,
) -> int:
    """Estimate how many clips fit in a duration at a given beat rate.

    Useful for planning music video edits.

    Args:
        audio_analysis: Beat analysis from music track
        target_duration: Target sequence duration (seconds)
        beats_per_clip: How many beats each clip should span

    Returns:
        Estimated number of clips
    """
    if audio_analysis.tempo_bpm <= 0:
        return 0

    seconds_per_beat = 60.0 / audio_analysis.tempo_bpm
    seconds_per_clip = seconds_per_beat * beats_per_clip

    return int(target_duration / seconds_per_clip)


def generate_cut_times_from_beats(
    audio_analysis: AudioAnalysis,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    beats_per_cut: int = 4,
    use_downbeats: bool = True,
) -> list[float]:
    """Generate cut times based on beat structure.

    Creates evenly-spaced cut points aligned to beats.

    Args:
        audio_analysis: Beat analysis from music track
        start_time: Where to start generating cuts
        end_time: Where to stop (default: audio duration)
        beats_per_cut: Number of beats between each cut
        use_downbeats: If True, align to downbeats when possible

    Returns:
        List of cut times (seconds)
    """
    if end_time is None:
        end_time = audio_analysis.duration_seconds

    if use_downbeats and audio_analysis.downbeat_times:
        # Use downbeats directly
        cuts = [t for t in audio_analysis.downbeat_times if start_time <= t <= end_time]
    else:
        # Select every Nth beat
        beats = audio_analysis.beat_times
        cuts = [beats[i] for i in range(0, len(beats), beats_per_cut)
                if start_time <= beats[i] <= end_time]

    return cuts

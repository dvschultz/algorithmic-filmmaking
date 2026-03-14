"""Staccato — beat-driven sequencing with onset-strength visual contrast.

Assigns clips to beat intervals from a music track. Onset strength at each
cut point determines how visually different the next clip should be from the
previous one, measured by DINOv2 embedding cosine distance.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.analysis.audio import AudioAnalysis

logger = logging.getLogger(__name__)


@dataclass
class StaccatoSlot:
    """A single slot in the beat-driven sequence.

    Attributes:
        start_time: Start time in the music (seconds)
        end_time: End time in the music (seconds)
        onset_strength: Normalized onset strength at this cut [0, 1]
        clip_index: Index into the original clips list
        needs_loop: True if clip is shorter than the slot duration
    """
    start_time: float
    end_time: float
    onset_strength: float
    clip_index: int = -1
    needs_loop: bool = False

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def generate_beat_slots(
    audio_analysis: AudioAnalysis,
    strategy: str = "onsets",
) -> list[StaccatoSlot]:
    """Generate time slots from the audio's beat/onset structure.

    Args:
        audio_analysis: Analyzed music file
        strategy: "beats", "downbeats", or "onsets"

    Returns:
        List of StaccatoSlot with timing and onset strength
    """
    if strategy == "downbeats":
        cut_times = audio_analysis.downbeat_times
    elif strategy == "beats":
        cut_times = audio_analysis.beat_times
    else:  # onsets
        cut_times = audio_analysis.onset_times

    if not cut_times:
        cut_times = audio_analysis.beat_times
    if not cut_times:
        return []

    duration = audio_analysis.duration_seconds
    slots = []

    for i in range(len(cut_times)):
        start = cut_times[i]
        end = cut_times[i + 1] if i + 1 < len(cut_times) else duration

        # Skip very short slots (< 0.1s)
        if end - start < 0.1:
            continue

        strength = audio_analysis.onset_strength_at(start)
        slots.append(StaccatoSlot(
            start_time=start,
            end_time=end,
            onset_strength=strength,
        ))

    return slots


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance between two L2-normalized vectors.

    Since DINOv2 embeddings are already L2-normalized:
    distance = 1 - dot(a, b)
    """
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    return float(1.0 - np.dot(a_arr, b_arr))


def _select_clip_for_slot(
    slot: StaccatoSlot,
    prev_embedding: Optional[list[float]],
    clips_with_embeddings: list[tuple],
    clip_durations: list[float],
) -> int:
    """Select the best clip for a beat slot based on onset strength.

    Stronger onset → pick clip with greater visual distance from previous.
    Also prefers clips longer than the slot duration to avoid looping.

    Args:
        slot: The beat slot to fill
        prev_embedding: Embedding of the previously placed clip (None for first)
        clips_with_embeddings: List of (index, embedding) tuples
        clip_durations: Duration of each clip in seconds

    Returns:
        Index of the selected clip in the original clips list
    """
    if not clips_with_embeddings:
        return 0

    if prev_embedding is None:
        # First clip: pick randomly from the pool
        return clips_with_embeddings[0][0]

    # Calculate distances from previous clip to all candidates
    distances = []
    for idx, emb in clips_with_embeddings:
        if emb is not None:
            dist = _cosine_distance(prev_embedding, emb)
        else:
            dist = 0.5  # Default for clips without embeddings
        distances.append((idx, dist))

    # Target distance based on onset strength (linear mapping)
    # Max cosine distance for normalized vectors is ~2.0 but typical range is 0-1
    target_distance = slot.onset_strength

    # Score each clip: lower is better
    scored = []
    for idx, dist in distances:
        # Primary: how close to target distance
        distance_score = abs(dist - target_distance)

        # Secondary: prefer clips that don't need looping (small bonus)
        duration_penalty = 0.0
        if idx < len(clip_durations) and clip_durations[idx] < slot.duration:
            duration_penalty = 0.05  # Small penalty for needing loop

        scored.append((idx, distance_score + duration_penalty))

    # Return the best-scoring clip
    scored.sort(key=lambda x: x[1])
    return scored[0][0]


def generate_staccato_sequence(
    clips: list,
    audio_analysis: AudioAnalysis,
    strategy: str = "onsets",
    progress_cb=None,
) -> list[tuple]:
    """Generate a beat-driven sequence using onset strength for visual contrast.

    Args:
        clips: List of (Clip, Source) tuples with DINOv2 embeddings
        audio_analysis: Analyzed music file with onset strengths
        strategy: "beats", "downbeats", or "onsets"
        progress_cb: Optional callback(current, total) for progress

    Returns:
        List of (Clip, Source) tuples in sequence order.
        Clips may repeat. Clip in_point/out_point should be set by caller
        based on slot timing.
    """
    if not clips:
        return []

    # Build the beat slot schedule
    slots = generate_beat_slots(audio_analysis, strategy)
    if not slots:
        logger.warning("No beat slots generated from audio analysis")
        return []

    # Prepare embeddings and durations
    clips_with_embeddings = []
    clip_durations = []
    for i, (clip, source) in enumerate(clips):
        emb = getattr(clip, 'embedding', None)
        clips_with_embeddings.append((i, emb))

        fps = getattr(source, 'fps', 24.0) or 24.0
        start = getattr(clip, 'start_frame', 0)
        end = getattr(clip, 'end_frame', start + int(fps))
        clip_durations.append((end - start) / fps)

    # Assign clips to slots
    result = []
    prev_embedding = None

    for slot_idx, slot in enumerate(slots):
        if progress_cb:
            progress_cb(slot_idx, len(slots))

        best_idx = _select_clip_for_slot(
            slot, prev_embedding, clips_with_embeddings, clip_durations,
        )

        clip, source = clips[best_idx]
        result.append((clip, source))

        # Update previous embedding for next iteration
        prev_embedding = getattr(clip, 'embedding', None)

    if progress_cb:
        progress_cb(len(slots), len(slots))

    return result

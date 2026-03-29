"""Staccato — beat-driven sequencing with onset-strength visual contrast.

Assigns clips to beat intervals from a music track. Onset strength at each
cut point determines how visually different the next clip should be from the
previous one, measured by DINOv2 embedding cosine distance.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.analysis.audio import AudioAnalysis

logger = logging.getLogger(__name__)


@dataclass
class StaccatoSlotDebug:
    """Debug metadata for a single slot assignment."""

    slot_index: int
    start_time: float
    end_time: float
    onset_strength: float
    clip_id: str
    clip_name: str
    source_filename: str
    cosine_distance: Optional[float]  # None for first slot
    target_distance: float
    distance_score: float
    needs_loop: bool


@dataclass
class StaccatoDebugInfo:
    """Complete debug information for a Staccato generation run."""

    strategy: str
    total_slots: int
    total_clips_available: int
    slots: list[StaccatoSlotDebug] = field(default_factory=list)


class StaccatoResult:
    """Result wrapper that acts as a list but also carries debug info."""

    def __init__(
        self,
        sequence: list[tuple],
        debug: Optional[StaccatoDebugInfo] = None,
    ):
        self.sequence = sequence
        self.debug = debug

    def __iter__(self):
        return iter(self.sequence)

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        return self.sequence[index]

    def __bool__(self):
        return bool(self.sequence)

    def __eq__(self, other):
        if isinstance(other, list):
            return self.sequence == other
        if isinstance(other, StaccatoResult):
            return self.sequence == other.sequence
        return NotImplemented

    def __repr__(self):
        debug_str = f", debug={self.debug.strategy}/{self.debug.total_slots} slots" if self.debug else ""
        return f"StaccatoResult({len(self.sequence)} clips{debug_str})"


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
) -> tuple[int, Optional[float], float]:
    """Select the best clip for a beat slot based on onset strength.

    Stronger onset → pick clip with greater visual distance from previous.
    Also prefers clips longer than the slot duration to avoid looping.

    Args:
        slot: The beat slot to fill
        prev_embedding: Embedding of the previously placed clip (None for first)
        clips_with_embeddings: List of (index, embedding) tuples
        clip_durations: Duration of each clip in seconds

    Returns:
        Tuple of (clip_index, cosine_distance, distance_score).
        cosine_distance is None when prev_embedding is None (first slot).
    """
    if not clips_with_embeddings:
        return (0, None, 0.0)

    if prev_embedding is None:
        # First clip: pick first from the pool
        return (clips_with_embeddings[0][0], None, 0.0)

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

        scored.append((idx, dist, distance_score + duration_penalty))

    # Return the best-scoring clip
    scored.sort(key=lambda x: x[2])
    best_idx, best_distance, best_score = scored[0]
    return (best_idx, best_distance, best_score)


def generate_staccato_sequence(
    clips: list,
    audio_analysis: AudioAnalysis,
    strategy: str = "onsets",
    progress_cb=None,
) -> StaccatoResult:
    """Generate a beat-driven sequence using onset strength for visual contrast.

    Args:
        clips: List of (Clip, Source) tuples with DINOv2 embeddings
        audio_analysis: Analyzed music file with onset strengths
        strategy: "beats", "downbeats", or "onsets"
        progress_cb: Optional callback(current, total) for progress

    Returns:
        StaccatoResult wrapping the sequence list and debug metadata.
        Behaves like list[tuple] for backward compat (supports iter/len/getitem).
        Clips may repeat. Clip in_point/out_point should be set by caller
        based on slot timing.
    """
    if not clips:
        return StaccatoResult([])

    # Build the beat slot schedule
    slots = generate_beat_slots(audio_analysis, strategy)
    if not slots:
        logger.warning("No beat slots generated from audio analysis")
        return StaccatoResult([])

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

    debug = StaccatoDebugInfo(
        strategy=strategy,
        total_slots=len(slots),
        total_clips_available=len(clips),
    )

    # Assign clips to slots, exhausting all clips before repeating
    result = []
    prev_embedding = None
    used_indices: set[int] = set()

    for slot_idx, slot in enumerate(slots):
        if progress_cb:
            progress_cb(slot_idx, len(slots))

        # Filter to unused clips; reset pool when all clips have been used
        available = [(i, emb) for i, emb in clips_with_embeddings if i not in used_indices]
        if not available:
            used_indices.clear()
            available = clips_with_embeddings

        available_durations = clip_durations

        best_idx, cosine_dist, dist_score = _select_clip_for_slot(
            slot, prev_embedding, available, available_durations,
        )
        used_indices.add(best_idx)

        clip, source = clips[best_idx]
        result.append((clip, source))

        # Build debug entry
        clip_name = getattr(clip, 'name', None) or getattr(clip, 'id', f'clip_{best_idx}')
        source_file = getattr(source, 'file_path', None)
        source_filename = source_file.name if hasattr(source_file, 'name') else str(source_file or '')
        needs_loop = (
            best_idx < len(clip_durations) and clip_durations[best_idx] < slot.duration
        )

        debug.slots.append(StaccatoSlotDebug(
            slot_index=slot_idx,
            start_time=slot.start_time,
            end_time=slot.end_time,
            onset_strength=slot.onset_strength,
            clip_id=getattr(clip, 'id', f'clip_{best_idx}'),
            clip_name=str(clip_name),
            source_filename=source_filename,
            cosine_distance=cosine_dist,
            target_distance=slot.onset_strength,
            distance_score=dist_score,
            needs_loop=needs_loop,
        ))

        # Update previous embedding for next iteration
        prev_embedding = getattr(clip, 'embedding', None)

    if progress_cb:
        progress_cb(len(slots), len(slots))

    return StaccatoResult(result, debug=debug)

"""Algorithmic remix algorithms for video clip sequencing."""

import logging
import random
from typing import List, Tuple, Any, Optional
from core.remix.shuffle import constrained_shuffle
from core.analysis.color import get_primary_hue, rgb_to_hsv, compute_color_purity
from core.analysis.shots import SHOT_TYPES
from core.remix.audio_sync import (
    AlignmentSuggestion,
    suggest_beat_aligned_cuts,
    align_times_to_beats,
    calculate_beat_intervals,
    get_beats_in_range,
    estimate_clip_count_for_duration,
    generate_cut_times_from_beats,
)

__all__ = [
    "constrained_shuffle",
    "generate_sequence",
    # Audio sync
    "AlignmentSuggestion",
    "suggest_beat_aligned_cuts",
    "align_times_to_beats",
    "calculate_beat_intervals",
    "get_beats_in_range",
    "estimate_clip_count_for_duration",
    "generate_cut_times_from_beats",
]

logger = logging.getLogger(__name__)

# Shot type order for sorting (wide to close)
SHOT_TYPE_ORDER = {shot: i for i, shot in enumerate(SHOT_TYPES)}

# 10-class cinematography shot size → proximity score
_SHOT_SIZE_PROXIMITY = {
    "ELS": 1.0,
    "VLS": 2.0,
    "LS": 3.0,
    "MLS": 4.0,
    "MS": 5.0,
    "MCU": 6.0,
    "CU": 7.0,
    "BCU": 8.0,
    "ECU": 9.0,
    "Insert": 10.0,
}

# 5-class shot_type → proximity score (fallback)
_SHOT_TYPE_PROXIMITY = {
    "wide shot": 2.0,
    "full shot": 4.0,
    "medium shot": 5.0,
    "close-up": 7.0,
    "extreme close-up": 9.0,
}

# Default color purity threshold for color_cycle
COLOR_CYCLE_PURITY_THRESHOLD = 0.4


def _get_warmth_score(hue: float) -> float:
    """Calculate a warmth score from hue for warm_to_cool sorting.

    Warm colors (reds, oranges, yellows) get low scores.
    Cool colors (cyans, blues) get high scores.

    Args:
        hue: Hue value 0-360

    Returns:
        Warmth score where 0 = warmest (red), 1 = coolest (cyan)
    """
    # Calculate angular distance from cyan (180°), the coolest point
    # This gives us: red (0/360) = 180, cyan (180) = 0
    distance_from_cyan = abs(180 - hue)
    if distance_from_cyan > 180:
        distance_from_cyan = 360 - distance_from_cyan

    # Normalize to 0-1 where 0 = cool, 1 = warm
    warmth = distance_from_cyan / 180.0

    # Invert so warm colors have LOW scores for ascending sort
    return 1.0 - warmth


def generate_sequence(
    algorithm: str,
    clips: List[Tuple[Any, Any]],  # List of (Clip, Source) tuples
    clip_count: int,
    direction: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[Tuple[Any, Any]]:
    """
    Generate a sequence of clips using the specified algorithm.

    Args:
        algorithm: Algorithm name ("shuffle", "sequential", "color", "shot_type",
                   "duration", "duration_long", "duration_short")
        clips: List of (Clip, Source) tuples to sequence
        clip_count: Maximum number of clips to include
        direction: For color: "rainbow", "warm_to_cool", "cool_to_warm"
                   For duration: "short_first", "long_first"
        seed: Random seed for shuffle reproducibility (0 = random)

    Returns:
        Ordered list of (Clip, Source) tuples ready for timeline
    """
    clips_to_use = clips[:clip_count]

    if algorithm == "shuffle":
        # Set random seed if provided and non-zero
        if seed and seed > 0:
            random.seed(seed)

        # Constrained shuffle - no same source back-to-back
        result = constrained_shuffle(
            items=clips_to_use,
            get_category=lambda x: x[1].id,  # x is (Clip, Source), category by source
            max_consecutive=1,
        )

        # Reset random state
        if seed and seed > 0:
            random.seed()

        return result

    elif algorithm == "color":
        color_direction = direction or "rainbow"

        if color_direction == "rainbow":
            # Sort by primary hue (HSV color wheel order)
            def get_hue(item: Tuple[Any, Any]) -> float:
                clip, _ = item
                if clip.dominant_colors:
                    return get_primary_hue(clip.dominant_colors)
                return 0.0

            return sorted(clips_to_use, key=get_hue)

        elif color_direction == "warm_to_cool":
            # Sort warm colors first, cool colors last
            def get_warmth(item: Tuple[Any, Any]) -> float:
                clip, _ = item
                if clip.dominant_colors:
                    hue = get_primary_hue(clip.dominant_colors)
                    return _get_warmth_score(hue)
                return 0.5  # Neutral for clips without colors

            return sorted(clips_to_use, key=get_warmth)

        elif color_direction == "cool_to_warm":
            # Sort cool colors first, warm colors last
            def get_coolness(item: Tuple[Any, Any]) -> float:
                clip, _ = item
                if clip.dominant_colors:
                    hue = get_primary_hue(clip.dominant_colors)
                    return 1.0 - _get_warmth_score(hue)
                return 0.5  # Neutral for clips without colors

            return sorted(clips_to_use, key=get_coolness)

        else:
            # Default to rainbow
            def get_hue(item: Tuple[Any, Any]) -> float:
                clip, _ = item
                if clip.dominant_colors:
                    return get_primary_hue(clip.dominant_colors)
                return 0.0

            return sorted(clips_to_use, key=get_hue)

    elif algorithm == "shot_type":
        # Sort by shot type (wide -> medium -> close-up -> extreme close-up)
        def get_shot_order(item: Tuple[Any, Any]) -> int:
            clip, _ = item
            if clip.shot_type:
                return SHOT_TYPE_ORDER.get(clip.shot_type, 999)
            return 999  # Unknown shot types at end

        return sorted(clips_to_use, key=get_shot_order)

    elif algorithm == "duration":
        # Unified duration sort with direction parameter
        duration_direction = direction or "short_first"

        if duration_direction == "long_first":
            def get_duration(item: Tuple[Any, Any]) -> float:
                clip, source = item
                return -clip.duration_seconds(source.fps)  # Negative for descending
            return sorted(clips_to_use, key=get_duration)
        else:  # short_first
            def get_duration(item: Tuple[Any, Any]) -> float:
                clip, source = item
                return clip.duration_seconds(source.fps)
            return sorted(clips_to_use, key=get_duration)

    elif algorithm == "duration_long":
        # Legacy: Sort by duration (longest first)
        def get_duration(item: Tuple[Any, Any]) -> float:
            clip, source = item
            return -clip.duration_seconds(source.fps)  # Negative for descending

        return sorted(clips_to_use, key=get_duration)

    elif algorithm == "duration_short":
        # Legacy: Sort by duration (shortest first)
        def get_duration(item: Tuple[Any, Any]) -> float:
            clip, source = item
            return clip.duration_seconds(source.fps)

        return sorted(clips_to_use, key=get_duration)

    elif algorithm == "brightness":
        # Sort by average brightness (luminance)
        brightness_direction = direction or "bright_to_dark"

        # Auto-compute brightness for clips that don't have it cached
        _auto_compute_brightness(clips_to_use)

        def get_brightness(item: Tuple[Any, Any]) -> float:
            clip, _ = item
            val = clip.average_brightness if clip.average_brightness is not None else 0.5
            return -val if brightness_direction == "bright_to_dark" else val

        return sorted(clips_to_use, key=get_brightness)

    elif algorithm == "volume":
        # Sort by audio volume (RMS level in dB)
        volume_direction = direction or "quiet_to_loud"

        # Auto-compute volume for clips that don't have it cached
        _auto_compute_volume(clips_to_use)

        # Filter out clips without volume data (no audio track)
        clips_with_volume = [
            (clip, source) for clip, source in clips_to_use
            if clip.rms_volume is not None
        ]
        excluded = len(clips_to_use) - len(clips_with_volume)
        if excluded > 0:
            logger.info(f"Volume sort: excluded {excluded} clips (no audio)")

        if not clips_with_volume:
            logger.warning("No clips with audio data for volume sort")
            return clips_to_use

        def get_volume(item: Tuple[Any, Any]) -> float:
            clip, _ = item
            val = clip.rms_volume if clip.rms_volume is not None else -60.0
            return val if volume_direction == "quiet_to_loud" else -val

        return sorted(clips_with_volume, key=get_volume)

    elif algorithm == "proximity":
        # Sort by camera-to-subject distance (proximity score)
        proximity_direction = direction or "far_to_close"

        def get_proximity(item: Tuple[Any, Any]) -> float:
            clip, _ = item
            # Prefer 10-class cinematography shot_size
            if clip.cinematography and clip.cinematography.shot_size:
                score = _SHOT_SIZE_PROXIMITY.get(clip.cinematography.shot_size, 5.0)
            elif clip.shot_type:
                score = _SHOT_TYPE_PROXIMITY.get(clip.shot_type, 5.0)
            else:
                score = 5.0  # Middle default
            return score if proximity_direction == "far_to_close" else -score

        return sorted(clips_to_use, key=get_proximity)

    elif algorithm == "similarity_chain":
        from core.remix.similarity_chain import similarity_chain
        # Auto-compute embeddings for clips that don't have them
        _auto_compute_embeddings(clips_to_use)
        return similarity_chain(clips_to_use, start_clip_id=None)

    elif algorithm == "match_cut":
        from core.remix.match_cut import match_cut_chain
        # Auto-compute boundary embeddings
        _auto_compute_boundary_embeddings(clips_to_use)
        return match_cut_chain(clips_to_use, start_clip_id=None)

    elif algorithm == "color_cycle":
        # Filter to clips with strong color identity, then cycle through hue wheel
        color_cycle_direction = direction or "spectrum"

        # Filter to clips with dominant_colors and sufficient purity
        filtered = []
        for clip, source in clips_to_use:
            if clip.dominant_colors:
                purity = compute_color_purity(clip.dominant_colors)
                if purity >= COLOR_CYCLE_PURITY_THRESHOLD:
                    filtered.append((clip, source))

        excluded = len(clips_to_use) - len(filtered)
        if excluded > 0:
            logger.info(
                f"Color cycle: {len(filtered)} included, {excluded} excluded (low purity)"
            )

        if not filtered:
            logger.warning("No clips with sufficient color purity for color cycle")
            return clips_to_use

        # Sort by primary hue
        def get_hue(item: Tuple[Any, Any]) -> float:
            clip, _ = item
            if clip.dominant_colors:
                return get_primary_hue(clip.dominant_colors)
            return 0.0

        sorted_by_hue = sorted(filtered, key=get_hue)

        if color_cycle_direction == "complementary":
            # Interleave from bottom and top for maximum contrast
            result = []
            lo, hi = 0, len(sorted_by_hue) - 1
            toggle = True
            while lo <= hi:
                if toggle:
                    result.append(sorted_by_hue[lo])
                    lo += 1
                else:
                    result.append(sorted_by_hue[hi])
                    hi -= 1
                toggle = not toggle
            return result
        else:
            # spectrum: linear hue progression
            return sorted_by_hue

    else:
        # Sequential - use original order
        return clips_to_use


def _auto_compute_brightness(clips: List[Tuple[Any, Any]]) -> None:
    """Compute brightness for clips that don't have it cached."""
    from core.analysis.color import get_average_brightness

    for clip, source in clips:
        if clip.average_brightness is None:
            try:
                brightness = get_average_brightness(
                    source_path=source.file_path,
                    start_frame=clip.start_frame,
                    end_frame=clip.end_frame,
                    fps=source.fps,
                )
                clip.average_brightness = brightness
            except Exception as e:
                logger.warning(f"Failed to compute brightness for clip {clip.id}: {e}")
                clip.average_brightness = 0.5


def _auto_compute_volume(clips: List[Tuple[Any, Any]]) -> None:
    """Compute volume for clips that don't have it cached."""
    from core.analysis.audio import extract_clip_volume

    for clip, source in clips:
        if clip.rms_volume is None:
            try:
                start_seconds = clip.start_frame / source.fps
                duration_seconds = clip.duration_seconds(source.fps)
                volume = extract_clip_volume(
                    source_path=source.file_path,
                    start_seconds=start_seconds,
                    duration_seconds=duration_seconds,
                )
                clip.rms_volume = volume  # None if no audio track
            except Exception as e:
                logger.warning(f"Failed to compute volume for clip {clip.id}: {e}")


def _auto_compute_embeddings(clips: List[Tuple[Any, Any]]) -> None:
    """Compute CLIP embeddings for clips that don't have them."""
    from core.analysis.embeddings import extract_clip_embeddings_batch

    needs_embedding = [
        (clip, source) for clip, source in clips
        if clip.embedding is None and clip.thumbnail_path
    ]
    if not needs_embedding:
        return

    thumbnail_paths = [clip.thumbnail_path for clip, _ in needs_embedding]
    try:
        embeddings = extract_clip_embeddings_batch(thumbnail_paths)
        for (clip, _), emb in zip(needs_embedding, embeddings):
            clip.embedding = emb
    except Exception as e:
        logger.warning(f"Failed to compute embeddings: {e}")


def _auto_compute_boundary_embeddings(clips: List[Tuple[Any, Any]]) -> None:
    """Compute first/last frame CLIP embeddings for clips that don't have them."""
    from core.analysis.embeddings import extract_boundary_embeddings

    for clip, source in clips:
        if clip.first_frame_embedding is None or clip.last_frame_embedding is None:
            try:
                first_emb, last_emb = extract_boundary_embeddings(
                    source_path=source.file_path,
                    start_frame=clip.start_frame,
                    end_frame=clip.end_frame,
                    fps=source.fps,
                )
                clip.first_frame_embedding = first_emb
                clip.last_frame_embedding = last_emb
            except Exception as e:
                logger.warning(
                    f"Failed to compute boundary embeddings for clip {clip.id}: {e}"
                )

"""Algorithmic remix algorithms for video clip sequencing."""

import random
from typing import List, Tuple, Any, Optional
from core.remix.shuffle import constrained_shuffle
from core.analysis.color import get_primary_hue
from core.analysis.shots import SHOT_TYPES

__all__ = ["constrained_shuffle", "generate_sequence"]

# Shot type order for sorting (wide to close)
SHOT_TYPE_ORDER = {shot: i for i, shot in enumerate(SHOT_TYPES)}


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

    else:
        # Sequential - use original order
        return clips_to_use

"""Algorithmic remix algorithms for video clip sequencing."""

from typing import List, Tuple, Any
from core.remix.shuffle import constrained_shuffle
from core.analysis.color import get_primary_hue
from core.analysis.shots import SHOT_TYPES

__all__ = ["constrained_shuffle", "generate_sequence"]

# Shot type order for sorting (wide to close)
SHOT_TYPE_ORDER = {shot: i for i, shot in enumerate(SHOT_TYPES)}


def generate_sequence(
    algorithm: str,
    clips: List[Tuple[Any, Any]],  # List of (Clip, Source) tuples
    clip_count: int,
) -> List[Tuple[Any, Any]]:
    """
    Generate a sequence of clips using the specified algorithm.

    Args:
        algorithm: Algorithm name ("shuffle", "sequential", "color", "shot_type",
                   "duration_long", "duration_short")
        clips: List of (Clip, Source) tuples to sequence
        clip_count: Maximum number of clips to include

    Returns:
        Ordered list of (Clip, Source) tuples ready for timeline
    """
    clips_to_use = clips[:clip_count]

    if algorithm == "shuffle":
        # Constrained shuffle - no same source back-to-back
        return constrained_shuffle(
            items=clips_to_use,
            get_category=lambda x: x[1].id,  # x is (Clip, Source), category by source
            max_consecutive=1,
        )
    elif algorithm == "color":
        # Sort by primary hue (HSV color wheel order)
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

    elif algorithm == "duration_long":
        # Sort by duration (longest first)
        def get_duration(item: Tuple[Any, Any]) -> float:
            clip, source = item
            return -clip.duration_seconds(source.fps)  # Negative for descending

        return sorted(clips_to_use, key=get_duration)

    elif algorithm == "duration_short":
        # Sort by duration (shortest first)
        def get_duration(item: Tuple[Any, Any]) -> float:
            clip, source = item
            return clip.duration_seconds(source.fps)

        return sorted(clips_to_use, key=get_duration)

    else:
        # Sequential - use original order
        return clips_to_use

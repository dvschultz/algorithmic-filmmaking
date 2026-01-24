"""Algorithmic remix algorithms for video clip sequencing."""

from typing import List, Tuple, Any
from core.remix.shuffle import constrained_shuffle

__all__ = ["constrained_shuffle", "generate_sequence"]


def generate_sequence(
    algorithm: str,
    clips: List[Tuple[Any, Any]],  # List of (Clip, Source) tuples
    clip_count: int,
) -> List[Tuple[Any, Any]]:
    """
    Generate a sequence of clips using the specified algorithm.

    Args:
        algorithm: Algorithm name ("shuffle" or "sequential")
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
    else:
        # Sequential - use original order
        return clips_to_use

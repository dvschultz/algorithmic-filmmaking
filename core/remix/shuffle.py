"""Constrained shuffle algorithm for video clip remixing."""

import random
from typing import TypeVar, Callable, List

T = TypeVar("T")


def constrained_shuffle(
    items: List[T],
    get_category: Callable[[T], str],
    max_consecutive: int = 1,
    max_attempts: int = 1000,
) -> List[T]:
    """
    Shuffle items with constraint: no more than max_consecutive
    items from the same category in a row.

    Uses rejection sampling - retry if constraint violated.
    Falls back to greedy repair if rejection sampling fails.

    Args:
        items: List of items to shuffle
        get_category: Function to get category string from an item
        max_consecutive: Maximum consecutive items from same category
        max_attempts: Number of shuffle attempts before falling back

    Returns:
        Shuffled list satisfying the constraint
    """
    if len(items) <= 1:
        return items.copy()

    # Try rejection sampling
    for _ in range(max_attempts):
        shuffled = items.copy()
        random.shuffle(shuffled)

        if _check_constraints(shuffled, get_category, max_consecutive):
            return shuffled

    # Fallback: greedy repair
    return _greedy_repair(items, get_category, max_consecutive)


def _check_constraints(
    items: List[T],
    get_category: Callable[[T], str],
    max_consecutive: int,
) -> bool:
    """Check if sequence satisfies consecutive constraint."""
    if len(items) <= 1:
        return True

    consecutive = 1
    prev_category = get_category(items[0])

    for item in items[1:]:
        category = get_category(item)
        if category == prev_category:
            consecutive += 1
            if consecutive > max_consecutive:
                return False
        else:
            consecutive = 1
        prev_category = category

    return True


def _greedy_repair(
    items: List[T],
    get_category: Callable[[T], str],
    max_consecutive: int,
) -> List[T]:
    """
    Greedy algorithm when rejection sampling fails.
    Build sequence by always picking valid next item.
    """
    remaining = items.copy()
    random.shuffle(remaining)
    result = []

    while remaining:
        # Get recent categories
        recent_categories = []
        for item in result[-max_consecutive:]:
            recent_categories.append(get_category(item))

        # Find valid candidates (different category from recent streak)
        if len(recent_categories) == max_consecutive and len(set(recent_categories)) == 1:
            # We're at max consecutive of same category - must pick different
            blocked_category = recent_categories[0]
            valid = [item for item in remaining if get_category(item) != blocked_category]
        else:
            valid = remaining

        if not valid:
            # No valid options - just take any (constraint can't be satisfied)
            valid = remaining

        # Pick random from valid options
        chosen = random.choice(valid)
        result.append(chosen)
        remaining.remove(chosen)

    return result


def shuffle_clips(
    clips: list,
    max_consecutive_same_source: int = 1,
) -> list:
    """
    Shuffle clips ensuring no more than N consecutive clips from the same source.

    Args:
        clips: List of Clip objects with source_id attribute
        max_consecutive_same_source: Maximum consecutive clips from same source

    Returns:
        Shuffled list of clips
    """
    return constrained_shuffle(
        items=clips,
        get_category=lambda clip: clip.source_id,
        max_consecutive=max_consecutive_same_source,
    )

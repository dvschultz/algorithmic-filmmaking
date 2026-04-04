"""Gaze-based sequencing algorithms for Eyes Without a Face dialog.

Three modes:
- Eyeline Match: pair clips with approximately negated yaw angles (shot/reverse-shot)
- Gaze Filter: keep clips matching a selected gaze category
- Gaze Rotation: arrange clips in monotonically progressing angle order
"""

import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def eyeline_match(
    clips_with_sources: List[Tuple[Any, Any]],
    tolerance: float = 20.0,
) -> List[Tuple[Any, Any]]:
    """Pair clips with approximately negated yaw for eyeline matching.

    Simulates shot/reverse-shot: a clip looking left (-yaw) is paired with
    a clip looking right (+yaw) of similar magnitude.

    Args:
        clips_with_sources: List of (Clip, Source) tuples
        tolerance: Maximum difference in degrees for abs(yaw_a + yaw_b)
            to consider a pair (default 20.0)

    Returns:
        List of (Clip, Source) tuples: paired clips interleaved (A, B, A, B...),
        followed by unmatched clips, followed by clips without gaze data.
    """
    with_gaze = []
    without_gaze = []
    for clip, source in clips_with_sources:
        if clip.gaze_yaw is not None:
            with_gaze.append((clip, source))
        else:
            without_gaze.append((clip, source))

    if not with_gaze:
        return list(clips_with_sources)

    # Sort by absolute yaw so we try to pair the most extreme angles first
    with_gaze.sort(key=lambda item: abs(item[0].gaze_yaw), reverse=True)

    used = set()
    pairs = []

    for i, (clip_a, source_a) in enumerate(with_gaze):
        if i in used:
            continue

        best_j = None
        best_score = float("inf")

        for j, (clip_b, source_b) in enumerate(with_gaze):
            if j in used or j == i:
                continue
            # Perfect eyeline match: yaw_a + yaw_b = 0
            # e.g., -25 (looking left) + 25 (looking right) = 0
            score = abs(clip_a.gaze_yaw + clip_b.gaze_yaw)
            if score <= tolerance and score < best_score:
                best_score = score
                best_j = j

        if best_j is not None:
            used.add(i)
            used.add(best_j)
            pairs.append((i, best_j))

    # Build result: paired clips interleaved
    result = []
    for i, j in pairs:
        result.append(with_gaze[i])
        result.append(with_gaze[j])

    # Unmatched clips with gaze data
    unmatched = [
        with_gaze[k] for k in range(len(with_gaze)) if k not in used
    ]
    result.extend(unmatched)

    # Clips without gaze data at the end
    result.extend(without_gaze)

    logger.info(
        "Eyeline match: %d pairs, %d unmatched, %d without gaze",
        len(pairs), len(unmatched), len(without_gaze),
    )
    return result


def gaze_filter(
    clips_with_sources: List[Tuple[Any, Any]],
    category: str,
) -> List[Tuple[Any, Any]]:
    """Filter clips by gaze category.

    Args:
        clips_with_sources: List of (Clip, Source) tuples
        category: Gaze category to keep (e.g., "at_camera", "looking_left")

    Returns:
        List of (Clip, Source) tuples: matching clips first,
        then non-matching clips appended at end.
    """
    matching = []
    non_matching = []

    for clip, source in clips_with_sources:
        if clip.gaze_category == category:
            matching.append((clip, source))
        else:
            non_matching.append((clip, source))

    logger.info(
        "Gaze filter (%s): %d matching, %d non-matching",
        category, len(matching), len(non_matching),
    )
    return matching + non_matching


def gaze_rotation(
    clips_with_sources: List[Tuple[Any, Any]],
    axis: str = "yaw",
    range_start: float = -30.0,
    range_end: float = 30.0,
    ascending: bool = True,
) -> List[Tuple[Any, Any]]:
    """Arrange clips in monotonically progressing angle order.

    Selects clips whose angle falls within the specified range and sorts
    them by their actual angle value on the given axis. Clips outside
    the range are appended after in-range clips. Clips without gaze data
    are appended at the very end.

    Args:
        clips_with_sources: List of (Clip, Source) tuples
        axis: "yaw" or "pitch"
        range_start: Start of the angle range in degrees
        range_end: End of the angle range in degrees
        ascending: If True, angles progress from range_start to range_end;
            if False, reversed

    Returns:
        List of (Clip, Source) tuples in monotonically progressing angle order,
        followed by out-of-range clips, followed by clips without gaze data.
    """
    lo = min(range_start, range_end)
    hi = max(range_start, range_end)

    in_range = []
    out_of_range = []
    without_gaze = []

    for clip, source in clips_with_sources:
        angle = clip.gaze_yaw if axis == "yaw" else clip.gaze_pitch
        if angle is not None:
            if lo <= angle <= hi:
                in_range.append((clip, source, angle))
            else:
                out_of_range.append((clip, source))
        else:
            without_gaze.append((clip, source))

    if not in_range and not out_of_range:
        return list(clips_with_sources)

    # Sort in-range clips by their actual angle (guarantees monotonicity)
    in_range.sort(key=lambda item: item[2])

    result = [(clip, source) for clip, source, _ in in_range]

    if not ascending:
        result.reverse()

    result.extend(out_of_range)
    result.extend(without_gaze)

    logger.info(
        "Gaze rotation (%s, %.1f to %.1f, %s): %d in-range, %d out-of-range, %d without gaze",
        axis, range_start, range_end, "asc" if ascending else "desc",
        len(in_range), len(out_of_range), len(without_gaze),
    )
    return result

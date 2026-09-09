"""Helpers shared by the library item models."""

from __future__ import annotations


def runs(rows: list[int]) -> list[tuple[int, int]]:
    """Collapse sorted row numbers into inclusive contiguous (first, last) runs."""
    result: list[tuple[int, int]] = []
    if not rows:
        return result
    start = prev = rows[0]
    for row in rows[1:]:
        if row == prev + 1:
            prev = row
            continue
        result.append((start, prev))
        start = prev = row
    result.append((start, prev))
    return result

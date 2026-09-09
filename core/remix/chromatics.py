"""Chromatics: order clips along a hue gradient or alternate complementary hues.

The definition is deterministic and unseeded. Clips lacking color data are
handled explicitly by ``no_color_handling`` and reported in proposal notes.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Mapping, Sequence

from core.remix.engine import (
    AlgorithmDefinition, ClipInput, ParameterSpec, ProposedEntry, SequenceProposal,
)

logger = logging.getLogger(__name__)

COLOR_DIRECTIONS = ("rainbow", "warm_to_cool", "cool_to_warm", "complementary")
NO_COLOR_HANDLING = ("append_end", "exclude", "sort_inline")


def warmth_score(hue: float) -> float:
    """0 = warmest (red), 1 = coolest (cyan)."""
    distance_from_cyan = abs(180 - hue)
    if distance_from_cyan > 180:
        distance_from_cyan = 360 - distance_from_cyan
    return 1.0 - distance_from_cyan / 180.0


def _hue(item: ClipInput) -> float:
    from core.analysis.color import get_primary_hue

    clip, _ = item
    return get_primary_hue(clip.dominant_colors) if clip.dominant_colors else 0.0


def _warmth(item: ClipInput) -> float:
    clip, _ = item
    return warmth_score(_hue(item)) if clip.dominant_colors else 0.5


def _coolness(item: ClipInput) -> float:
    clip, _ = item
    return 1.0 - warmth_score(_hue(item)) if clip.dominant_colors else 0.5


def _interleave_extremes(items: list[ClipInput]) -> list[ClipInput]:
    result: list[ClipInput] = []
    lo, hi = 0, len(items) - 1
    take_low = True
    while lo <= hi:
        if take_low:
            result.append(items[lo])
            lo += 1
        else:
            result.append(items[hi])
            hi -= 1
        take_low = not take_low
    return result


def order_by_color(
    clips: Sequence[ClipInput], direction: str = "rainbow", no_color_handling: str = "append_end",
) -> tuple[list[ClipInput], list[str]]:
    """Pure Chromatics ordering. Returns the order and human-readable notes."""
    if direction not in COLOR_DIRECTIONS:
        raise ValueError(f"Unknown Chromatics direction: {direction!r}")
    if no_color_handling not in NO_COLOR_HANDLING:
        raise ValueError(f"Unknown no-color handling: {no_color_handling!r}")
    with_colors = [item for item in clips if item[0].dominant_colors]
    without_colors = [item for item in clips if not item[0].dominant_colors]
    notes: list[str] = []
    if without_colors:
        notes.append(
            f"{len(without_colors)} clips lack color data (handling: {no_color_handling})"
        )
        logger.info("Chromatics: %s", notes[-1])
    if not with_colors:
        notes.append("No clips have color data; original order kept")
        logger.warning("No clips with color data for Chromatics sort")
        return ([] if no_color_handling == "exclude" else list(clips)), notes

    if direction == "warm_to_cool":
        ordered = sorted(with_colors, key=_warmth)
    elif direction == "cool_to_warm":
        ordered = sorted(with_colors, key=_coolness)
    elif direction == "complementary":
        ordered = _interleave_extremes(sorted(with_colors, key=_hue))
    else:
        ordered = sorted(with_colors, key=_hue)

    if no_color_handling == "exclude":
        return ordered, notes
    if no_color_handling == "sort_inline":
        everything = with_colors + without_colors
        if direction == "warm_to_cool":
            return sorted(everything, key=_warmth), notes
        if direction == "cool_to_warm":
            return sorted(everything, key=_coolness), notes
        # Complementary falls back to rainbow order when colorless clips are inlined.
        return sorted(everything, key=_hue), notes
    return ordered + without_colors, notes


class ChromaticsDefinition(AlgorithmDefinition):
    key = "color"
    version = 1
    prerequisites = ("colors",)
    seeded = False
    parameters = (
        ParameterSpec(
            "direction", "string", "rainbow",
            "Hue progression: rainbow, warm_to_cool, cool_to_warm, or complementary",
            choices=COLOR_DIRECTIONS,
        ),
        ParameterSpec(
            "no_color_handling", "string", "append_end",
            "Clips without color data: append_end, exclude, or sort_inline",
            choices=NO_COLOR_HANDLING,
        ),
    )

    def legacy_parameters(self, *, direction=None, no_color_handling=None, transform_options=None) -> dict[str, Any]:
        parameters: dict[str, Any] = {}
        if direction is not None:
            parameters["direction"] = direction
        if no_color_handling is not None:
            parameters["no_color_handling"] = no_color_handling
        return parameters

    def generate(
        self, inputs: Sequence[ClipInput], parameters: Mapping[str, Any], rng: random.Random | None, context=None,
    ) -> SequenceProposal:
        ordered, notes = order_by_color(
            inputs, parameters["direction"], parameters["no_color_handling"],
        )
        return SequenceProposal(
            "ordering",
            tuple(ProposedEntry(clip.id, source.id) for clip, source in ordered),
            notes=tuple(notes),
        )

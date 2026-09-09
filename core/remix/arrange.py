"""Arrange-family definitions: deterministic orderings over clip metadata.

Each definition here is unseeded and produces an ordering proposal. The
brightness and volume definitions resolve their scalar prerequisites on
detached inputs in ``prepare`` so generation stays pure.
"""

from __future__ import annotations

import logging
from threading import Event
from typing import Any, Sequence

from core.remix.engine import (
    AlgorithmDefinition, ClipInput, ParameterSpec, ProposedEntry, SequenceProposal,
)

logger = logging.getLogger(__name__)

# 10-class cinematography shot size → proximity score
SHOT_SIZE_PROXIMITY = {
    "ELS": 1.0, "VLS": 2.0, "LS": 3.0, "MLS": 4.0, "MS": 5.0,
    "MCU": 6.0, "CU": 7.0, "BCU": 8.0, "ECU": 9.0, "Insert": 10.0,
}

# 5-class shot_type → proximity score (fallback)
SHOT_TYPE_PROXIMITY = {
    "wide shot": 2.0, "full shot": 4.0, "medium shot": 5.0,
    "close-up": 7.0, "extreme close-up": 9.0,
}

GAZE_DIRECTIONS = ("left_to_right", "right_to_left", "up_to_down", "down_to_up")


def _ordering(ordered: Sequence[ClipInput], notes: Sequence[str] = ()) -> SequenceProposal:
    return SequenceProposal(
        "ordering",
        tuple(ProposedEntry(clip.id, source.id) for clip, source in ordered),
        notes=tuple(notes),
    )


def _direction_spec(choices: tuple[str, ...], description: str) -> ParameterSpec:
    return ParameterSpec("direction", "string", choices[0], description, choices=choices)


class _DirectionalDefinition(AlgorithmDefinition):
    """Shared legacy translation for algorithms whose only option is a direction."""

    def legacy_parameters(self, *, direction=None, no_color_handling=None, transform_options=None) -> dict[str, Any]:
        return {"direction": direction} if direction is not None else {}


class SequentialDefinition(AlgorithmDefinition):
    key = "sequential"
    version = 1

    def generate(self, inputs, parameters, rng):
        return _ordering(inputs)


class DurationDefinition(_DirectionalDefinition):
    key = "duration"
    version = 1
    parameters = (_direction_spec(("short_first", "long_first"), "Order by clip duration"),)

    def generate(self, inputs, parameters, rng):
        longest_first = parameters["direction"] == "long_first"
        ordered = sorted(
            inputs,
            key=lambda item: item[0].duration_seconds(item[1].fps),
            reverse=longest_first,
        )
        return _ordering(ordered)


class ShotTypeDefinition(AlgorithmDefinition):
    key = "shot_type"
    version = 1
    prerequisites = ("shots",)

    def generate(self, inputs, parameters, rng):
        from core.analysis.shots import SHOT_TYPES

        order = {shot: i for i, shot in enumerate(SHOT_TYPES)}
        ordered = sorted(inputs, key=lambda item: order.get(item[0].shot_type, 999) if item[0].shot_type else 999)
        return _ordering(ordered)


def proximity_score(clip: Any) -> float:
    if clip.cinematography and clip.cinematography.shot_size:
        return SHOT_SIZE_PROXIMITY.get(clip.cinematography.shot_size, 5.0)
    if clip.shot_type:
        return SHOT_TYPE_PROXIMITY.get(clip.shot_type, 5.0)
    return 5.0


class ProximityDefinition(_DirectionalDefinition):
    key = "proximity"
    version = 1
    prerequisites = ("shots", "cinematography")
    parameters = (_direction_spec(("far_to_close", "close_to_far"), "Camera-to-subject distance progression"),)

    def generate(self, inputs, parameters, rng):
        sign = 1.0 if parameters["direction"] == "far_to_close" else -1.0
        return _ordering(sorted(inputs, key=lambda item: sign * proximity_score(item[0])))


class _ScalarDefinition(_DirectionalDefinition):
    operation: str = ""

    def prepare(self, inputs, parameters, *, cancel_event: Event | None = None):
        # The package-level helpers are the patchable prerequisite seam shared
        # with the legacy dispatcher and its tests.
        import core.remix as remix

        compute = remix._auto_compute_brightness if self.operation == "brightness" else remix._auto_compute_volume
        return compute(list(inputs), cancel_event=cancel_event)

    def generate(self, inputs, parameters, rng):
        from core.remix.scalar_inputs import sort_scalar_inputs

        ordered = sort_scalar_inputs(list(inputs), self.operation, direction=parameters["direction"])  # type: ignore[arg-type]
        notes = []
        dropped = len(inputs) - len(ordered)
        if dropped:
            notes.append(f"{dropped} clips lack {self.operation} data and were excluded")
        return _ordering(ordered, notes)


class BrightnessDefinition(_ScalarDefinition):
    key = "brightness"
    version = 1
    operation = "brightness"
    prerequisites = ("brightness",)
    parameters = (_direction_spec(("bright_to_dark", "dark_to_bright"), "Luminance progression"),)


class VolumeDefinition(_ScalarDefinition):
    key = "volume"
    version = 1
    operation = "volume"
    prerequisites = ("volume",)
    parameters = (_direction_spec(("quiet_to_loud", "loud_to_quiet"), "Loudness progression"),)


def _split_gaze(inputs: Sequence[ClipInput]) -> tuple[list[ClipInput], list[ClipInput]]:
    with_gaze = [item for item in inputs if item[0].gaze_category is not None]
    without_gaze = [item for item in inputs if item[0].gaze_category is None]
    return with_gaze, without_gaze


class GazeSortDefinition(_DirectionalDefinition):
    key = "gaze_sort"
    version = 1
    prerequisites = ("gaze",)
    parameters = (_direction_spec(GAZE_DIRECTIONS, "Gaze angle progression"),)

    def generate(self, inputs, parameters, rng):
        with_gaze, without_gaze = _split_gaze(inputs)
        if not with_gaze:
            logger.warning("No clips with gaze data for Gaze Sort")
            return _ordering(inputs, ["No clips have gaze data; original order kept"])
        direction = parameters["direction"]

        def yaw(item: ClipInput) -> float:
            return item[0].gaze_yaw if item[0].gaze_yaw is not None else 0.0

        def pitch(item: ClipInput) -> float:
            return item[0].gaze_pitch if item[0].gaze_pitch is not None else 0.0

        if direction == "left_to_right":
            ordered = sorted(with_gaze, key=yaw)
        elif direction == "right_to_left":
            ordered = sorted(with_gaze, key=lambda item: -yaw(item))
        elif direction == "up_to_down":
            ordered = sorted(with_gaze, key=pitch)
        else:
            ordered = sorted(with_gaze, key=lambda item: -pitch(item))
        notes = [f"{len(without_gaze)} clips lack gaze data (appended at end)"] if without_gaze else []
        return _ordering(ordered + without_gaze, notes)


class GazeConsistencyDefinition(AlgorithmDefinition):
    key = "gaze_consistency"
    version = 1
    prerequisites = ("gaze",)

    def generate(self, inputs, parameters, rng):
        with_gaze, without_gaze = _split_gaze(inputs)
        if not with_gaze:
            logger.warning("No clips with gaze data for Gaze Consistency")
            return _ordering(inputs, ["No clips have gaze data; original order kept"])
        groups: dict[str, list[ClipInput]] = {}
        for item in with_gaze:
            groups.setdefault(item[0].gaze_category, []).append(item)
        result: list[ClipInput] = []
        for category, members in sorted(groups.items(), key=lambda g: -len(g[1])):
            if category in ("looking_left", "looking_right", "at_camera"):
                members.sort(key=lambda item: item[0].gaze_yaw if item[0].gaze_yaw is not None else 0.0)
            else:
                members.sort(key=lambda item: item[0].gaze_pitch if item[0].gaze_pitch is not None else 0.0)
            result.extend(members)
        notes = [f"{len(without_gaze)} clips lack gaze data (appended at end)"] if without_gaze else []
        return _ordering(result + without_gaze, notes)


ARRANGE_DEFINITIONS = (
    SequentialDefinition, DurationDefinition, ShotTypeDefinition, ProximityDefinition,
    BrightnessDefinition, VolumeDefinition, GazeSortDefinition, GazeConsistencyDefinition,
)


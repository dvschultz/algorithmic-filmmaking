"""Algorithmic remix algorithms for video clip sequencing."""

import logging
from copy import deepcopy
from threading import Event
import random
from typing import List, Tuple, Any, Optional, Literal
from core.remix.shuffle import constrained_shuffle
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
    "run_registry_algorithm",
    "assign_random_transforms",
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


def generate_sequence(
    algorithm: str,
    clips: List[Tuple[Any, Any]],  # List of (Clip, Source) tuples
    clip_count: int,
    direction: Optional[str] = None,
    seed: Optional[int] = None,
    no_color_handling: Optional[str] = None,
    *,
    cancel_event: Event | None = None,
) -> List[Tuple[Any, Any]]:
    """
    Generate a sequence of clips using the specified algorithm.

    Args:
        algorithm: Algorithm name ("shuffle", "sequential", "color", "shot_type",
                   "duration", "brightness", "volume", etc.)
        clips: List of (Clip, Source) tuples to sequence
        clip_count: Maximum number of clips to include
        direction: For color: "rainbow", "warm_to_cool", "cool_to_warm", "complementary"
                   For duration: "short_first", "long_first"
        seed: Random seed for shuffle reproducibility (0 = random)
        no_color_handling: For color algorithm — how to handle clips without color data.
                   "append_end" (default): append after sorted clips
                   "exclude": drop clips without color data
                   "sort_inline": treat as hue 0 and sort normally
        cancel_event: Stops pending scalar and embedding batches and discards a
                   cancelled brightness, volume, similarity-chain, or Match Cut result. Other algorithms currently
                   check only before dispatch.

    Returns:
        Ordered list of (Clip, Source) tuples ready for timeline
    """
    if cancel_event is not None and cancel_event.is_set():
        return []
    clips_to_use = clips[:clip_count]

    if algorithm in ("shuffle", "color"):
        run = run_registry_algorithm(
            algorithm, clips_to_use, direction=direction, seed=seed,
            no_color_handling=no_color_handling, cancel_event=cancel_event,
        )
        return [] if run is None else run.ordered_clips

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

    elif algorithm in ("brightness", "volume"):
        from core.remix.scalar_inputs import sort_scalar_inputs

        scalar_kind: Literal["brightness", "volume"]
        if algorithm == "brightness":
            scalar_kind = "brightness"
            clips_to_use = _auto_compute_brightness(clips_to_use, cancel_event=cancel_event)
        else:
            scalar_kind = "volume"
            clips_to_use = _auto_compute_volume(clips_to_use, cancel_event=cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            return []
        return sort_scalar_inputs(clips_to_use, scalar_kind, direction=direction)

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
        clips_to_use = _auto_compute_embeddings(clips_to_use, cancel_event=cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            return []
        return similarity_chain(clips_to_use, start_clip_id=None)

    elif algorithm == "match_cut":
        from core.remix.match_cut import match_cut_chain
        # Auto-compute boundary embeddings
        clips_to_use = _auto_compute_boundary_embeddings(clips_to_use, cancel_event=cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            return []
        return match_cut_chain(clips_to_use, start_clip_id=None)

    elif algorithm == "gaze_sort":
        gaze_direction = direction or "left_to_right"

        # Split clips into with/without gaze data (consistent with gaze_consistency)
        with_gaze = []
        without_gaze = []
        for clip, source in clips_to_use:
            if clip.gaze_category is not None:
                with_gaze.append((clip, source))
            else:
                without_gaze.append((clip, source))

        if not with_gaze:
            logger.warning("No clips with gaze data for Gaze Sort")
            return clips_to_use

        if gaze_direction == "left_to_right":
            sorted_clips = sorted(with_gaze, key=lambda item: item[0].gaze_yaw)
        elif gaze_direction == "right_to_left":
            sorted_clips = sorted(with_gaze, key=lambda item: -item[0].gaze_yaw)
        elif gaze_direction == "up_to_down":
            sorted_clips = sorted(
                with_gaze,
                key=lambda item: item[0].gaze_pitch if item[0].gaze_pitch is not None else 0.0,
            )
        elif gaze_direction == "down_to_up":
            sorted_clips = sorted(
                with_gaze,
                key=lambda item: -(item[0].gaze_pitch if item[0].gaze_pitch is not None else 0.0),
            )
        else:
            raise ValueError(f"Unknown gaze_sort direction: {gaze_direction!r}")

        if without_gaze:
            logger.info("Gaze Sort: %d clips lack gaze data (appended at end)", len(without_gaze))
        return sorted_clips + without_gaze

    elif algorithm == "gaze_consistency":
        # Group clips by gaze_category, largest group first
        with_gaze = []
        without_gaze = []
        for clip, source in clips_to_use:
            if clip.gaze_category is not None:
                with_gaze.append((clip, source))
            else:
                without_gaze.append((clip, source))

        if not with_gaze:
            logger.warning("No clips with gaze data for Gaze Consistency")
            return clips_to_use

        # Group by category
        groups: dict[str, list[tuple]] = {}
        for clip, source in with_gaze:
            cat = clip.gaze_category
            if cat not in groups:
                groups[cat] = []
            groups[cat].append((clip, source))

        # Sort groups by count (largest first)
        sorted_groups = sorted(groups.items(), key=lambda g: -len(g[1]))

        # Within each group, sort by the relevant angle
        result = []
        for category, group_clips in sorted_groups:
            if category in ("looking_left", "looking_right", "at_camera"):
                group_clips.sort(key=lambda item: item[0].gaze_yaw if item[0].gaze_yaw is not None else 0.0)
            else:
                # looking_up, looking_down
                group_clips.sort(key=lambda item: item[0].gaze_pitch if item[0].gaze_pitch is not None else 0.0)
            result.extend(group_clips)

        if without_gaze:
            logger.info("Gaze Consistency: %d clips lack gaze data (appended at end)", len(without_gaze))
        return result + without_gaze

    else:
        if _is_dialog_only_algorithm(algorithm):
            raise NotImplementedError(
                f"Algorithm {algorithm!r} is dialog-only and cannot be generated "
                "with core.remix.generate_sequence(); use its dialog/direct "
                "generator instead."
            )

        # Sequential - use original order
        return clips_to_use


def run_registry_algorithm(
    algorithm: str,
    clips: List[Tuple[Any, Any]],
    *,
    direction: Optional[str] = None,
    seed: Optional[int] = None,
    no_color_handling: Optional[str] = None,
    transform_options: Optional[dict[str, bool]] = None,
    parameters: Optional[dict[str, Any]] = None,
    cancel_event: Event | None = None,
):
    """Run a registry algorithm from legacy keyword arguments.

    Each definition translates the pre-registry conventions (``direction`` /
    ``no_color_handling``, ``transform_options``) through its
    ``legacy_parameters`` hook; explicit ``parameters`` override them. Legacy
    ``seed=0``/``None`` means "draw a fresh seed", which the recipe records.
    Repeated clips (a timeline that placed one clip twice) collapse to their
    first occurrence because recipes describe unique inputs. Returns a
    :class:`core.remix.engine.GenerationRun` or ``None`` when cancelled.
    """
    from core.remix.registry import legacy_seed, registry, run_algorithm

    definition = registry.require(algorithm)
    merged = definition.legacy_parameters(
        direction=direction, no_color_handling=no_color_handling,
        transform_options=transform_options,
    )
    merged.update(parameters or {})
    seen: set[str] = set()
    unique = [pair for pair in clips if not (pair[0].id in seen or seen.add(pair[0].id))]
    return run_algorithm(
        definition, unique, merged,
        seed=legacy_seed(seed) if definition.seeded else None,
        cancel_event=cancel_event,
    )


def assign_random_transforms(
    sequence_clips: list,
    transform_options: dict[str, bool],
    seed: Optional[int] = None,
) -> None:
    """Assign random transforms to SequenceClip objects in-place.

    Each enabled transform has a 50% chance of being applied per clip.

    Args:
        sequence_clips: List of SequenceClip objects to modify
        transform_options: Dict of transform flags, e.g. {"hflip": True, "vflip": False, "reverse": True}
        seed: Optional random seed for deterministic assignment
    """
    rng = random.Random(seed) if seed is not None else random.Random()

    for seq_clip in sequence_clips:
        seq_clip.hflip = bool(transform_options.get("hflip")) and rng.random() < 0.5
        seq_clip.vflip = bool(transform_options.get("vflip")) and rng.random() < 0.5
        seq_clip.reverse = bool(transform_options.get("reverse")) and rng.random() < 0.5


def _is_dialog_only_algorithm(algorithm: str) -> bool:
    """Return whether an unhandled registered algorithm requires a dialog."""
    try:
        from ui.algorithm_config import ALGORITHM_CONFIG
    except Exception as e:
        logger.warning("Could not load algorithm config for fallback check: %s", e)
        return False

    config = ALGORITHM_CONFIG.get(algorithm.lower())
    return bool(config and config.get("is_dialog"))


def _auto_compute_brightness(
    clips: List[Tuple[Any, Any]], *, cancel_event: Event | None = None
) -> List[Tuple[Any, Any]]:
    """Verify brightness on detached sequencing inputs."""
    from core.remix.scalar_inputs import scalar_inputs

    return scalar_inputs(clips, "brightness", cancel_event=cancel_event)


def _auto_compute_volume(
    clips: List[Tuple[Any, Any]], *, cancel_event: Event | None = None
) -> List[Tuple[Any, Any]]:
    """Verify volume, including no-audio results, on detached inputs."""
    from core.remix.scalar_inputs import scalar_inputs

    return scalar_inputs(clips, "volume", cancel_event=cancel_event)


def _auto_compute_embeddings(
    clips: List[Tuple[Any, Any]], *, cancel_event: Event | None = None
) -> List[Tuple[Any, Any]]:
    """Compute shared embedding prerequisites on detached sequencing inputs."""
    from core.remix.embedding_inputs import populate_embeddings

    snapshots = deepcopy(clips)
    populate_embeddings(snapshots, cancel_event=cancel_event)
    return snapshots


def _auto_compute_boundary_embeddings(
    clips: List[Tuple[Any, Any]], *, cancel_event: Event | None = None
) -> List[Tuple[Any, Any]]:
    """Resolve first/last-frame prerequisites on private sequencing snapshots."""
    from core.remix.embedding_inputs import populate_boundary_embeddings

    snapshots = deepcopy(clips)
    populate_boundary_embeddings(snapshots, cancel_event=cancel_event)
    return snapshots

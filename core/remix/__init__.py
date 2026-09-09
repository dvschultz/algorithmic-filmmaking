"""Algorithmic remix algorithms for video clip sequencing."""

import logging
from copy import deepcopy
from threading import Event
import random
from typing import Any, Callable, List, Optional, Tuple
from core.remix.shuffle import constrained_shuffle
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
    resolve_prerequisites: bool = True,
    progress: Optional[Callable[[str], None]] = None,
    resources: Optional[dict[str, Any]] = None,
    explicit_seed: bool = False,
):
    """Run a registry algorithm from legacy keyword arguments.

    Each definition translates the pre-registry conventions (``direction`` /
    ``no_color_handling``, ``transform_options``) through its
    ``legacy_parameters`` hook; explicit ``parameters`` override them. Legacy
    ``seed=0``/``None`` means "draw a fresh seed", which the recipe records.
    ``explicit_seed=True`` keeps the registry contract instead (0 is a seed).
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
        seed=(seed if explicit_seed else legacy_seed(seed)) if definition.seeded else None,
        cancel_event=cancel_event,
        resolve_prerequisites=resolve_prerequisites,
        progress=progress,
        resources=resources,
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

"""Legacy keyword-contract dispatcher kept for algorithm tests only.

``core.remix.generate_sequence`` was the pre-registry entry point
(``direction``/``no_color_handling`` keywords, ``seed=0`` meaning random).
Production code runs the registry directly (``core.remix.run_registry_algorithm``
or ``core.spine.sequences.generate_sequence``); this shim lets the algorithm
suites keep exercising implementations through the old keyword contract.
"""

from __future__ import annotations

from threading import Event
from typing import Any, List, Optional, Tuple

from core.remix import run_registry_algorithm


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
    """Test-only dispatcher over the algorithm registry (legacy keywords).

    Args:
        algorithm: Registry algorithm key (parameter-free or direction-based ones)
        clips: List of (Clip, Source) tuples to sequence
        clip_count: Maximum number of clips to include
        direction: Legacy direction keyword for directional algorithms
        seed: Random seed for shuffle reproducibility (0 = random)
        no_color_handling: Chromatics handling of clips without color data
        cancel_event: Stops prerequisite work and discards a cancelled result

    Returns:
        Ordered list of (Clip, Source) tuples ready for timeline; unknown keys
        keep the original order.

    Raises:
        NotImplementedError: for provider-assisted algorithms, which need
            explicit parameters through the registry.
    """
    if cancel_event is not None and cancel_event.is_set():
        return []
    clips_to_use = clips[:clip_count]

    from core.remix.registry import registry

    if algorithm not in registry:
        return clips_to_use
    if registry.require(algorithm).provider:
        raise NotImplementedError(
            f"Algorithm {algorithm!r} is provider-assisted (dialog-only here) and cannot be "
            "generated with core.remix.generate_sequence(); run it through the registry "
            "with explicit parameters (core.spine.sequences.generate_sequence)."
        )
    run = run_registry_algorithm(
        algorithm, clips_to_use, direction=direction, seed=seed,
        no_color_handling=no_color_handling, cancel_event=cancel_event,
    )
    return [] if run is None else run.ordered_clips

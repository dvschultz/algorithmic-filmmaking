"""Cost and time estimation for sequence analysis operations.

Provides hardcoded pricing constants and estimation logic so users can
see what analysis is needed and its approximate cost before generating
a sequence.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OperationEstimate:
    """Cost and time estimate for a single analysis operation."""

    operation: str  # analysis operation key
    label: str  # display label
    clips_needing: int  # clips that still need this analysis
    clips_total: int  # total clips in selection
    tier: str  # "local" or "cloud"
    time_seconds: float  # estimated wall-clock time
    cost_dollars: float  # estimated dollar cost


# Per-clip time estimates in seconds (conservative, CPU-only)
TIME_PER_CLIP: dict[str, dict[str, float]] = {
    "colors": {"local": 0.3},
    "shots": {"local": 1.0, "cloud": 1.5},
    "extract_text": {"local": 0.5, "cloud": 0.8},
    "describe": {"local": 1.5, "cloud": 0.8},  # Qwen3-VL on Apple Silicon; Moondream ~3.0s
    "brightness": {"local": 0.1},
    "volume": {"local": 0.2},
    "embeddings": {"local": 0.8},
    "boundary_embeddings": {"local": 1.5},
    "transcribe": {"local": 0.4},  # mlx-whisper on Apple Silicon; faster-whisper ~2.0s
    "cinematography": {"cloud": 1.0},
}

# Per-clip dollar costs (cloud tiers only)
COST_PER_CLIP: dict[str, dict[str, float]] = {
    "shots": {"cloud": 0.00026},
    "extract_text": {"cloud": 0.001},
    "describe": {"cloud": 0.001},
    "cinematography": {"cloud": 0.002},
}

# Operations that support local/cloud tier switching
TIERED_OPERATIONS: dict[str, dict[str, str]] = {
    "shots": {"local": "Local (Free)", "cloud": "Cloud (Paid)"},
    "extract_text": {"local": "Local (Free)", "cloud": "Cloud (Paid)"},
    "describe": {"local": "Local (Free)", "cloud": "Cloud (Paid)"},
}

# Display labels for operations
OPERATION_LABELS: dict[str, str] = {
    "colors": "Extract Colors",
    "shots": "Classify Shots",
    "extract_text": "Extract Text",
    "describe": "Describe",
    "brightness": "Brightness",
    "volume": "Volume",
    "embeddings": "Embeddings",
    "boundary_embeddings": "Boundary Embeddings",
    "transcribe": "Transcribe",
    "cinematography": "Rich Analysis",
}

# Map operation key to a function that checks if a clip has that metadata
METADATA_CHECKS: dict[str, callable] = {
    "colors": lambda clip: bool(clip.dominant_colors),
    "shots": lambda clip: bool(clip.shot_type) or bool(clip.cinematography),
    "extract_text": lambda clip: bool(clip.extracted_texts),
    "describe": lambda clip: bool(clip.description),
    "brightness": lambda clip: clip.average_brightness is not None,
    "volume": lambda clip: clip.rms_volume is not None,
    "embeddings": lambda clip: bool(clip.embedding),
    "boundary_embeddings": lambda clip: bool(clip.first_frame_embedding),
    "transcribe": lambda clip: bool(clip.transcript),
    "cinematography": lambda clip: bool(clip.cinematography),
}

# Default parallelism by operation (used when no settings provided)
_DEFAULT_PARALLELISM: dict[str, int] = {
    "colors": 4,
    "shots": 1,
    "extract_text": 1,
    "describe": 3,
    "brightness": 4,
    "volume": 1,
    "embeddings": 1,
    "boundary_embeddings": 1,
    "transcribe": 2,
    "cinematography": 2,
}


def _get_algorithm_config() -> dict:
    """Late import to avoid circular dependency with ui.algorithm_config."""
    from ui.algorithm_config import ALGORITHM_CONFIG
    return ALGORITHM_CONFIG


def _resolve_tier(operation: str, tier_overrides: dict[str, str] | None,
                  settings=None) -> str:
    """Determine which tier to use for an operation.

    Priority: tier_overrides > global settings > default (local if available).
    """
    if tier_overrides and operation in tier_overrides:
        return tier_overrides[operation]

    # Check global settings for known tier fields
    if settings is not None:
        if operation == "describe" and hasattr(settings, "description_model_tier"):
            tier = settings.description_model_tier
            return "cloud" if tier == "cloud" else "local"
        if operation == "shots" and hasattr(settings, "shot_classifier_tier"):
            tier = settings.shot_classifier_tier
            return "cloud" if tier == "cloud" else "local"
        if operation == "extract_text" and hasattr(settings, "text_extraction_method"):
            method = settings.text_extraction_method
            return "cloud" if method == "vlm" else "local"

    # Default: prefer local if available
    time_info = TIME_PER_CLIP.get(operation, {})
    if "local" in time_info:
        return "local"
    return "cloud"


def _get_parallelism(operation: str, tier: str, settings=None) -> int:
    """Get parallelism factor for wall-clock time calculation."""
    if settings is not None:
        if operation == "colors" and hasattr(settings, "color_analysis_parallelism"):
            return max(1, settings.color_analysis_parallelism)
        if operation == "describe" and hasattr(settings, "description_parallelism"):
            return max(1, settings.description_parallelism)
        if operation == "transcribe" and hasattr(settings, "transcription_parallelism"):
            return max(1, settings.transcription_parallelism)
        if operation in ("shots", "embeddings", "boundary_embeddings"):
            if hasattr(settings, "local_model_parallelism"):
                return max(1, settings.local_model_parallelism)
        if operation == "cinematography":
            if hasattr(settings, "cinematography_batch_parallelism"):
                return max(1, settings.cinematography_batch_parallelism)
    return _DEFAULT_PARALLELISM.get(operation, 1)


def estimate_sequence_cost(
    algorithm: str,
    clips: list,
    tier_overrides: dict[str, str] | None = None,
    settings=None,
) -> list[OperationEstimate]:
    """Calculate cost estimates for a sequence algorithm.

    Args:
        algorithm: Algorithm key from ALGORITHM_CONFIG
        clips: List of Clip objects (or any objects with the metadata fields)
        tier_overrides: Per-operation tier overrides {"describe": "cloud"}
        settings: Settings object for tier defaults and parallelism

    Returns:
        List of OperationEstimate, one per required operation.
        Empty list if no analysis needed or all clips are ready.
    """
    algo_config = _get_algorithm_config()
    config = algo_config.get(algorithm.lower())
    if not config:
        return []

    required = config.get("required_analysis", [])
    if not required:
        return []

    total = len(clips)
    if total == 0:
        return []

    estimates = []
    for op_key in required:
        check = METADATA_CHECKS.get(op_key)
        if check is None:
            continue

        needing = sum(1 for clip in clips if not check(clip))
        if needing == 0:
            continue

        tier = _resolve_tier(op_key, tier_overrides, settings)
        time_info = TIME_PER_CLIP.get(op_key, {})
        cost_info = COST_PER_CLIP.get(op_key, {})

        per_clip_time = time_info.get(tier, time_info.get("local", 0.0))
        per_clip_cost = cost_info.get(tier, 0.0)

        parallelism = _get_parallelism(op_key, tier, settings)
        wall_time = (per_clip_time * needing) / parallelism
        total_cost = per_clip_cost * needing

        estimates.append(OperationEstimate(
            operation=op_key,
            label=OPERATION_LABELS.get(op_key, op_key.replace("_", " ").title()),
            clips_needing=needing,
            clips_total=total,
            tier=tier,
            time_seconds=wall_time,
            cost_dollars=total_cost,
        ))

    return estimates


def estimate_intention_cost(
    algorithm: str,
    clip_count: int,
    tier_overrides: dict[str, str] | None = None,
    settings=None,
) -> list[OperationEstimate]:
    """Calculate cost estimates for the intention flow (no clips yet).

    Assumes all clips will need analysis since clips don't exist yet.

    Args:
        algorithm: Algorithm key from ALGORITHM_CONFIG
        clip_count: Expected number of clips
        tier_overrides: Per-operation tier overrides
        settings: Settings object for tier defaults and parallelism

    Returns:
        List of OperationEstimate, one per required operation.
    """
    algo_config = _get_algorithm_config()
    config = algo_config.get(algorithm.lower())
    if not config:
        return []

    required = config.get("required_analysis", [])
    if not required or clip_count == 0:
        return []

    estimates = []
    for op_key in required:
        if op_key not in METADATA_CHECKS:
            continue

        tier = _resolve_tier(op_key, tier_overrides, settings)
        time_info = TIME_PER_CLIP.get(op_key, {})
        cost_info = COST_PER_CLIP.get(op_key, {})

        per_clip_time = time_info.get(tier, time_info.get("local", 0.0))
        per_clip_cost = cost_info.get(tier, 0.0)

        parallelism = _get_parallelism(op_key, tier, settings)
        wall_time = (per_clip_time * clip_count) / parallelism
        total_cost = per_clip_cost * clip_count

        estimates.append(OperationEstimate(
            operation=op_key,
            label=OPERATION_LABELS.get(op_key, op_key.replace("_", " ").title()),
            clips_needing=clip_count,
            clips_total=clip_count,
            tier=tier,
            time_seconds=wall_time,
            cost_dollars=total_cost,
        ))

    return estimates

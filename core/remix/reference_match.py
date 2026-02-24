"""Reference-guided matching algorithm for clip sequencing.

Matches user clips to reference video clips across weighted multi-dimensional
distance. Each reference clip position finds its best-matching user clip based
on artist-selected dimensions and weights.
"""

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from models.clip import Clip, Source

logger = logging.getLogger(__name__)

# Dimensions that use cosine distance on embedding vectors
_EMBEDDING_DIMENSIONS = {"embedding"}

# Dimensions that use exact-match categorical distance
_CATEGORICAL_DIMENSIONS = {"movement"}

# Mapping from dimension key to the analysis fields checked
DIMENSION_ANALYSIS_REQUIREMENTS = {
    "color": ["colors"],
    "brightness": ["brightness"],
    "shot_scale": ["shots"],
    "audio": ["volume"],
    "embedding": ["embeddings"],
    "movement": ["cinematography"],
    "duration": [],  # Always available from clip frames
}

# Shot size proximity scores (10-class cinematography)
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

# Fallback: 5-class shot_type
_SHOT_TYPE_PROXIMITY = {
    "wide shot": 2.0,
    "full shot": 4.0,
    "medium shot": 5.0,
    "close-up": 7.0,
    "extreme close-up": 9.0,
}


def _get_proximity_score(clip: "Clip") -> float:
    """Get numeric proximity score for a clip's shot scale.

    Prefers 10-class cinematography.shot_size, falls back to 5-class shot_type.
    """
    if clip.cinematography and clip.cinematography.shot_size:
        return _SHOT_SIZE_PROXIMITY.get(clip.cinematography.shot_size, 5.0)
    if clip.shot_type:
        return _SHOT_TYPE_PROXIMITY.get(clip.shot_type, 5.0)
    return 5.0


def extract_feature_vector(
    clip: "Clip",
    source: "Source",
    active_dimensions: list[str],
) -> dict:
    """Extract feature values for active dimensions from a clip.

    Returns a dict mapping dimension key to its raw value. Values are NOT
    normalized here — normalization happens in compute_normalizers().

    Args:
        clip: Clip object with analysis metadata
        source: Source object with fps
        active_dimensions: List of dimension keys to extract
    """
    from core.analysis.color import get_primary_hue

    vector: dict = {}

    if "color" in active_dimensions and clip.dominant_colors:
        vector["color"] = get_primary_hue(clip.dominant_colors) / 360.0

    if "brightness" in active_dimensions and clip.average_brightness is not None:
        vector["brightness"] = clip.average_brightness

    if "shot_scale" in active_dimensions:
        vector["shot_scale"] = _get_proximity_score(clip) / 10.0

    if "audio" in active_dimensions and clip.rms_volume is not None:
        vector["audio"] = clip.rms_volume

    if "embedding" in active_dimensions and clip.embedding:
        vector["embedding"] = clip.embedding

    if "movement" in active_dimensions:
        movement = None
        if clip.cinematography and clip.cinematography.camera_movement:
            movement = clip.cinematography.camera_movement
        if movement:
            vector["movement"] = movement

    if "duration" in active_dimensions:
        vector["duration"] = clip.duration_seconds(source.fps)

    return vector


def compute_normalizers(
    all_vectors: list[dict],
    active_dimensions: list[str],
) -> dict[str, tuple[float, float]]:
    """Compute min-max normalizers for scalar dimensions.

    Embedding and categorical dimensions are excluded (they use their own
    distance metrics).

    Args:
        all_vectors: Feature vectors from both reference and user clips
        active_dimensions: Active dimension keys

    Returns:
        Dict mapping dimension key to (min_val, max_val) tuple
    """
    normalizers: dict[str, tuple[float, float]] = {}

    for dim in active_dimensions:
        if dim in _EMBEDDING_DIMENSIONS or dim in _CATEGORICAL_DIMENSIONS:
            continue

        values = [v[dim] for v in all_vectors if dim in v and isinstance(v[dim], (int, float))]
        if not values:
            normalizers[dim] = (0.0, 1.0)
            continue

        lo, hi = min(values), max(values)
        normalizers[dim] = (lo, hi)

    return normalizers


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Compute cosine distance between two vectors, normalized to [0, 1]."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)

    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 1.0

    similarity = np.dot(a_arr, b_arr) / (norm_a * norm_b)
    similarity = float(np.clip(similarity, -1.0, 1.0))
    # Cosine distance: 0 (identical) to 2 (opposite), normalize to [0, 1]
    return (1.0 - similarity) / 2.0


def _batch_cosine_distances(
    ref_embeddings: list[list[float]],
    user_embeddings: list[list[float]],
) -> Optional[np.ndarray]:
    """Pre-compute R×U cosine distance matrix for embedding vectors.

    Returns an (R, U) ndarray of distances in [0, 1], or None if either list
    is empty. Vectors with zero norm get distance 1.0.
    """
    if not ref_embeddings or not user_embeddings:
        return None

    R = np.array(ref_embeddings, dtype=np.float32)  # (R, D)
    U = np.array(user_embeddings, dtype=np.float32)  # (U, D)

    r_norms = np.linalg.norm(R, axis=1, keepdims=True)  # (R, 1)
    u_norms = np.linalg.norm(U, axis=1, keepdims=True)  # (U, 1)

    # Avoid division by zero: replace zero norms with 1 (result overridden below)
    r_safe = np.where(r_norms == 0, 1.0, r_norms)
    u_safe = np.where(u_norms == 0, 1.0, u_norms)

    similarity = (R / r_safe) @ (U / u_safe).T  # (R, U)
    np.clip(similarity, -1.0, 1.0, out=similarity)
    distances = (1.0 - similarity) / 2.0

    # Set distance to 1.0 where either vector had zero norm
    zero_ref = (r_norms.squeeze(1) == 0)  # (R,)
    zero_user = (u_norms.squeeze(1) == 0)  # (U,)
    distances[zero_ref, :] = 1.0
    distances[:, zero_user] = 1.0

    return distances


def weighted_distance(
    ref_vector: dict,
    user_vector: dict,
    weights: dict[str, float],
    normalizers: dict[str, tuple[float, float]],
    embedding_dist: Optional[float] = None,
) -> float:
    """Compute weighted multi-dimensional distance between two feature vectors.

    Args:
        ref_vector: Feature vector for reference clip
        user_vector: Feature vector for user clip
        weights: Dimension -> weight (0.0 to 1.0)
        normalizers: Min-max bounds for scalar dimensions
        embedding_dist: Pre-computed cosine distance for embedding dimension.
            If provided, skips per-pair array allocation.

    Returns:
        Weighted average distance (0 = perfect match, higher = worse)
    """
    total = 0.0
    total_weight = 0.0

    for dim, weight in weights.items():
        if weight <= 0 or dim not in ref_vector or dim not in user_vector:
            continue

        if dim in _EMBEDDING_DIMENSIONS:
            dist = embedding_dist if embedding_dist is not None else _cosine_distance(ref_vector[dim], user_vector[dim])
        elif dim in _CATEGORICAL_DIMENSIONS:
            dist = 0.0 if ref_vector[dim] == user_vector[dim] else 1.0
        else:
            # Scalar: min-max normalize then absolute difference
            lo, hi = normalizers.get(dim, (0.0, 1.0))
            if hi > lo:
                r = (ref_vector[dim] - lo) / (hi - lo)
                u = (user_vector[dim] - lo) / (hi - lo)
            else:
                r = 0.5
                u = 0.5
            dist = abs(r - u)

        total += weight * dist
        total_weight += weight

    return total / total_weight if total_weight > 0 else float("inf")


def reference_guided_match(
    reference_clips: list[tuple["Clip", "Source"]],
    user_clips: list[tuple["Clip", "Source"]],
    weights: dict[str, float],
    allow_repeats: bool = False,
) -> list[tuple["Clip", "Source"]]:
    """Match user clips to reference clip positions via weighted distance.

    Greedy sequential matching: for each reference clip (in order), find the
    best-matching user clip. With allow_repeats=False, each user clip is used
    at most once. Unmatched reference positions produce no output clip.

    Args:
        reference_clips: List of (Clip, Source) tuples from reference video
        user_clips: List of (Clip, Source) tuples from user's footage
        weights: Dimension key -> weight (0.0 to 1.0). Only dimensions with
            weight > 0 participate in matching.
        allow_repeats: If True, same user clip can match multiple positions.

    Returns:
        List of (Clip, Source) tuples in reference order. May be shorter than
        reference_clips if user clips run out (allow_repeats=False).
    """
    if not reference_clips or not user_clips:
        return []

    # Filter to active dimensions (weight > 0)
    active_dimensions = [dim for dim, w in weights.items() if w > 0]
    if not active_dimensions:
        logger.warning("All dimension weights are zero, cannot match")
        return []

    active_weights = {dim: weights[dim] for dim in active_dimensions}

    # Extract feature vectors
    ref_vectors = []
    for clip, source in reference_clips:
        ref_vectors.append(extract_feature_vector(clip, source, active_dimensions))

    user_vectors = []
    for clip, source in user_clips:
        user_vectors.append(extract_feature_vector(clip, source, active_dimensions))

    # Compute normalizers from union of all vectors
    all_vectors = ref_vectors + user_vectors
    normalizers = compute_normalizers(all_vectors, active_dimensions)

    # Pre-compute embedding distance matrix (R×U) to avoid per-pair allocation
    embedding_dists = None
    if "embedding" in active_weights:
        ref_embs = [v.get("embedding") for v in ref_vectors]
        user_embs = [v.get("embedding") for v in user_vectors]
        # Only batch where both sides have embeddings
        if all(e is not None for e in ref_embs) and all(e is not None for e in user_embs):
            embedding_dists = _batch_cosine_distances(ref_embs, user_embs)

    # Greedy matching
    used_indices: set[int] = set()
    result: list[tuple["Clip", "Source"]] = []

    for ref_idx, ref_vec in enumerate(ref_vectors):
        best_distance = float("inf")
        best_user_idx: Optional[int] = None

        for user_idx, user_vec in enumerate(user_vectors):
            if not allow_repeats and user_idx in used_indices:
                continue

            dist = weighted_distance(
                ref_vec, user_vec, active_weights, normalizers,
                embedding_dist=float(embedding_dists[ref_idx, user_idx])
                if embedding_dists is not None else None,
            )
            if dist < best_distance:
                best_distance = dist
                best_user_idx = user_idx

        if best_user_idx is not None:
            used_indices.add(best_user_idx)
            clip, source = user_clips[best_user_idx]
            result.append((clip, source))
            logger.debug(
                f"Ref clip {ref_idx}: matched user clip {best_user_idx} "
                f"(distance={best_distance:.4f})"
            )
        else:
            logger.info(f"Ref clip {ref_idx}: no match available (all user clips used)")

    unmatched = len(reference_clips) - len(result)
    if unmatched > 0:
        logger.info(
            f"Reference-guided match: {len(result)}/{len(reference_clips)} "
            f"positions filled ({unmatched} unmatched)"
        )

    return result


def get_active_dimensions_for_clips(
    clips: list["Clip"],
) -> list[str]:
    """Determine which dimensions have data available across a set of clips.

    Returns dimension keys where at least some clips have the required data.
    Used by the dialog to enable/disable dimension sliders.
    """
    from core.cost_estimates import METADATA_CHECKS

    dimension_to_checks = {
        "color": "colors",
        "brightness": "brightness",
        "shot_scale": "shots",
        "audio": "volume",
        "embedding": "embeddings",
        "movement": "cinematography",
        "duration": None,  # Always available
    }

    available = []
    for dim, check_key in dimension_to_checks.items():
        if check_key is None:
            available.append(dim)
            continue
        check_fn = METADATA_CHECKS.get(check_key)
        if check_fn and any(check_fn(clip) for clip in clips):
            available.append(dim)

    return available

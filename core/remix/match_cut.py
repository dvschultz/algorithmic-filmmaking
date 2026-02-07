"""Match cut sequencing algorithm.

Finds clips whose ending frames match the starting frames of other clips,
creating the illusion of continuous movement across cuts from different films.
Uses boundary frame CLIP embeddings (last frame → first frame similarity).
"""

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def match_cut_chain(
    clips: List[Tuple[Any, Any]],
    start_clip_id: Optional[str] = None,
    refine_iterations: int = 1000,
) -> List[Tuple[Any, Any]]:
    """Order clips for smooth match cuts using boundary frame similarity.

    Greedy nearest-neighbor chain comparing last_frame_embedding of current
    clip to first_frame_embedding of candidates. Optionally applies 2-opt
    refinement to improve chain quality.

    Args:
        clips: List of (Clip, Source) tuples. Clips must have
            first_frame_embedding and last_frame_embedding set.
        start_clip_id: ID of the clip to start from. If None, uses first clip.
        refine_iterations: Maximum 2-opt improvement iterations (0 to disable).

    Returns:
        Reordered list of (Clip, Source) tuples optimized for match cuts.
        Clips without boundary embeddings are appended at the end.
    """
    if len(clips) <= 1:
        return list(clips)

    # Separate clips with and without boundary embeddings
    with_emb = []
    without_emb = []
    for clip, source in clips:
        if clip.first_frame_embedding is not None and clip.last_frame_embedding is not None:
            with_emb.append((clip, source))
        else:
            without_emb.append((clip, source))

    if not with_emb:
        logger.warning("No clips have boundary embeddings for match cut")
        return list(clips)

    n = len(with_emb)
    if n == 1:
        return with_emb + without_emb

    dim = len(with_emb[0][0].first_frame_embedding)

    # Build embedding matrices
    first_embeddings = np.zeros((n, dim), dtype=np.float32)
    last_embeddings = np.zeros((n, dim), dtype=np.float32)
    for idx, (clip, _) in enumerate(with_emb):
        first_embeddings[idx] = clip.first_frame_embedding
        last_embeddings[idx] = clip.last_frame_embedding

    # Transition cost: cosine distance from last frame of i to first frame of j
    # cost[i][j] = 1 - cosine_sim(last_embedding[i], first_embedding[j])
    similarity = last_embeddings @ first_embeddings.T
    np.clip(similarity, -1.0, 1.0, out=similarity)
    cost_matrix = 1.0 - similarity

    # Determine start index
    start_idx = 0
    if start_clip_id:
        for idx, (clip, _) in enumerate(with_emb):
            if clip.id == start_clip_id:
                start_idx = idx
                break

    # Greedy nearest-neighbor chain
    from core.remix._traversal import greedy_nearest_neighbor

    chain = greedy_nearest_neighbor(cost_matrix, start_idx)

    # 2-opt refinement: try swapping pairs to reduce total chain cost
    if refine_iterations > 0 and n > 3:
        chain = _two_opt_refine(chain, cost_matrix, refine_iterations)

    # Build result
    result = [(with_emb[idx][0], with_emb[idx][1]) for idx in chain]
    result.extend(without_emb)

    return result


def _two_opt_refine(
    chain: list[int],
    cost_matrix: np.ndarray,
    max_iterations: int,
) -> list[int]:
    """Apply 2-opt local search to improve chain quality.

    Uses O(1) delta evaluation per candidate swap via prefix sums over
    forward and reverse edge costs, giving O(N^2) per iteration instead
    of the naive O(N^3).

    Args:
        chain: Initial chain (list of indices)
        cost_matrix: NxN asymmetric transition cost matrix
        max_iterations: Maximum number of improvement iterations

    Returns:
        Improved chain (possibly the same if no improvement found)
    """
    n = len(chain)
    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False

        # Precompute edge costs and prefix sums for O(1) delta checks.
        # fwd_edge[k] = cost of chain[k] → chain[k+1]
        # rev_edge[k] = cost of chain[k+1] → chain[k] (used when segment is reversed)
        fwd_edge = np.array(
            [cost_matrix[chain[k], chain[k + 1]] for k in range(n - 1)],
            dtype=np.float64,
        )
        rev_edge = np.array(
            [cost_matrix[chain[k + 1], chain[k]] for k in range(n - 1)],
            dtype=np.float64,
        )
        # prefix_fwd[k] = sum of fwd_edge[0..k-1], prefix_fwd[0] = 0
        prefix_fwd = np.empty(n, dtype=np.float64)
        prefix_fwd[0] = 0.0
        np.cumsum(fwd_edge, out=prefix_fwd[1:])
        prefix_rev = np.empty(n, dtype=np.float64)
        prefix_rev[0] = 0.0
        np.cumsum(rev_edge, out=prefix_rev[1:])

        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Delta from reversing segment chain[i..j]:
                # 1. Boundary edge before segment changes
                delta = (
                    cost_matrix[chain[i - 1], chain[j]]
                    - cost_matrix[chain[i - 1], chain[i]]
                )
                # 2. Boundary edge after segment changes (if not at end)
                if j < n - 1:
                    delta += (
                        cost_matrix[chain[i], chain[j + 1]]
                        - cost_matrix[chain[j], chain[j + 1]]
                    )
                # 3. Internal edges reverse direction
                delta += (prefix_rev[j] - prefix_rev[i]) - (prefix_fwd[j] - prefix_fwd[i])

                if delta < -1e-6:
                    chain[i:j + 1] = chain[i:j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
        iterations += 1

    return chain

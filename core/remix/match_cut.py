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
    visited = {start_idx}
    chain = [start_idx]

    for _ in range(n - 1):
        current = chain[-1]
        costs = cost_matrix[current].copy()

        # Set visited to infinity
        for v in visited:
            costs[v] = float("inf")

        next_idx = int(np.argmin(costs))
        chain.append(next_idx)
        visited.add(next_idx)

    # 2-opt refinement: try swapping pairs to reduce total chain cost
    if refine_iterations > 0 and n > 3:
        chain = _two_opt_refine(chain, cost_matrix, refine_iterations)

    # Build result
    result = [(with_emb[idx][0], with_emb[idx][1]) for idx in chain]
    result.extend(without_emb)

    return result


def _chain_cost(chain: list[int], cost_matrix: np.ndarray) -> float:
    """Compute total transition cost of a chain."""
    total = 0.0
    for i in range(len(chain) - 1):
        total += cost_matrix[chain[i]][chain[i + 1]]
    return total


def _two_opt_refine(
    chain: list[int],
    cost_matrix: np.ndarray,
    max_iterations: int,
) -> list[int]:
    """Apply 2-opt local search to improve chain quality.

    Tries reversing sub-sequences to find lower-cost orderings.

    Args:
        chain: Initial chain (list of indices)
        cost_matrix: NxN transition cost matrix
        max_iterations: Maximum number of improvement iterations

    Returns:
        Improved chain (possibly the same if no improvement found)
    """
    best_cost = _chain_cost(chain, cost_matrix)
    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        for i in range(1, len(chain) - 1):
            for j in range(i + 1, len(chain)):
                # Try reversing the segment between i and j
                new_chain = chain[:i] + chain[i:j + 1][::-1] + chain[j + 1:]
                new_cost = _chain_cost(new_chain, cost_matrix)

                if new_cost < best_cost - 1e-6:
                    chain = new_chain
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
        iterations += 1

    return chain

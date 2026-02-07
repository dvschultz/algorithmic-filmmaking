"""Visual similarity chain algorithm for clip sequencing.

Greedy nearest-neighbor chain: starting from one clip, find the most
visually similar unvisited clip, then repeat. Produces sequences where
every cut feels intentional because adjacent clips share visual qualities.
"""

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix.

    Args:
        embeddings: NxD matrix of L2-normalized embedding vectors

    Returns:
        NxN distance matrix where dist[i][j] = 1 - cos_sim(i, j)
    """
    # Embeddings are already L2-normalized, so dot product = cosine similarity
    similarity = embeddings @ embeddings.T
    # Clip to valid range (floating point can cause >1.0)
    np.clip(similarity, -1.0, 1.0, out=similarity)
    return 1.0 - similarity


def similarity_chain(
    clips: List[Tuple[Any, Any]],
    start_clip_id: Optional[str] = None,
) -> List[Tuple[Any, Any]]:
    """Order clips by greedy nearest-neighbor visual similarity chain.

    Starting from a seed clip, repeatedly pick the most visually similar
    unvisited clip. Uses CLIP embeddings stored on clip objects.

    Args:
        clips: List of (Clip, Source) tuples. Clips must have .embedding set.
        start_clip_id: ID of the clip to start the chain from.
            If None, uses the first clip.

    Returns:
        Reordered list of (Clip, Source) tuples forming the similarity chain.
        Clips without embeddings are appended at the end.
    """
    if len(clips) <= 1:
        return list(clips)

    # Separate clips with and without embeddings
    with_emb = [(i, clip, source) for i, (clip, source) in enumerate(clips)
                if clip.embedding is not None]
    without_emb = [(clip, source) for clip, source in clips
                   if clip.embedding is None]

    if not with_emb:
        logger.warning("No clips have embeddings for similarity chain")
        return list(clips)

    # Build embedding matrix
    n = len(with_emb)
    dim = len(with_emb[0][1].embedding)
    embeddings = np.zeros((n, dim), dtype=np.float32)
    for idx, (_, clip, _) in enumerate(with_emb):
        embeddings[idx] = clip.embedding

    # Compute distance matrix
    dist_matrix = _cosine_distance_matrix(embeddings)

    # Determine start index
    start_idx = 0
    if start_clip_id:
        for idx, (_, clip, _) in enumerate(with_emb):
            if clip.id == start_clip_id:
                start_idx = idx
                break

    # Greedy nearest-neighbor chain
    from core.remix._traversal import greedy_nearest_neighbor

    chain = greedy_nearest_neighbor(dist_matrix, start_idx)

    # Build result
    result = [(with_emb[idx][1], with_emb[idx][2]) for idx in chain]

    # Append clips without embeddings at the end
    result.extend(without_emb)

    return result

"""Shared traversal utilities for chain-based sequencing algorithms."""

import numpy as np


def greedy_nearest_neighbor(
    cost_matrix: np.ndarray,
    start_idx: int = 0,
) -> list[int]:
    """Build a chain by greedily visiting the nearest unvisited node.

    Args:
        cost_matrix: N×N matrix where cost_matrix[i][j] is the cost of
            transitioning from node i to node j.
        start_idx: Index of the first node in the chain.

    Returns:
        List of indices representing the greedy traversal order.
    """
    n = cost_matrix.shape[0]
    visited = {start_idx}
    chain = [start_idx]

    for _ in range(n - 1):
        current = chain[-1]
        costs = cost_matrix[current].copy()

        for v in visited:
            costs[v] = float("inf")

        next_idx = int(np.argmin(costs))
        chain.append(next_idx)
        visited.add(next_idx)

    return chain

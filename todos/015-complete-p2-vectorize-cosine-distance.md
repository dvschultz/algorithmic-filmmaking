---
status: complete
priority: p2
issue_id: "015"
tags: [code-review, performance, algorithm]
dependencies: []
---

# Vectorize Cosine Distance Computation in reference_match.py

## Problem Statement

`_cosine_distance()` in `core/remix/reference_match.py` allocates two new numpy arrays per call by converting Python lists to `np.array`. With 768-dimensional CLIP embeddings and R reference clips * U user clips pairs, this creates massive garbage collection pressure at scale. For example, 200 reference clips and 500 user clips means 100,000 calls to `_cosine_distance()`, each allocating two 768-element float32 arrays (6 KB per call = ~600 MB of short-lived allocations).

The codebase already has the correct vectorized pattern in `core/remix/similarity_chain.py` and `core/remix/match_cut.py`, where embeddings are pre-converted to numpy matrices and batch dot products are used.

## Findings

**Location:** `core/remix/reference_match.py` lines 148-162

```python
def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Compute cosine distance between two vectors, normalized to [0, 1]."""
    a_arr = np.array(a, dtype=np.float32)  # Allocation per call
    b_arr = np.array(b, dtype=np.float32)  # Allocation per call

    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 1.0

    similarity = np.dot(a_arr, b_arr) / (norm_a * norm_b)
    similarity = float(np.clip(similarity, -1.0, 1.0))
    return (1.0 - similarity) / 2.0
```

**Called from:** `weighted_distance()` line 190, which is called inside the nested matching loop at lines 264-272:

```python
for ref_idx, ref_vec in enumerate(ref_vectors):       # R iterations
    for user_idx, user_vec in enumerate(user_vectors): # U iterations
        dist = weighted_distance(...)                   # Calls _cosine_distance()
```

**Existing vectorized pattern:** `core/remix/similarity_chain.py` lines 16-28

```python
def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix."""
    # Embeddings are already L2-normalized, so dot product = cosine similarity
    similarity = embeddings @ embeddings.T  # Single matrix multiply
    np.clip(similarity, -1.0, 1.0, out=similarity)
    return 1.0 - similarity
```

**Same pattern in:** `core/remix/match_cut.py` lines 66-70

```python
similarity = last_embeddings @ first_embeddings.T  # Vectorized
np.clip(similarity, -1.0, 1.0, out=similarity)
cost_matrix = 1.0 - similarity
```

## Proposed Solutions

### Option A: Pre-build Embedding Matrices and Vectorized Distance (Recommended)
**Pros:** Eliminates all per-call allocations, 10-100x faster for large clip sets, follows existing codebase pattern
**Cons:** Slightly more complex matching loop, requires splitting embedding distance from the weighted_distance function
**Effort:** Medium
**Risk:** Low (well-tested pattern already in codebase)

```python
def _precompute_embedding_distances(
    ref_vectors: list[dict],
    user_vectors: list[dict],
) -> np.ndarray | None:
    """Pre-compute R x U embedding cosine distance matrix.

    Returns None if no embeddings are available.
    """
    ref_embeddings = []
    user_embeddings = []
    for v in ref_vectors:
        if "embedding" in v:
            ref_embeddings.append(v["embedding"])
    for v in user_vectors:
        if "embedding" in v:
            user_embeddings.append(v["embedding"])

    if not ref_embeddings or not user_embeddings:
        return None

    ref_matrix = np.array(ref_embeddings, dtype=np.float32)  # R x D
    user_matrix = np.array(user_embeddings, dtype=np.float32)  # U x D

    # L2 normalize
    ref_norms = np.linalg.norm(ref_matrix, axis=1, keepdims=True)
    user_norms = np.linalg.norm(user_matrix, axis=1, keepdims=True)
    ref_norms[ref_norms == 0] = 1.0
    user_norms[user_norms == 0] = 1.0
    ref_matrix /= ref_norms
    user_matrix /= user_norms

    # Single matrix multiply: R x U similarity matrix
    similarity = ref_matrix @ user_matrix.T
    np.clip(similarity, -1.0, 1.0, out=similarity)
    return (1.0 - similarity) / 2.0  # R x U distance matrix
```

Then in `reference_guided_match()`:
```python
embedding_distances = _precompute_embedding_distances(ref_vectors, user_vectors)
# In inner loop: look up embedding_distances[ref_idx, user_idx] instead of calling _cosine_distance()
```

### Option B: Cache numpy Arrays in Feature Vectors
**Pros:** Simpler change - store embeddings as numpy arrays in extract_feature_vector
**Cons:** Still O(R*U) individual dot products, just avoids the list->array conversion
**Effort:** Small
**Risk:** None

### Option C: Use scipy.spatial.distance.cdist
**Pros:** One-liner, battle-tested implementation
**Cons:** Adds scipy dependency (heavy), overkill for this single use
**Effort:** Small
**Risk:** Low but adds dependency

## Recommended Action

Option A - Pre-compute the full R x U embedding distance matrix with a single matrix multiply, matching the vectorized pattern already used in `similarity_chain.py` and `match_cut.py`.

## Technical Details

**Affected Files:**
- `core/remix/reference_match.py` - Add `_precompute_embedding_distances()`, modify `reference_guided_match()` to use it, potentially simplify or remove `_cosine_distance()`

**Performance Impact (estimated):**
- 200 ref clips x 500 user clips x 768-dim embeddings:
  - Current: ~100,000 `np.array()` allocations + individual dot products = ~2-5 seconds
  - After: One matrix multiply (200 x 768) @ (768 x 500) = ~5-20 ms

**Verification:**
1. Run `pytest tests/test_reference_match.py` - all tests pass with identical results
2. Add a benchmark test with 200 x 500 clips to verify speedup
3. Verify matching results are numerically identical (within float32 tolerance)

## Acceptance Criteria

- [ ] Embedding distances computed via single matrix multiply, not per-pair `np.array()` calls
- [ ] `_cosine_distance()` either removed or only used as fallback for edge cases
- [ ] Matching results numerically identical to current implementation
- [ ] All existing tests pass without modification
- [ ] Performance improvement measurable for clip sets > 50 (no regression for small sets)

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-24 | Created from PR #58 code review | Per-call numpy array allocation from Python lists is a common performance anti-pattern; pre-build matrices and use vectorized operations (pattern already in similarity_chain.py and match_cut.py) |

## Resources

- PR #58: Reference-Guided Remixing
- `core/remix/similarity_chain.py` lines 16-28: Vectorized cosine distance matrix pattern
- `core/remix/match_cut.py` lines 66-70: Same vectorized pattern
- NumPy broadcasting docs: https://numpy.org/doc/stable/user/basics.broadcasting.html

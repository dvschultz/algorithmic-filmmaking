---
status: complete
priority: p2
issue_id: "062"
tags: [code-review, performance, rose-hobart]
dependencies: []
---

# Project JSON Bloat from Face Embeddings Serialization

## Problem Statement

Each face embedding is 512 floats serialized as JSON (~5-6 KB per face entry). With typical sampling (1 frame/sec on a 5-sec clip, 1-2 faces per frame), each clip adds 25-60 KB. At scale: 500 clips with 5 faces each = ~15 MB added to project JSON. This causes multi-second save/load times.

## Findings

**Performance Oracle (Critical)**: Projects could easily exceed 50 MB with face embeddings. Scale projections show 500 clips = 15-60 MB depending on face density.

## Proposed Solutions

### Option A: Truncate Float Precision (Quick Win)

Round embedding values to 4-5 decimal places in `to_dict()`. Cuts JSON size roughly in half with negligible impact on cosine similarity accuracy.

```python
"embedding": [round(v, 5) for v in entry["embedding"]]
```

**Pros:** Simple, significant size reduction, no schema change
**Cons:** Slight precision loss (negligible for similarity)
**Effort:** Small
**Risk:** Low

### Option B: Sidecar Binary File (Long-term)

Store face embeddings in a numpy `.npz` or SQLite sidecar file alongside the project JSON.

**Pros:** Optimal size, fast load
**Cons:** Schema change, migration needed, more complex I/O
**Effort:** Large
**Risk:** Medium

## Technical Details

- **File:** `models/clip.py` lines 323-324 (`to_dict`)

## Acceptance Criteria

- [ ] Project file size growth is reasonable (< 5 MB for 500 clips)
- [ ] Save/load times remain under 2 seconds

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Performance Oracle | 512 floats as JSON is ~5KB per entry — truncate precision or use binary format |

---
status: complete
priority: p2
issue_id: "066"
tags: [code-review, performance, rose-hobart]
dependencies: []
---

# VideoCapture Opened Per-Clip Instead of Per-Source

## Problem Statement

`FaceDetectionWorker` iterates clips without grouping by source, opening a new `cv2.VideoCapture` for every clip. For 500 clips from 5 source videos, that's 500 container opens (50-200ms each) instead of 5. Total overhead: ~50 seconds wasted on container parsing.

## Findings

**Performance Oracle (P2)**: Grouping clips by `source_id`, opening VideoCapture once per source, and processing all clips for that source sequentially saves significant time.

## Proposed Solutions

### Option A: Group by Source in Worker (Recommended)

```python
from itertools import groupby
sorted_clips = sorted(clips, key=lambda c: c.source_id)
for source_id, group in groupby(sorted_clips, key=lambda c: c.source_id):
    source = sources_by_id[source_id]
    cap = cv2.VideoCapture(str(source.file_path))
    for clip in group:
        # process clip using existing cap
    cap.release()
```

**Pros:** ~50 sec saved on large projects, enables forward seeking
**Cons:** Changes iteration order (acceptable for analysis)
**Effort:** Medium
**Risk:** Low

## Technical Details

- **Files:** `ui/workers/face_detection_worker.py`, `core/analysis/faces.py` lines 180-204

## Acceptance Criteria

- [ ] VideoCapture opened once per source video
- [ ] All clips from same source processed before moving to next
- [ ] Progress reporting still works per-clip

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Performance Oracle | Group clips by source to avoid repeated container open/close |

---
status: complete
priority: p3
issue_id: "068"
tags: [code-review, quality, rose-hobart]
dependencies: []
---

# Duplicated Bbox Conversion in faces.py

## Problem Statement

The bbox `[x1,y1,x2,y2]` to `[x,y,w,h]` conversion plus dict construction is duplicated between `extract_faces_from_image` (lines 126-133) and `extract_faces_from_clip` (lines 194-202) — 7 lines repeated verbatim.

## Findings

**Python Reviewer + Code Simplicity Reviewer**: Extract a `_format_face(face, frame_number=None)` helper.

## Proposed Solutions

### Option A: Extract Helper (Recommended)

```python
def _format_face(face, frame_number: int | None = None) -> dict:
    x, y, x2, y2 = face.bbox.astype(int).tolist()
    result = {
        "bbox": [x, y, x2 - x, y2 - y],
        "embedding": face.embedding.tolist(),
        "confidence": float(face.det_score),
    }
    if frame_number is not None:
        result["frame_number"] = frame_number
    return result
```

**Effort:** Small | **Risk:** Low

## Technical Details

- **File:** `core/analysis/faces.py` lines 126-133 and 194-202

## Acceptance Criteria

- [ ] Single `_format_face` helper used by both functions
- [ ] Output format unchanged

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Python Reviewer | Extract helpers for duplicated data transformation logic |

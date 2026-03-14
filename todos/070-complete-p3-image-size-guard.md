---
status: complete
priority: p3
issue_id: "070"
tags: [code-review, security, rose-hobart]
dependencies: []
---

# No File Size Guard on Image Processing

## Problem Statement

Neither `extract_faces_from_image` nor the dialog checks image dimensions before processing. A decompression-bomb image (e.g., 30,000x30,000 pixels = ~2.7 GB RAM) could cause OOM and crash.

## Findings

**Security Sentinel (Medium)**: Denial of service via memory exhaustion.

## Proposed Solutions

### Option A: Add Pixel Count Guard (Recommended)

```python
h, w = img.shape[:2]
MAX_PIXELS = 4096 * 4096  # ~16 megapixels
if h * w > MAX_PIXELS:
    logger.warning(f"Image too large ({w}x{h}), skipping")
    return []
```

**Effort:** Small | **Risk:** Low

## Technical Details

- **File:** `core/analysis/faces.py` after `cv2.imread` call

## Acceptance Criteria

- [ ] Oversized images rejected with warning
- [ ] Normal images (up to 16MP) still processed

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Security Sentinel | Guard against decompression bombs before ML inference |

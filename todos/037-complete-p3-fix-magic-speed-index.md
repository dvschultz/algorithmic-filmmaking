---
status: complete
priority: p3
issue_id: "037"
tags: [code-review, code-quality, video-player]
dependencies: []
---

# Replace Magic Index 3 for 1.0x Speed Default

## Problem Statement

In `VideoPlayer`, the default speed combo box index is hardcoded as `3` (the position of "1.0x" in the speed list). If the speed list order changes, this silently selects the wrong default.

## Findings

**Python Reviewer**: Magic index `3` for 1.0x speed. Should use `SPEEDS.index(1.0)` or a named constant.

## Proposed Solutions

### Option A: Use list.index() (Recommended)

```python
PLAYBACK_SPEEDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
DEFAULT_SPEED_INDEX = PLAYBACK_SPEEDS.index(1.0)
```

**Pros:** Self-documenting, resilient to list changes
**Cons:** Trivial
**Effort:** Small
**Risk:** Low

## Acceptance Criteria

- [ ] No magic number for speed index
- [ ] Default speed is still 1.0x
- [ ] Works if speed list is reordered

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-27 | Created from branch review — Python Reviewer finding | Magic indices break when lists change |

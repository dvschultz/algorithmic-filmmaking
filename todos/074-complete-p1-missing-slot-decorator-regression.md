---
status: complete
priority: p1
issue_id: "074"
tags: [code-review, quality, regression]
dependencies: []
---

# Missing @Slot Decorator on _on_object_detection_progress (Regression)

## Problem Statement

The `@Slot(int, int)` decorator that was on `_on_object_detection_progress` in `ui/main_window.py` has been moved to the new `_on_face_detection_progress` method. The original method now has NO `@Slot` decorator. This is a regression — without the decorator, the slot won't be properly registered with Qt's meta-object system, which can cause subtle thread-safety issues with signal/slot connection types.

## Findings

**Python Reviewer (Critical)**: The decorator was moved rather than copied. The new face detection handler got it, the old object detection handler lost it.

**Architecture Strategist**: Confirmed the missing decorator — the method immediately above has `@Slot(int, int)` and the one below does not.

## Proposed Solutions

### Option A: Add @Slot Back (Recommended)

Add `@Slot(int, int)` to `_on_object_detection_progress` in `ui/main_window.py`.

**Effort:** Small (one line) | **Risk:** Low

## Technical Details

- **File:** `ui/main_window.py` — `_on_object_detection_progress` method

## Acceptance Criteria

- [ ] Both `_on_face_detection_progress` and `_on_object_detection_progress` have `@Slot(int, int)`

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Python Reviewer + Architecture Strategist | When adding new signal handlers, don't move decorators — copy them |

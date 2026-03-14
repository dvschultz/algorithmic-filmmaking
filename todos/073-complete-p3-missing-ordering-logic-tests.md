---
status: complete
priority: p3
issue_id: "073"
tags: [code-review, testing, rose-hobart]
dependencies: ["059"]
---

# Missing Test Coverage for Ordering Logic

## Problem Statement

Neither the dialog `_order_clips` nor the chat tool ordering is tested. This is the most complex logic in the feature after face comparison. The `tests/test_rose_hobart_dialog.py` file referenced in the plan as completed is absent from the repository. No test covers the Generate button enable/disable, sensitivity dropdown, or ordering dropdown behavior.

## Findings

**Code Simplicity Reviewer**: The ordering logic is the part most likely to regress during the refactor to a shared helper (todo 059). Zero test coverage.

## Proposed Solutions

### Option A: Add Unit Tests for Ordering (Recommended)

Test the shared ordering function (after todo 059) with each of the 6 strategies using mock clip/source data.

**Effort:** Medium | **Risk:** Low

## Technical Details

- Depends on todo 059 (extract shared ordering helper)
- Test all 6 strategies: original, duration, color, brightness, confidence, random

## Acceptance Criteria

- [ ] Each ordering strategy has at least one test
- [ ] Edge cases tested (empty list, single clip, clips without colors)

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Code Simplicity Reviewer | Always test sorting/ordering logic — it's the most likely to regress |

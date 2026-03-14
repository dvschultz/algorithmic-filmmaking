---
status: pending
priority: p3
issue_id: "083"
tags: [code-review, quality, simplicity]
dependencies: []
---

# Remove Deprecated reference_image_path Singular Param from Brand-New Function

## Problem Statement

`generate_rose_hobart` in `core/chat_tools.py` accepts both `reference_image_path` (singular, marked deprecated) and `reference_image_paths` (plural). This is a brand-new function with no prior API to maintain backward compatibility with. The singular form adds ~8 lines of unnecessary fallback logic.

## Findings

- **Code Simplicity Reviewer**: Finding #3

## Proposed Solutions

### Option A: Remove singular param entirely
Delete `reference_image_path` parameter and its fallback logic. Only accept `reference_image_paths`.

- **Effort**: Small
- **Risk**: Low (no existing callers to break)

## Acceptance Criteria

- [ ] Only `reference_image_paths` parameter exists
- [ ] No deprecated param in a function that has never been released

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | YAGNI - no backward compat needed for new code |

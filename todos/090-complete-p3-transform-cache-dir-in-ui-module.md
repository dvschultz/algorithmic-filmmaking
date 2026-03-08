---
status: pending
priority: p3
issue_id: "090"
tags: [code-review, architecture]
dependencies: []
---

# _get_transform_cache_dir Lives in UI Dialog Module

## Problem Statement

`_get_transform_cache_dir` in `ui/dialogs/dice_roll_dialog.py` lines 34-37 is imported by `ui/tabs/sequence_tab.py` line 1564 for the agent code path. UI dialog module should not be a dependency for core logic.

## Findings

- **Code Reviewer**: P3 architecture issue

## Proposed Solutions

### Option A: Move to core module
Move to `core/remix/prerender.py` or `core/settings.py` where it logically belongs.

- **Effort**: Small
- **Risk**: Low

## Acceptance Criteria

- [ ] No imports from `ui/dialogs/` in non-dialog code paths

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | UI dialog leaks into core dependency graph |

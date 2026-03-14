---
status: complete
priority: p3
issue_id: "067"
tags: [code-review, quality, rose-hobart]
dependencies: []
---

# Dead Code and Unused Imports Cleanup

## Problem Statement

Several minor cleanup items across the feature:

1. `Optional` imported but unused in `core/analysis/faces.py` (line 12) — all functions use modern `| None`
2. `_ReferenceImageWidget.best_embedding` property (dialog lines 272-276) is never called — dead code
3. `Optional` used in `rose_hobart_dialog.py` where `| None` should be preferred for Python 3.11+ consistency
4. `colorsys` imported inside function body (dialog line 181) instead of at module level

## Findings

**Python Reviewer**: Unused import, inconsistent typing style, dead property.
**Code Simplicity Reviewer**: `best_embedding` is dead code — dialog passes image paths to worker, which re-extracts. ~10 LOC total.

## Proposed Solutions

### Option A: Clean Up All Items

Remove unused `Optional` import, delete `best_embedding` property, move `colorsys` to module level, replace `Optional[X]` with `X | None`.

**Effort:** Small | **Risk:** Low

## Technical Details

- **Files:** `core/analysis/faces.py` line 12, `ui/dialogs/rose_hobart_dialog.py` lines 11, 181, 272-276

## Acceptance Criteria

- [ ] No unused imports
- [ ] No dead code
- [ ] Consistent typing style

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Python Reviewer + Code Simplicity | Keep imports and typing style consistent |

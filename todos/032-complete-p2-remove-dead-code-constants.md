---
status: complete
priority: p2
issue_id: "032"
tags: [code-review, simplicity, dead-code, constants]
dependencies: []
---

# Remove Dead Code in core/constants.py

## Problem Statement

`core/constants.py` imports `SHOT_TYPES` from `core/analysis/shots.py` but never uses it. `SHOT_TYPE_DISPLAY_TO_ANALYSIS` mapping is defined but has no consumers. `VALID_SHOT_TYPES` is not derived from `SHOT_TYPES` despite the comment suggesting it should be.

## Findings

**Simplicity Reviewer**:
- `from core.analysis.shots import SHOT_TYPES` — unused import
- `SHOT_TYPE_DISPLAY_TO_ANALYSIS` — defined but no code references it
- Comment says "Maps from display name -> analysis name (lowercase in SHOT_TYPES)" but VALID_SHOT_TYPES is a plain list

## Proposed Solutions

### Option A: Remove Unused Code (Recommended)

1. Remove `from core.analysis.shots import SHOT_TYPES`
2. Remove `SHOT_TYPE_DISPLAY_TO_ANALYSIS` dict
3. Keep `VALID_SHOT_TYPES`, `VALID_ASPECT_RATIOS`, `VALID_COLOR_PALETTES`, `VALID_SORT_ORDERS`

**Pros:** Cleaner, no dead code
**Cons:** None
**Effort:** Small
**Risk:** Low

## Acceptance Criteria

- [ ] Unused import removed
- [ ] Unused mapping removed
- [ ] No other file references the removed code (verify with grep)
- [ ] All tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-27 | Created from branch review — Simplicity Reviewer finding | New constants files should only contain what's actually consumed |

---
status: pending
priority: p3
issue_id: "019"
tags: [code-review, simplification, yagni]
dependencies: []
---

# Remove Unused sort_colors_by_hue Function

## Problem Statement

The `sort_colors_by_hue()` function in `core/analysis/color.py` is defined and exported but never used anywhere in the codebase. This is a YAGNI violation.

**Why it matters:** Dead code increases maintenance burden and cognitive load.

## Findings

**Location:** `/Users/derrickschultz/repos/algorithmic-filmmaking/core/analysis/color.py:108-121`

```python
def sort_colors_by_hue(
    colors: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """Sort colors by their hue value."""
    return sorted(colors, key=lambda c: rgb_to_hsv(c)[0])
```

**Usage search results:**
- Defined in `color.py` - YES
- Exported in `__init__.py` - YES
- Used anywhere - NO

The function was likely added speculatively for "maybe someday" sorting colors within a clip's swatch display.

**Found by:** code-simplicity-reviewer agent

## Proposed Solutions

### Option A: Delete the Function (Recommended)
Remove from both `color.py` and `__init__.py`:
- **Pros:** 14 LOC removed, cleaner API surface
- **Cons:** None (function is unused)
- **Effort:** Small
- **Risk:** None

### Option B: Keep but Mark as Internal
Remove from exports but keep the function:
- **Pros:** Available if needed later
- **Cons:** Still dead code
- **Effort:** Small
- **Risk:** None

## Technical Details

**Files to modify:**
- `core/analysis/color.py` - delete lines 108-121
- `core/analysis/__init__.py` - remove `sort_colors_by_hue` from imports and exports

**Also consider:** Remove `rgb_to_hsv` from exports (it's only used internally by `get_primary_hue`)

## Acceptance Criteria

- [ ] `sort_colors_by_hue` function deleted
- [ ] `__init__.py` exports only `extract_dominant_colors` and `get_primary_hue`
- [ ] All tests pass
- [ ] Application still functions correctly

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | YAGNI - don't add functions until they're needed |

## Resources

- YAGNI principle documentation

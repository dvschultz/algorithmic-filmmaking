---
status: complete
priority: p2
issue_id: "003"
tags: [code-review, duplication, cleanup]
dependencies: []
---

# Remove Duplicate .edl Suffix Validation

## Problem Statement

The `.edl` suffix is validated and added in TWO locations - both the UI handler and the export function. This violates DRY and adds unnecessary code.

## Findings

**Location 1:** `ui/main_window.py` lines 1232-1233

```python
output_path = Path(file_path)
if not output_path.suffix.lower() == ".edl":
    output_path = output_path.with_suffix(".edl")
```

**Location 2:** `core/edl_export.py` lines 127-129

```python
output_path = config.output_path
if not output_path.suffix.lower() == ".edl":
    output_path = output_path.with_suffix(".edl")
```

Additionally, the file dialog filter already specifies `*.edl`, so users selecting from the dialog will naturally have the correct suffix.

## Proposed Solutions

### Option A: Keep Only in export_edl() (Recommended)
**Pros:** Core function handles all validation, UI stays simple
**Cons:** None
**Effort:** Small (5 min)
**Risk:** None

Remove the check from `main_window.py`, keep in `export_edl()`.

### Option B: Keep Only in UI Handler
**Pros:** Core function assumes valid input
**Cons:** Core function would be less robust if called directly
**Effort:** Small (5 min)
**Risk:** Low

### Option C: Keep Both
**Pros:** Defense in depth
**Cons:** Code duplication, violates DRY
**Effort:** None
**Risk:** None

## Recommended Action

Option A - Keep validation in core function, remove from UI

## Technical Details

**Affected Files:**
- `ui/main_window.py` - Remove lines 1232-1233

## Acceptance Criteria

- [ ] Suffix check removed from one location
- [ ] Export still adds .edl suffix when needed
- [ ] No regression in functionality

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Validate at boundaries, not everywhere |

## Resources

- PR #7: https://github.com/dvschultz/algorithmic-filmmaking/pull/7
- Pattern Recognition Specialist review findings

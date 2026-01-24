---
status: pending
priority: p3
issue_id: "006"
tags: [code-review, yagni, dead-code]
dependencies: []
---

# Remove Unused drop_frame Parameter

## Problem Statement

The `EDLExportConfig.drop_frame` parameter is defined with a default of `False` but is never set to `True` anywhere in the codebase. This is a YAGNI violation - building for a feature that isn't needed.

## Findings

**Location:** `core/edl_export.py` line 17

```python
@dataclass
class EDLExportConfig:
    output_path: Path
    title: str = "Scene Ripper Export"
    drop_frame: bool = False  # <-- Never set to True anywhere
```

The `drop_frame` parameter affects timecode formatting (`;` vs `:` separator) but since it's always `False`, all EDL files use non-drop-frame format.

## Proposed Solutions

### Option A: Remove drop_frame Parameter (Recommended)
**Pros:** YAGNI compliance, simpler API, removes dead code path
**Cons:** Would need to re-add if drop-frame support needed later
**Effort:** Small (10 min)
**Risk:** None - feature isn't used

### Option B: Keep for Future Compatibility
**Pros:** API won't change if drop-frame needed later
**Cons:** Dead code, violates YAGNI
**Effort:** None
**Risk:** None

### Option C: Actually Implement Drop-Frame Support
**Pros:** Complete feature
**Cons:** No requirement for it, more work
**Effort:** Medium
**Risk:** Low

## Recommended Action

Option A - Remove unused parameter. Add back when there's an actual requirement.

## Technical Details

**Affected Files:**
- `core/edl_export.py` - Remove from EDLExportConfig, remove from function calls

**Note:** The `frames_to_timecode()` function would need to be simplified to always use `:` separator.

## Acceptance Criteria

- [ ] drop_frame parameter removed from EDLExportConfig
- [ ] frames_to_timecode() simplified (or kept if used elsewhere)
- [ ] All EDL exports use non-drop-frame format (unchanged behavior)
- [ ] No regression in functionality

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Don't build features until they're needed |

## Resources

- PR #7: https://github.com/dvschultz/algorithmic-filmmaking/pull/7
- Code Simplicity Reviewer findings

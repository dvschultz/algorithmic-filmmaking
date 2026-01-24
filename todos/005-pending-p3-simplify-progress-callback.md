---
status: pending
priority: p3
issue_id: "005"
tags: [code-review, simplification, yagni]
dependencies: []
---

# Simplify or Remove progress_callback from EDL Export

## Problem Statement

The `export_edl()` function has a `progress_callback` parameter with 9 conditional checks throughout the code, but:
1. EDL export is extremely fast (milliseconds)
2. The caller comments: "EDL export is fast, no need for worker thread"
3. The complexity adds no user-visible benefit

This is a YAGNI violation - building progress infrastructure for a feature that doesn't need it.

## Findings

**Location:** `core/edl_export.py` - progress_callback used at lines 60, 76, 83, 122, 133, 139

The function has 9 `if progress_callback:` checks for an operation that takes <30ms for 1000 clips.

**Caller:** `ui/main_window.py` line 1255-1256

```python
progress_callback=lambda p, m: self.status_bar.showMessage(m),
```

Only the message is used - the progress value is ignored!

## Proposed Solutions

### Option A: Remove progress_callback Entirely (Recommended)
**Pros:** Simpler code, removes 15 LOC, YAGNI compliance
**Cons:** API change
**Effort:** Small (15 min)
**Risk:** None - operation is synchronous and fast

Move status messages to caller:
```python
self.status_bar.showMessage("Exporting EDL...")
success = export_edl(sequence, sources, config)
if success:
    self.status_bar.showMessage(f"EDL exported to {output_path.name}")
```

### Option B: Keep for API Consistency
**Pros:** Matches other export functions
**Cons:** Unused complexity, 9 conditional checks
**Effort:** None
**Risk:** None

### Option C: Keep but Simplify to Only Entry/Exit
**Pros:** Some progress indication, less code
**Cons:** Still has unnecessary abstraction
**Effort:** Small (10 min)
**Risk:** None

## Recommended Action

Option A - Remove progress_callback, handle messaging in caller

## Technical Details

**Affected Files:**
- `core/edl_export.py` - Remove parameter and all conditional checks
- `ui/main_window.py` - Update caller to show status directly

## Acceptance Criteria

- [ ] progress_callback parameter removed (or simplified)
- [ ] Status messages still shown to user
- [ ] Code is simpler and easier to understand
- [ ] Export still works correctly

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Don't add progress for synchronous fast operations |

## Resources

- PR #7: https://github.com/dvschultz/algorithmic-filmmaking/pull/7
- Code Simplicity Reviewer findings
- Performance Oracle findings (operation takes <30ms)

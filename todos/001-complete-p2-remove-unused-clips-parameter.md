---
status: complete
priority: p2
issue_id: "001"
tags: [code-review, dead-code, api-cleanup]
dependencies: []
---

# Remove Unused `clips` Parameter from export_edl()

## Problem Statement

The `export_edl()` function accepts a `clips: dict[str, tuple]` parameter that is never used in the function body. The caller in `main_window.py` builds this dictionary unnecessarily. This is dead code that adds confusion and maintenance burden.

## Findings

**Location:** `core/edl_export.py` lines 44, 53

```python
def export_edl(
    sequence: Sequence,
    sources: dict[str, Source],
    clips: dict[str, tuple],  # <-- NEVER USED
    config: EDLExportConfig,
    progress_callback: Callable[[float, str], None] | None = None,
) -> bool:
```

The function iterates over `sequence.get_all_clips()` and looks up sources via `sources.get(seq_clip.source_id)`. The `clips` dict is never referenced.

**Caller builds unused dict:** `ui/main_window.py` lines 1241-1243

```python
clips = {}
for clip in self.clips:
    if self.current_source:
        clips[clip.id] = (clip, self.current_source)
```

This loop runs unnecessarily.

## Proposed Solutions

### Option A: Remove Parameter Entirely (Recommended)
**Pros:** Clean API, removes dead code, simplifies caller
**Cons:** None
**Effort:** Small (10 min)
**Risk:** None

1. Remove `clips` parameter from `export_edl()` signature
2. Remove `clips` from docstring
3. Update caller to not build/pass clips dict

### Option B: Keep for Future Compatibility
**Pros:** API won't change if clips needed later
**Cons:** Violates YAGNI, keeps dead code
**Effort:** None
**Risk:** Future confusion

## Recommended Action

Option A - Remove the unused parameter

## Technical Details

**Affected Files:**
- `core/edl_export.py` - Remove parameter, update docstring
- `ui/main_window.py` - Remove dict building and parameter passing

## Acceptance Criteria

- [ ] `clips` parameter removed from `export_edl()` signature
- [ ] Docstring updated to not reference `clips`
- [ ] Caller no longer builds or passes `clips` dict
- [ ] EDL export still works correctly
- [ ] No linting errors

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Dead parameters should be removed promptly |

## Resources

- PR #7: https://github.com/dvschultz/algorithmic-filmmaking/pull/7
- Pattern Recognition Specialist review findings
- Architecture Strategist review findings

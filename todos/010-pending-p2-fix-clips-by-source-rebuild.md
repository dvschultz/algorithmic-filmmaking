---
status: pending
priority: p2
issue_id: "010"
tags: [bug, state-sync, detection, multi-source]
dependencies: ["007"]
---

# Fix clips_by_source Partial Rebuild Issue

## Problem Statement

When scene detection finishes for a source, `_on_detection_finished()` doesn't fully rebuild `clips_by_source`. If clips existed for that source before re-analysis, the dict may contain stale entries or duplicates.

## Findings

**Location:** `ui/main_window.py` in `_on_detection_finished()`

The method adds new clips to `clips_by_source[source_id]` but doesn't clear previous clips for that source first. If a user:
1. Detects scenes (gets 10 clips)
2. Changes threshold and re-detects (gets 8 clips)

The old 10 clips may still be in `clips_by_source` alongside the new 8.

**Related Issues:**
- `AnalyzeTab._clips` also has sync issues with MainWindow state
- Multiple parallel structures track clips (clips list, clips_by_source dict, ClipBrowser)

## Proposed Solutions

### Option A: Clear Source Entry Before Adding (Recommended)
**Pros:** Simple fix, maintains current architecture
**Cons:** Doesn't address root cause (multiple parallel structures)
**Effort:** Small
**Risk:** Low

```python
def _on_detection_finished(self, clips, source_id):
    # Clear old clips for this source first
    self.clips_by_source[source_id] = []

    # Add new clips
    for clip in clips:
        self.clips.append(clip)
        self.clips_by_source[source_id].append(clip)
```

### Option B: Use defaultdict with Clear on Re-detect
**Pros:** Slightly cleaner syntax
**Cons:** Same fundamental approach
**Effort:** Small
**Risk:** Low

### Option C: Single Source of Truth Refactor
**Pros:** Eliminates sync issues entirely
**Cons:** Larger refactor, outside current scope
**Effort:** Large
**Risk:** Medium

## Recommended Action

Option A for now, consider Option C in future architecture cleanup

## Technical Details

**Affected Files:**
- `ui/main_window.py` - Clear source entry in `_on_detection_finished()`

**Verification:**
1. Detect scenes for a source
2. Note clip count
3. Change threshold and re-detect
4. Verify clip count matches new detection, not sum of old + new

## Acceptance Criteria

- [ ] Re-detection replaces clips, doesn't accumulate
- [ ] `clips_by_source[source_id]` matches actual clips for source
- [ ] Main `clips` list stays in sync
- [ ] ClipBrowser display matches internal state

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Parallel data structures need coordinated updates |

## Resources

- Data Integrity Guardian review findings
- Architecture Strategist review findings

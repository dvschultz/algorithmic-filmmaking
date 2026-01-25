---
status: complete
priority: p1
issue_id: "007"
tags: [bug, data-integrity, project-load, multi-source]
dependencies: []
---

# Fix Project Load Source Mapping Bug

## Problem Statement

When loading a project with multiple sources, ALL clips are mapped to the first source instead of their correct source. This causes data corruption where clips appear associated with the wrong video file.

## Findings

**Location:** `ui/main_window.py` around line 1957

```python
for clip in clips:
    self.analyze_tab.add_clip(clip, self.current_source)  # BUG: Uses first source for ALL clips
```

The code iterates through all clips but passes `self.current_source` (the first loaded source) to every `add_clip()` call. Each clip has a `source_id` attribute that should be used to look up the correct source.

**Impact:**
- Thumbnails generated from wrong video file
- Clip preview shows wrong footage
- Timeline playback incorrect
- Re-saving project could corrupt clip associations

## Proposed Solutions

### Option A: Look Up Source Per Clip (Recommended)
**Pros:** Correct behavior, uses existing clip.source_id
**Cons:** None
**Effort:** Small
**Risk:** None

```python
sources_by_id = {s.id: s for s in sources}
for clip in clips:
    source = sources_by_id.get(clip.source_id)
    if source:
        self.analyze_tab.add_clip(clip, source)
    else:
        logging.warning(f"Clip {clip.id} references unknown source {clip.source_id}")
```

## Recommended Action

Option A - Build source lookup dict and use clip.source_id

## Technical Details

**Affected Files:**
- `ui/main_window.py` - Fix `_load_project()` or relevant project load method

**Verification:**
1. Create project with 2+ videos, detect scenes in each
2. Save project
3. Close and reload project
4. Verify each clip shows correct video source thumbnail
5. Verify clip preview plays correct footage

## Acceptance Criteria

- [ ] Each clip is passed to `add_clip()` with its correct source
- [ ] Source lookup handles missing sources gracefully
- [ ] Project with multiple sources loads correctly
- [ ] Clip thumbnails match their source videos

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Multi-source state requires per-item source lookup |

## Resources

- Data Integrity Guardian review findings
- Python Quality Reviewer findings

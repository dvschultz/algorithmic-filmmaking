---
status: complete
priority: p1
issue_id: "008"
tags: [bug, data-integrity, timeline, multi-source]
dependencies: ["007"]
---

# Fix load_sequence Passing Only First Source

## Problem Statement

The `load_sequence()` call in project loading only passes the first source, but the timeline may contain clips from multiple sources. This breaks timeline playback for clips from non-primary sources.

## Findings

**Location:** `ui/main_window.py` around line 1968

```python
self.sequence_tab.timeline.load_sequence(sequence, sources[0], clips)
```

The timeline's `load_sequence()` method receives only `sources[0]` but the sequence may contain `SequenceClip` objects referencing different sources via `source_id`.

**Impact:**
- Timeline clips from secondary sources cannot be previewed
- Export may use wrong source file for clips
- FPS calculations may be incorrect for clips with different source framerates

## Proposed Solutions

### Option A: Pass All Sources Dict (Recommended)
**Pros:** Timeline can look up any source, consistent with export pattern
**Cons:** Requires updating `load_sequence()` signature
**Effort:** Small
**Risk:** Low

```python
sources_by_id = {s.id: s for s in sources}
self.sequence_tab.timeline.load_sequence(sequence, sources_by_id, clips)
```

Update `load_sequence()` to accept `sources: dict[str, Source]` instead of single source.

### Option B: Timeline Tracks Own Source Registry
**Pros:** Timeline self-contained
**Cons:** Duplicates state, more complex
**Effort:** Medium
**Risk:** State sync issues

## Recommended Action

Option A - Pass sources dict, update signature

## Technical Details

**Affected Files:**
- `ui/main_window.py` - Change call to pass dict
- `ui/timeline.py` - Update `load_sequence()` signature and usage

**Verification:**
1. Create sequence with clips from 2+ sources
2. Save project
3. Reload and verify timeline shows all clips correctly
4. Play timeline and verify each clip plays from correct source

## Acceptance Criteria

- [ ] `load_sequence()` accepts dict of sources
- [ ] Timeline can preview clips from any source
- [ ] Timeline playback uses correct source per clip
- [ ] No regressions in single-source projects

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Timeline needs access to all sources for multi-source playback |

## Resources

- Data Integrity Guardian review findings
- Architecture Strategist review findings

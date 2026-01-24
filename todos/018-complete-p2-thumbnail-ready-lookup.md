---
status: complete
priority: p2
issue_id: "018"
tags: [code-review, performance, bug-fix]
dependencies: []
---

# Use clips_by_id Dict in _on_thumbnail_ready

## Problem Statement

The `_on_thumbnail_ready` method uses O(n) linear search to find clips by ID, despite a `clips_by_id` dictionary being available. This creates O(n^2) complexity over all thumbnail generations.

**Why it matters:** For 1000 clips, this causes ~500,000 unnecessary comparisons during thumbnail loading.

## Findings

**Location:** `/Users/derrickschultz/repos/algorithmic-filmmaking/ui/main_window.py:477-483`

```python
def _on_thumbnail_ready(self, clip_id: str, thumb_path: str):
    """Handle individual thumbnail completion."""
    # Find clip and add to browser
    for clip in self.clips:  # O(n) search
        if clip.id == clip_id:
            self.clip_browser.add_clip(clip, self.current_source)
            break
```

The `clips_by_id` dictionary is already created at line 455:
```python
self.clips_by_id = {clip.id: clip for clip in clips}
```

But `_on_thumbnail_ready` doesn't use it.

Note: `_on_color_ready` (line 513) correctly uses `clips_by_id.get()`.

**Found by:** performance-oracle agent

## Proposed Solutions

### Option A: Use clips_by_id (Recommended)
```python
def _on_thumbnail_ready(self, clip_id: str, thumb_path: str):
    """Handle individual thumbnail completion."""
    clip = self.clips_by_id.get(clip_id)
    if clip:
        self.clip_browser.add_clip(clip, self.current_source)
```
- **Pros:** O(1) lookup, consistent with _on_color_ready
- **Cons:** None
- **Effort:** Small (2 line change)
- **Risk:** Low

## Technical Details

**Affected files:**
- `ui/main_window.py` - _on_thumbnail_ready method

**Complexity improvement:**
- Before: O(n^2) total for n clips
- After: O(n) total for n clips

## Acceptance Criteria

- [ ] _on_thumbnail_ready uses clips_by_id dictionary
- [ ] Thumbnail loading remains functional
- [ ] Performance improvement measurable for 100+ clips

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | clips_by_id dict exists but wasn't used consistently |

## Resources

- Performance analysis in review

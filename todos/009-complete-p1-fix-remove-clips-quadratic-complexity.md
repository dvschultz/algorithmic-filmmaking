---
status: complete
priority: p1
issue_id: "009"
tags: [performance, algorithm, clip-browser]
dependencies: []
---

# Fix O(n^2) Algorithm in remove_clips_for_source

## Problem Statement

The `remove_clips_for_source()` method in ClipBrowser has O(n^2) complexity due to calling `list.remove()` inside a loop. For videos with many clips (100+), this causes noticeable UI freezes.

## Findings

**Location:** `ui/clip_browser.py` in `remove_clips_for_source()`

```python
thumbs_to_remove = [t for t in self.thumbnails if t.clip.source_id == source_id]
for thumb in thumbs_to_remove:
    self.grid.removeWidget(thumb)
    thumb.deleteLater()
    self.thumbnails.remove(thumb)  # O(n) operation inside O(n) loop = O(n^2)
```

Each `self.thumbnails.remove(thumb)` call is O(n) because it must search the list to find the item. When removing m clips from a list of n thumbnails, this is O(m*n).

**Impact:**
- UI freeze when re-analyzing a source with many clips
- Proportionally worse as clip count grows
- 500 clips with 100 to remove = 50,000 operations

## Proposed Solutions

### Option A: Filter and Reassign (Recommended)
**Pros:** O(n) complexity, simple, Pythonic
**Cons:** None
**Effort:** Small
**Risk:** None

```python
def remove_clips_for_source(self, source_id: str):
    # Separate into keep and remove
    keep = []
    remove = []
    for thumb in self.thumbnails:
        if thumb.clip.source_id == source_id:
            remove.append(thumb)
        else:
            keep.append(thumb)

    # Clean up widgets
    for thumb in remove:
        self.grid.removeWidget(thumb)
        thumb.deleteLater()
        del self._thumbnail_by_clip_id[thumb.clip.id]

    # Replace list in one operation
    self.thumbnails = keep
```

### Option B: Use Set for Removal Tracking
**Pros:** Also O(n)
**Cons:** More complex, needs ID extraction
**Effort:** Small
**Risk:** None

## Recommended Action

Option A - Single-pass filter with list reassignment

## Technical Details

**Affected Files:**
- `ui/clip_browser.py` - Rewrite `remove_clips_for_source()`

**Verification:**
1. Import video with 100+ scenes detected
2. Time the re-analysis operation
3. Confirm no UI freeze
4. Verify all clips correctly removed and replaced

## Acceptance Criteria

- [ ] `remove_clips_for_source()` is O(n) complexity
- [ ] No `list.remove()` inside loops
- [ ] UI remains responsive during clip removal
- [ ] Widget cleanup (deleteLater) still happens correctly

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Avoid list.remove() in loops - use filter + reassign |

## Resources

- Performance Oracle review findings
- Pattern Recognition Specialist review findings

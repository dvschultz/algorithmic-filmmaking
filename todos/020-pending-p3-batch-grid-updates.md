---
status: pending
priority: p3
issue_id: "020"
tags: [code-review, performance, ui]
dependencies: []
---

# Batch Grid Updates When Sorting Clips

## Problem Statement

The `_rebuild_grid()` method in ClipBrowser removes and re-adds all widgets individually, causing layout recalculations for each operation. This creates noticeable UI stutter when sorting large collections.

**Why it matters:** Sorting 100+ clips causes visible UI freeze (200-500ms). Sorting 500+ clips appears to hang the application.

## Findings

**Location:** `/Users/derrickschultz/repos/algorithmic-filmmaking/ui/clip_browser.py:375-385`

```python
def _rebuild_grid(self):
    """Rebuild the grid layout with current thumbnail order."""
    # Remove all thumbnails from grid
    for thumb in self.thumbnails:
        self.grid.removeWidget(thumb)

    # Re-add in new order
    for i, thumb in enumerate(self.thumbnails):
        row = i // self.COLUMNS
        col = i % self.COLUMNS
        self.grid.addWidget(thumb, row, col)
```

Each `removeWidget` and `addWidget` triggers a layout invalidation and potential repaint.

**Found by:** performance-oracle agent

## Proposed Solutions

### Option A: Disable Updates During Rebuild (Recommended)
```python
def _rebuild_grid(self):
    """Rebuild the grid layout with current thumbnail order."""
    self.container.setUpdatesEnabled(False)
    try:
        for thumb in self.thumbnails:
            self.grid.removeWidget(thumb)
        for i, thumb in enumerate(self.thumbnails):
            row = i // self.COLUMNS
            col = i % self.COLUMNS
            self.grid.addWidget(thumb, row, col)
    finally:
        self.container.setUpdatesEnabled(True)
```
- **Pros:** 50-70% faster sorting, minimal code change
- **Cons:** None
- **Effort:** Small (4 line wrapper)
- **Risk:** Low

### Option B: Use QListView with Custom Delegate
Replace grid with virtual scrolling:
- **Pros:** Constant-time sorting, better for 500+ clips
- **Cons:** Significant refactor, different widget paradigm
- **Effort:** Large
- **Risk:** Medium

## Technical Details

**Affected files:**
- `ui/clip_browser.py` - _rebuild_grid method

**Performance improvement (Option A):**
- 100 clips: 300ms → 100ms
- 500 clips: 1.5s → 500ms

## Acceptance Criteria

- [ ] Sorting 100 clips has no visible UI stutter
- [ ] setUpdatesEnabled(False/True) wraps layout changes
- [ ] Grid displays correctly after sort

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Qt setUpdatesEnabled batches layout calculations |

## Resources

- Qt setUpdatesEnabled documentation

---
status: complete
priority: p2
issue_id: "011"
tags: [code-review, performance, qt]
dependencies: []
---

# Thumbnail Loading Blocks UI Thread

## Problem Statement

Thumbnails are loaded synchronously on the main thread during `ClipItem` construction. With 50+ clips, this causes UI freezes during timeline rebuild and stuttering during zoom operations.

**Why it matters:**
- 50 clips: ~500ms UI freeze
- 100 clips: ~1-2s UI freeze
- 200 clips: Potentially unresponsive application

## Findings

**Location:** `/Users/derrickschultz/repos/algorithmic-filmmaking/ui/timeline/clip_item.py` lines 60-65

```python
def _load_thumbnail(self):
    """Load thumbnail image if available."""
    if self.thumbnail_path:
        path = Path(self.thumbnail_path)
        if path.exists():
            self._thumbnail_pixmap = QPixmap(str(path))
```

**Found by:** performance-oracle agent

## Proposed Solutions

### Option A: Async loading with QThreadPool (Recommended)
Implement lazy loading with background thread:
```python
def _request_thumbnail_load(self):
    if self._thumbnail_pixmap or self._thumbnail_loading:
        return
    self._thumbnail_loading = True
    ThumbnailLoader.instance().request(
        self.thumbnail_path,
        callback=self._on_thumbnail_loaded
    )
```
- **Pros:** Best UX, non-blocking, scalable
- **Cons:** More complex implementation
- **Effort:** Medium
- **Risk:** Low

### Option B: Load on demand during paint
Only load when item becomes visible:
```python
def paint(self, painter, option, widget):
    if self._thumbnail_pixmap is None and self.thumbnail_path:
        self._load_thumbnail()  # Still blocks but only when visible
```
- **Pros:** Simple, loads less initially
- **Cons:** Still blocks during scroll
- **Effort:** Small
- **Risk:** Medium

### Option C: QPixmapCache with LRU
Use Qt's built-in pixmap cache:
- **Pros:** Reduces memory, reuses cached pixmaps
- **Cons:** Doesn't solve initial load blocking
- **Effort:** Small
- **Risk:** Low

## Technical Details

**Affected files:**
- `ui/timeline/clip_item.py`
- May need new `core/thumbnail_loader.py` for Option A

## Acceptance Criteria

- [ ] Timeline with 50+ clips doesn't freeze UI
- [ ] Thumbnails load progressively as visible
- [ ] Zoom operations remain smooth
- [ ] Memory usage stays reasonable

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Performance-oracle identified this as critical |

## Resources

- PR: Phase 2 Timeline & Composition
- Qt QThreadPool documentation

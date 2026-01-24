---
status: complete
priority: p2
issue_id: "012"
tags: [code-review, performance, qt]
dependencies: []
---

# O(n) Scene Update on Every Zoom Change

## Problem Statement

Every zoom operation iterates through ALL clips, triggering geometry updates and repaints for items that may not even be visible. This causes choppy zoom experience with many clips.

**Why it matters:**
- 50 clips: Noticeable lag during zoom
- 100+ clips: Choppy zoom experience, dropped frames
- 200+ clips: Unusable zoom interaction

## Findings

**Location:** `/Users/derrickschultz/repos/algorithmic-filmmaking/ui/timeline/timeline_scene.py` lines 94-107

```python
def set_pixels_per_second(self, pps: float):
    self.pixels_per_second = pps
    self._update_scene_rect()

    # Update all clip positions - O(n) iteration
    for clip_item in self._clip_items.values():
        clip_item.set_pixels_per_second(pps)

    self.update()  # Forces full repaint
```

**Found by:** performance-oracle agent

## Proposed Solutions

### Option A: Viewport-based updates (Recommended)
Only update visible items:
```python
def set_pixels_per_second(self, pps: float):
    self.pixels_per_second = pps
    visible_rect = self.views()[0].mapToScene(
        self.views()[0].viewport().rect()
    ).boundingRect()

    for clip_item in self._clip_items.values():
        if clip_item.sceneBoundingRect().intersects(visible_rect):
            clip_item.set_pixels_per_second(pps)
```
- **Pros:** Significant performance improvement, simple concept
- **Cons:** Need to update off-screen items eventually
- **Effort:** Small
- **Risk:** Low

### Option B: Debounced zoom updates
Use QTimer to batch rapid zoom events:
```python
self._zoom_timer = QTimer()
self._zoom_timer.setSingleShot(True)
self._zoom_timer.setInterval(16)  # ~60fps
self._zoom_timer.timeout.connect(self._apply_pending_zoom)
```
- **Pros:** Smoother feel during rapid zoom
- **Cons:** Slight visual lag
- **Effort:** Small
- **Risk:** Low

### Option C: Transform-based zoom
Use view transform instead of repositioning items.
- **Pros:** Best performance
- **Cons:** More complex, may affect click handling
- **Effort:** Medium
- **Risk:** Medium

## Technical Details

**Affected files:**
- `ui/timeline/timeline_scene.py`
- `ui/timeline/timeline_view.py`

## Acceptance Criteria

- [ ] Zoom with 50+ clips feels smooth
- [ ] No dropped frames during zoom
- [ ] Visible clips update immediately
- [ ] Off-screen clips update when scrolled into view

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | O(n) per zoom event is bottleneck |

## Resources

- PR: Phase 2 Timeline & Composition
- Qt QGraphicsView optimization docs

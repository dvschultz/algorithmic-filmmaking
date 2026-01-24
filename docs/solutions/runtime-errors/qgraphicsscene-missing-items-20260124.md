---
date: 2026-01-24
problem_type: runtime_error
component: pyside6_graphics
symptoms:
  - "IndexError: list index out of range"
  - "Error when clicking Generate button"
  - "_track_items list is empty"
root_cause: missing_initialization
severity: high
tags: [pyside6, qt, qgraphicsscene, initialization]
---

# QGraphicsScene Track Items Not Initialized

## Problem Statement

Clicking the "Generate" button to create a shuffled sequence raised an `IndexError` because the scene's track items list was empty.

## Symptoms

```
Traceback (most recent call last):
  File "ui/timeline/timeline_widget.py", line 200, in _on_generate
    self.add_clip(clip, source, track_index=0, start_frame=current_frame)
  File "ui/timeline/timeline_widget.py", line 284, in add_clip
    self.scene.add_clip_to_track(...)
  File "ui/timeline/timeline_scene.py", line 160, in add_clip_to_track
    track_item = self._track_items[track_index]
IndexError: list index out of range
```

## Investigation

1. **Checked `_track_items`**: List was empty `[]`
2. **Traced initialization**: `_track_items` populated in `rebuild()` method
3. **Checked `_setup_scene()`**: Only called `_update_scene_rect()` and set background brush
4. **Found the issue**: `rebuild()` was never called during initialization

## Root Cause

`TimelineScene._setup_scene()` set up the scene dimensions and background but never called `rebuild()` to create the visual `TrackItem` instances. The default `Sequence` had one track in its data model, but no corresponding visual item was created.

```python
# _setup_scene() was missing rebuild() call
def _setup_scene(self):
    self._update_scene_rect()
    self.setBackgroundBrush(QBrush(QColor("#1e1e1e")))
    # Missing: self.rebuild()  <-- This creates TrackItem instances
```

## Solution

Added `rebuild()` call to `_setup_scene()`:

```python
def _setup_scene(self):
    """Initialize scene with default dimensions."""
    self.setBackgroundBrush(QBrush(QColor("#1e1e1e")))
    self.rebuild()  # Build track items for default sequence
```

The `rebuild()` method iterates through `self.sequence.tracks` and creates corresponding `TrackItem` visual instances, storing them in `self._track_items`.

## Prevention

**Pattern for QGraphicsScene initialization:**

1. **Always sync visuals with model**: After creating/setting data model, call the method that creates visual items
2. **Test initialization**: Verify lists/dicts that should contain items actually have items after `__init__`
3. **Document dependencies**: If method B requires method A to run first, document this or call A from B

**Initialization checklist for QGraphicsScene:**
- [ ] Data model created
- [ ] Visual items created for model elements
- [ ] Item collections populated (not empty lists)
- [ ] Scene rect updated

## Related

- Qt Model/View documentation on keeping views in sync with models
- `rebuild()` pattern common in Qt for data-driven graphics

## Files Changed

- `ui/timeline/timeline_scene.py` - Added `rebuild()` call in `_setup_scene()`

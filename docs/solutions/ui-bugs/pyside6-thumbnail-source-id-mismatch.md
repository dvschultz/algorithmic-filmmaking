---
title: "PySide6 Thumbnails Disappear Due to Source ID Mismatch Between Detection Worker and UI"
date: 2026-01-25
category: ui-bugs
tags:
  - pyside6
  - qt
  - thumbnails
  - source-id-mismatch
  - cached-property
  - worker-threads
  - state-management
symptoms:
  - Thumbnails appear briefly after scene detection completes
  - Thumbnails disappear when subsequent analysis (color detection) starts
  - No errors visible in logs
  - Callback appears to set clips correctly but UI shows empty state
module: ui/main_window.py
severity: medium
---

# PySide6 Thumbnails Disappear Due to Source ID Mismatch

## Problem

After scene detection completes, thumbnails appear briefly in the clip browser but then disappear when color detection or other analysis tools run. The UI shows an empty "Ready to Cut" state instead of the detected clips.

## Symptoms

1. Thumbnails appear briefly after scene detection
2. Thumbnails disappear when color detection starts
3. No errors in logs
4. The `_on_thumbnails_finished` callback runs but UI shows nothing
5. `clips_by_source.get(source_id)` returns empty list

## Root Cause

The scene detection worker creates a **NEW Source object** with a different UUID than the existing `self.current_source` in the UI:

```
UI State:           Source(id="abc-123", path="/video.mp4")
Detection Worker:   Source(id="xyz-789", path="/video.mp4")  # NEW object!
Clips created:      Clip(source_id="xyz-789", ...)           # Wrong ID!
```

When `_on_thumbnails_finished` calls `clips_by_source.get(self.current_source.id)`:
- It looks for clips with `source_id="abc-123"`
- But clips have `source_id="xyz-789"`
- Returns empty list
- `set_clips([])` is called, switching UI to "no clips" state

## Investigation Steps

1. Added logging to `_on_thumbnail_ready` - confirmed it was being called
2. Fixed `state_stack` not switching to show clips when first clip added
3. Thumbnails appeared but still disappeared after analysis started
4. Traced through `_on_thumbnails_finished` to find `set_clips` being called
5. Discovered `clips_by_source.get(self.current_source.id)` returning empty list
6. Found that detection creates NEW Source with different ID than existing source
7. Clips have `source_id` from detection's Source, not `self.current_source.id`

## Solution

### Primary Fix: Update Clip Source IDs (ui/main_window.py)

In `_on_detection_finished`, update each clip's `source_id` to reference the existing source before adding them to the project:

```python
def _on_detection_finished(self, source: Source, clips: list[Clip]):
    """Handle detection completion."""
    # ... existing code to update source metadata ...

    # Update clips to reference the existing source ID (detection creates a new Source object)
    # This ensures clips_by_source lookups work correctly
    logger.info(f"Updating {len(clips)} clips to use source_id={self.current_source.id}")
    for clip in clips:
        clip.source_id = self.current_source.id

    # Add new clips to the collection
    self.project.replace_source_clips(self.current_source.id, clips)
```

### Secondary Fix: Ensure UI State Transition (ui/tabs/cut_tab.py)

The `add_clip` method needed to switch to the clips view state when clips are added:

```python
def add_clip(self, clip, source):
    """Add a single clip to the browser (called during thumbnail generation)."""
    # Switch to clips state if not already showing clips
    if self.state_stack.currentIndex() != self.STATE_CLIPS:
        self.state_stack.setCurrentIndex(self.STATE_CLIPS)

    self.clip_browser.add_clip(clip, source)

    # Track clip and update count
    if clip not in self._clips:
        self._clips.append(clip)
        self.clip_count_label.setText(f"{len(self._clips)} clips")
```

## Prevention

### Code Review Checklist

- [ ] Does the worker create new model objects with auto-generated IDs?
- [ ] Are those objects already tracked in UI state with different IDs?
- [ ] Are IDs synchronized before storing/looking up?
- [ ] Are lookups validated (log warnings on unexpected empty results)?

### Pattern to Follow

```python
# When worker creates objects that correspond to existing UI objects:
def on_worker_complete(self, worker_objects: list[SomeModel]):
    # ALWAYS sync IDs to match existing UI state
    for obj in worker_objects:
        obj.parent_id = self.existing_parent.id

    # Then store
    self.storage.add(worker_objects)
```

### Warning Signs

- Lookups return empty when you expect data
- UI briefly shows correct state then reverts
- Same file/path but different object instances
- `@cached_property` lookups failing after data changes

## Related Patterns

- **Worker Thread Object Creation**: Workers often create new object instances rather than receiving shared references
- **UUID Identity Mismatches**: Auto-generated UUIDs cause silent failures when objects represent the same logical entity
- **Cached Property Invalidation**: `@cached_property` must be explicitly deleted when underlying data changes

## Files Modified

- `ui/main_window.py` - Added source_id sync in `_on_detection_finished`
- `ui/tabs/cut_tab.py` - Added state_stack switch in `add_clip`

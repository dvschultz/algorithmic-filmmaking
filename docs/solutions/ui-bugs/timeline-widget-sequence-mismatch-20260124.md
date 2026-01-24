---
date: 2026-01-24
problem_type: ui_bug
component: pyside6_widget
symptoms:
  - "Export Sequence button stays disabled despite clips on timeline"
  - "Timeline widget state doesn't match scene state"
root_cause: duplicate_state_objects
severity: medium
tags: [pyside6, qt, state-management, timeline]
---

# Export Button Disabled Despite Timeline Clips

## Problem Statement

After generating a shuffled sequence, the "Export Sequence" button remained disabled even though clips were clearly visible on the timeline.

## Symptoms

- Clicked "Generate" button successfully added clips to timeline
- Clips were visible with thumbnails
- Export Sequence button stayed grayed out/disabled
- No error messages

## Investigation

1. **Checked button enable logic**: `_update_export_button()` checks `any(track.clips for track in self.sequence.tracks)`
2. **Checked sequence state**: Found `self.sequence` was empty (no clips)
3. **Traced clip addition**: Clips were added via `self.scene.add_clip_to_track()` which adds to `self.scene.sequence`
4. **Found the issue**: Two separate `Sequence` objects existed

## Root Cause

`TimelineWidget` created its own `self.sequence = Sequence()` in `__init__`, while `TimelineScene` also created its own `self.sequence = Sequence()`. When clips were added through the scene, they went into the scene's sequence, but `_update_export_button()` was checking the widget's (empty) sequence.

```python
# TimelineWidget.__init__ - created its own sequence
self.sequence = Sequence()  # Widget's sequence - always empty

# TimelineScene.__init__ - created another sequence
self.sequence = Sequence()  # Scene's sequence - has the clips

# _update_export_button checked the wrong one
has_clips = any(track.clips for track in self.sequence.tracks)  # Widget's empty sequence
```

## Solution

Changed `TimelineWidget` to use the scene's sequence via a property instead of maintaining a separate object:

```python
# BEFORE (two sequences)
class TimelineWidget(QWidget):
    def __init__(self):
        self.sequence = Sequence()  # BAD: duplicate state
        self._setup_ui()

# AFTER (single source of truth)
class TimelineWidget(QWidget):
    def __init__(self):
        self._setup_ui()  # Creates self.scene

    @property
    def sequence(self) -> Sequence:
        """Get the sequence from the scene."""
        return self.scene.sequence  # GOOD: delegates to scene
```

## Prevention

**Pattern to follow for Qt parent/child state:**

1. **Single source of truth**: If a child component owns state, parent should access via the child
2. **Property delegation**: Use `@property` to delegate state access to the owner
3. **Audit state duplication**: When adding new state, grep for similar variables to catch duplicates

**Code review checklist:**
- [ ] Is this state duplicated elsewhere in the widget hierarchy?
- [ ] Which component should own this state?
- [ ] Are other components accessing their own copy instead of the owner's?

## Related

- Similar pattern issue documented in Qt docs about Model/View separation
- Timeline implementation: `ui/timeline/timeline_widget.py`, `ui/timeline/timeline_scene.py`

## Files Changed

- `ui/timeline/timeline_widget.py` - Removed `self.sequence`, added property

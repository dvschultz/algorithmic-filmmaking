---
name: pyside6-project-state-mutation
description: |
  Fix GUI not updating after programmatically adding data to a Project/model class
  in PySide6/PyQt applications. Use when: (1) data is added to project lists but
  doesn't appear in GUI, (2) logs show "clip or source not found" despite data
  existing, (3) cached_property lookups return stale data, (4) observer/signal
  callbacks aren't firing after state changes. Root cause: direct list manipulation
  bypasses cache invalidation and observer notifications.
author: Claude Code
version: 1.3.0
date: 2026-01-27
---

# PySide6 Project State Mutation Pattern

## Problem

In applications using a central Project/model class with cached properties and observer
patterns, directly modifying internal lists (e.g., `project.clips.append(clip)`) causes
data to be added but not reflected in the GUI. The data exists but cached lookups and
GUI observers don't see it.

## Context / Trigger Conditions

- Data added programmatically doesn't appear in GUI widgets
- Logs show warnings like "clip or source not found" when data clearly exists
- `cached_property` lookups (e.g., `clips_by_id`, `sources_by_id`) return stale data
- Observer callbacks / Qt signals aren't firing after adding data
- Code uses direct list manipulation: `project.clips.append()` or `project.sources.append()`
- GUI refresh after manual save/load shows the data was actually saved

## Solution

1. **Use dedicated mutation methods** instead of direct list access:

```python
# WRONG - bypasses cache invalidation and observer notification
project.sources.append(source)
project.clips.append(clip)

# CORRECT - invalidates caches and notifies observers
project.add_source(source)
project.add_clips([clip1, clip2])
```

2. **Ensure mutation methods follow this pattern**:

```python
class Project:
    @cached_property
    def clips_by_id(self) -> dict[str, Clip]:
        return {c.id: c for c in self._clips}

    def _invalidate_caches(self) -> None:
        """Clear cached properties when data changes."""
        for attr in ("sources_by_id", "clips_by_id", "clips_by_source"):
            self.__dict__.pop(attr, None)

    def _notify_observers(self, event: str, data: Any = None) -> None:
        """Notify all observers of state change."""
        for observer in self._observers:
            observer(event, data)

    def add_clips(self, clips: list[Clip]) -> None:
        """Add clips with proper cache/observer handling."""
        self._clips.extend(clips)
        self._invalidate_caches()  # Critical!
        self._dirty = True
        self._notify_observers("clips_added", clips)  # Critical!
```

3. **For tools/functions that modify project state**, always use the public methods:

```python
@tools.register(requires_project=True, modifies_gui_state=True)
def detect_scenes(project, video_path: str) -> dict:
    # ... detection logic ...

    # CORRECT: Use Project methods
    if not existing_source:
        project.add_source(source)
    project.add_clips(clips)

    return {"success": True, "clips_detected": len(clips)}
```

## Verification

After the fix:
1. Data appears immediately in GUI after operation completes
2. Cached property lookups return fresh data
3. No "not found" warnings in logs for newly added data
4. Observer callbacks fire (check with logging or breakpoints)

## Example

Before (broken):
```python
def detect_scenes(project, video_path):
    source, clips = detector.detect_scenes(video_path)
    project.sources.append(source)  # GUI doesn't know about this
    for clip in clips:
        project.clips.append(clip)  # Cached lookups still stale
    return {"success": True}
```

After (working):
```python
def detect_scenes(project, video_path):
    source, clips = detector.detect_scenes(video_path)
    project.add_source(source)  # Invalidates caches, notifies observers
    project.add_clips(clips)    # GUI updates automatically
    return {"success": True}
```

## Notes

- This pattern applies to any state management class with caching and observers
- Python's `@cached_property` stores results in `__dict__`, so clearing requires `pop()`
- The observer pattern is often connected to Qt signals via an adapter class
- Set `modifies_gui_state=True` in tool decorators when mutations affect visible state
- Path comparisons should use `.resolve()` to handle symlinks and relative paths consistently

## Common Pitfall: Disconnected Signal Handlers

Even when using proper mutation methods, GUI components may have their own lookup
dictionaries that become stale. Ensure ALL relevant signals are connected:

```python
# In MainWindow.__init__:
self._project_adapter = ProjectSignalAdapter(self.project, self)

# INCOMPLETE - only clips_updated connected
self._project_adapter.clips_updated.connect(self._on_clips_updated)

# COMPLETE - all state change signals connected
self._project_adapter.clips_updated.connect(self._on_clips_updated)
self._project_adapter.clips_added.connect(self._on_clips_added)   # Don't forget!
self._project_adapter.source_added.connect(self._on_source_added) # Don't forget!

# Handler must refresh any component that caches lookups:
@Slot(list)
def _on_clips_added(self, clips: list):
    # Refresh any tab/widget that has its own lookup cache
    self.analyze_tab.set_lookups(self.clips_by_id, self.sources_by_id)
```

Symptom of this bug: Data added via proper Project methods, but specific GUI
components show "not found" errors while others work correctly.

## Side Effects: Mirror GUI Operations

When GUI buttons trigger operations, they often have multiple side effects. When
adding programmatic APIs (like agent tools), ensure ALL side effects are replicated:

```python
# GUI "Detect" button does these side effects:
# 1. Runs scene detection
# 2. Adds clips to project
# 3. Generates thumbnails
# 4. Updates Cut tab (adds clips to ClipBrowser)
# 5. Sets current_source

# Agent tool must trigger same side effects via signal handler:
@Slot(list)
def _on_clips_added(self, clips: list):
    # 1. Refresh lookups
    self.analyze_tab.set_lookups(self.clips_by_id, self.sources_by_id)

    # 2. Set current_source if needed (before thumbnail generation!)
    clip_source = self.sources_by_id.get(clips[0].source_id)
    if clip_source and not self.current_source:
        self.current_source = clip_source

    # 3. Set source in Cut tab (prepares UI state)
    self.cut_tab.set_source(clip_source)

    # 4. Generate thumbnails - CRITICAL: route to correct handler!
    clips_needing_thumbnails = [c for c in clips if not c.thumbnail_path]
    if clips_needing_thumbnails:
        self.thumbnail_worker = ThumbnailWorker(...)
        # WRONG: _on_project_thumbnail_ready only updates existing thumbnails
        # self.thumbnail_worker.thumbnail_ready.connect(self._on_project_thumbnail_ready)
        # CORRECT: _on_thumbnail_ready calls cut_tab.add_clip() to add to ClipBrowser
        self.thumbnail_worker.thumbnail_ready.connect(self._on_thumbnail_ready)
        self.thumbnail_worker.start()
```

Symptom of missing side effects: Data appears in some UI components but not others,
or downstream operations fail silently (e.g., "0 success, 0 errors" in worker logs).

## Critical Pitfall: set_X() vs add_X() Methods

Many UI containers have two types of methods that sound similar but behave differently:

- `set_clips()` - Stores clips internally, updates labels/counts, but does NOT add to display widget
- `add_clip(clip, source)` - Actually adds the clip to the display widget (e.g., ClipBrowser)

```python
# WRONG - clips stored but not visible
self.cut_tab.set_clips(clips)  # Only updates internal state

# CORRECT - clips added to ClipBrowser and visible
self.cut_tab.add_clip(clip, source)  # Adds to _thumbnail_by_id, creates widget
```

In signal handler chains, the correct handler often calls `add_clip()`. Connecting to the
wrong handler (e.g., one that only updates existing items) causes "widget not found" errors.

Symptom: Logs show "Thumbnail widget not found! _thumbnail_by_id keys: []" - the container
dictionary is empty because `add_clip()` was never called.

## Related Patterns

- Observer pattern for state change notifications
- `@cached_property` cache invalidation
- Qt Model/View architecture
- Agent tool integration with GUI state

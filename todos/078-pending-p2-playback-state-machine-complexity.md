---
status: pending
priority: p2
issue_id: "078"
tags: [code-review, architecture, maintainability]
dependencies: []
---

# Playback State Machine Has 9 Pending Variables

## Problem Statement

The timeline playback refactoring in `ui/main_window.py` introduced 9 new instance variables for deferred playback state:

```python
self._preview_sync_clip = None
self._sequence_preview_source_id = None
self._sequence_preview_loading = False
self._pending_sequence_preview_source_id = None
self._pending_sequence_preview_clip_range = None
self._pending_sequence_preview_seek_seconds = None
self._pending_sequence_playback_source_id = None
self._pending_sequence_playback_range = None
self._syncing_timeline_from_video = False
```

These are cleared together in 3+ locations, creating a fragile state machine. A race condition between `_on_timeline_playhead_changed` and `_on_sequence_video_loaded` during rapid scrubbing could leave the player inconsistent.

## Findings

- **Architecture Strategist**: High risk #1
- **Code Simplicity Reviewer**: Finding #5
- **Performance Oracle**: UI responsiveness table notes tight coupling

## Proposed Solutions

### Option A: Group into a dataclass (Recommended)
```python
@dataclass
class _SequencePreviewState:
    sync_clip: Optional[SequenceClip] = None
    source_id: Optional[str] = None
    loading: bool = False
    pending_source_id: Optional[str] = None
    pending_clip_range: Optional[tuple] = None
    pending_seek_seconds: Optional[float] = None
    pending_playback_source_id: Optional[str] = None
    pending_playback_range: Optional[tuple] = None
    syncing_from_video: bool = False
```

Reset atomically with `self._preview_state = _SequencePreviewState()`.

- **Pros**: Atomic reset, self-documenting, testable
- **Cons**: Minor refactor across 3-4 methods
- **Effort**: Medium
- **Risk**: Low

### Option B: Extract SequencePlaybackController class
Full extraction of playback logic into a separate class.

- **Pros**: Testable in isolation, reduces MainWindow responsibility
- **Cons**: Larger refactor, may need to pass many references
- **Effort**: Large
- **Risk**: Medium

## Acceptance Criteria

- [ ] All 9 pending state variables grouped into a single struct or class
- [ ] Reset happens atomically in all locations
- [ ] Rapid timeline scrubbing does not leave player in inconsistent state

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | 3 agents flagged this independently |

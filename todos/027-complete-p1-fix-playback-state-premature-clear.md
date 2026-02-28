---
status: complete
priority: p1
issue_id: "027"
tags: [code-review, bug, video-player, gui-state]
dependencies: []
---

# Fix Premature Playback State Clear During Clip Transitions

## Problem Statement

`_on_video_state_changed` in `main_window.py` calls `gui_state.clear_playback_state()` when video stops, but this also fires during normal clip-to-clip transitions. The agent loses playback context mid-transition, seeing "Nothing playing" briefly before the next clip loads. This creates a race where agent decisions based on playback state may see stale/empty data.

## Findings

**Python Reviewer**: `_on_video_state_changed` clears playback state prematurely during clip transitions. The signal fires on every state change including transient stops between clips.

## Proposed Solutions

### Option A: Only Clear on Explicit Stop (Recommended)

Differentiate between "user/agent stopped playback" and "transitioning between clips" by checking if a new clip is queued:

```python
def _on_video_state_changed(self, is_playing):
    if is_playing:
        self.gui_state.update_playback_state(...)
    elif not self._next_clip_queued:
        self.gui_state.clear_playback_state()
```

**Pros:** Accurate state during transitions
**Cons:** Requires tracking transition state
**Effort:** Small
**Risk:** Low

### Option B: Debounce Clear with Short Timer

Delay the clear by 200ms — if a new clip starts in that window, cancel the clear.

**Pros:** Simple, no state tracking
**Cons:** 200ms of stale state possible
**Effort:** Small

## Acceptance Criteria

- [ ] Playback state persists during clip-to-clip transitions
- [ ] Playback state clears when user/agent explicitly stops
- [ ] Agent context shows correct clip during transitions
- [ ] Existing tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-27 | Created from branch review — Python Reviewer finding | Qt media state signals fire on transient states, not just user-initiated ones |

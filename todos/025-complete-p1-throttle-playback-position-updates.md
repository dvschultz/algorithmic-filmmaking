---
status: complete
priority: p1
issue_id: "025"
tags: [code-review, performance, video-player, threading]
dependencies: []
---

# Throttle Playback Position Updates to Prevent Main Thread Flood

## Problem Statement

The MPV video player fires position-changed callbacks at 24-60 Hz. Each callback emits a Qt signal that crosses threads, acquires a lock in `GUIState.update_playback_state()`, and triggers `to_context_string()` rebuilds. This floods the main thread with unnecessary work and can cause UI jank during playback.

## Findings

**Performance Oracle**: Position updates at video frame rate (24-60 Hz) cause:
- Thread-crossing signal emission per frame
- Lock acquisition in `GUIState` per frame
- Potential `to_context_string()` rebuild per frame (if agent is active)

**Python Reviewer**: `_on_video_state_changed` clears playback state prematurely during clip transitions, compounding the issue.

**Past Learning**: QThread signal duplicate delivery is a known issue in this codebase — high-frequency signals amplify it.

## Proposed Solutions

### Option A: Throttle at MpvSignalBridge Level (Recommended)

Add a timer-based throttle in `MpvSignalBridge` so position updates emit at most 4-5 Hz instead of every frame:

```python
def _on_position(self, value):
    now = time.monotonic()
    if now - self._last_position_emit < 0.2:  # 5 Hz max
        return
    self._last_position_emit = now
    self.position_changed.emit(value)
```

**Pros:** Fixes at the source, no downstream changes needed
**Cons:** Slightly less granular position display (200ms resolution)
**Effort:** Small
**Risk:** Low

### Option B: Debounce in GUIState

Keep full-rate signals but debounce the `update_playback_state()` lock acquisition using a dirty flag checked on a timer.

**Pros:** Full resolution available if needed
**Cons:** More complex, lock still acquired frequently
**Effort:** Medium

## Acceptance Criteria

- [ ] Position updates reach GUIState at <= 5 Hz during playback
- [ ] Video player UI still shows smooth playback (slider updates are separate)
- [ ] No lock contention warnings under normal playback
- [ ] Existing video player tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-27 | Created from branch review — Performance Oracle + Python Reviewer findings | High-frequency cross-thread signals are a known perf issue in this codebase |

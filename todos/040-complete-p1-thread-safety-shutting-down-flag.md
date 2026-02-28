---
status: complete
priority: p1
issue_id: "040"
tags: [code-review, thread-safety, video-player, mpv]
dependencies: []
---

# Thread Safety: `_shutting_down` Flag Not Atomic

## Problem Statement

The `_shutting_down` boolean is written from the main thread (`shutdown()`) and read from the MPV event thread (via property callbacks). Plain Python bools are not guaranteed atomic across threads. During shutdown, the MPV event thread could emit signals after cleanup.

## Findings

**Performance Oracle (Critical)**: Race condition during app exit — MPV event thread could fire after `_bridge.cleanup()` and `_mpv.terminate()`.

**Python Reviewer**: `_shutting_down` guard is consistently applied but the flag itself is not synchronized.

## Proposed Solutions

### Option A: threading.Event (Recommended)

Replace `_shutting_down: bool` with `threading.Event`:

```python
self._shutdown_event = threading.Event()

# In shutdown():
self._shutdown_event.set()

# In guards:
if self._shutdown_event.is_set():
    return
```

**Pros:** Properly synchronized, `Event.is_set()` is thread-safe
**Cons:** Slightly more verbose than bool check
**Effort:** Small
**Risk:** Low

## Technical Details

- **File:** `ui/video_player.py` lines 120, 344-355, 437-446
- **Component:** VideoPlayer

## Acceptance Criteria

- [ ] `_shutting_down` replaced with `threading.Event`
- [ ] All guard checks use `.is_set()`
- [ ] Shutdown sequence works correctly
- [ ] All tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-28 | Created from PR #61 review — Performance Oracle + Python Reviewer | Use threading primitives for cross-thread flags |

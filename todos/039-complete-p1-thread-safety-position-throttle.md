---
status: complete
priority: p1
issue_id: "039"
tags: [code-review, thread-safety, video-player, mpv]
dependencies: []
---

# Thread Safety: Data Race on `_last_position_emit` in MpvSignalBridge

## Problem Statement

`_last_position_emit` is read and written from the MPV event thread without synchronization. While CPython's GIL makes this mostly benign today, it is formally undefined behavior and will break on free-threaded Python 3.13+.

## Findings

**Python Reviewer (CRITICAL)**: The throttle timestamp in `on_time_pos` callback is a classic data race — read/write from MPV event thread with no lock.

**Performance Oracle**: Confirmed the `_last_position_emit` float is single-writer (MPV event thread only), but lack of synchronization is formally unsafe.

## Proposed Solutions

### Option A: threading.Lock (Recommended)

Add a `_throttle_lock` to MpvSignalBridge and wrap the read-compare-write in `on_time_pos`:

```python
def on_time_pos(_name, value):
    if value is not None:
        with self._throttle_lock:
            now = time.monotonic()
            if now - self._last_position_emit < self._POSITION_EMIT_INTERVAL:
                return
            self._last_position_emit = now
        self.position_changed.emit(value)
```

**Pros:** Correct, future-proof for free-threaded Python
**Cons:** Minor overhead (Lock acquisition at MPV callback rate)
**Effort:** Small
**Risk:** Low

## Technical Details

- **File:** `ui/video_player.py` lines 57-63
- **Component:** MpvSignalBridge

## Acceptance Criteria

- [ ] `_last_position_emit` reads/writes are protected by a lock
- [ ] Position throttling still works at 5 Hz
- [ ] All tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-28 | Created from PR #61 review — Python Reviewer + Performance Oracle | Thread safety matters even for single-writer fields |

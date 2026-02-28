---
status: complete
priority: p2
issue_id: "046"
tags: [code-review, performance, video-player, mpv]
dependencies: []
---

# Cache playback_speed to Avoid MPV IPC Reads at 5 Hz

## Problem Statement

Every position update (5 Hz) reads `self._mpv.speed` via IPC to the MPV process. The speed rarely changes, making this wasteful.

## Findings

**Performance Oracle**: `playback_speed` property reads MPV IPC every position update. Sub-millisecond per call, but architecturally wasteful.

## Proposed Solutions

### Option A: Cache Speed Value (Recommended)

Store `_cached_speed` in VideoPlayer, update in setter and `_on_speed_changed`:

```python
self._cached_speed: float = 1.0

@property
def playback_speed(self) -> float:
    return self._cached_speed

@playback_speed.setter
def playback_speed(self, speed: float):
    self._mpv.speed = speed
    self._cached_speed = speed
```

**Pros:** Eliminates IPC call, zero overhead
**Cons:** Cached value could drift (but speed changes are always through setter)
**Effort:** Small
**Risk:** Low

## Technical Details

- **File:** `ui/video_player.py` lines 465-477
- **File:** `ui/main_window.py` line 4292

## Acceptance Criteria

- [ ] `playback_speed` getter returns cached value
- [ ] Setter updates both MPV and cache
- [ ] Speed combo change callback updates cache
- [ ] All tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-28 | Created from PR #61 review — Performance Oracle | Cache values that rarely change but are read frequently |

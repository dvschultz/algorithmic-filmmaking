---
status: complete
priority: p2
issue_id: "049"
tags: [code-review, architecture, encapsulation, video-player]
dependencies: []
---

# set_playback_speed Tool Directly Manipulates speed_combo Widget

## Problem Statement

The `set_playback_speed` agent tool reaches through the VideoPlayer abstraction to directly manipulate `player.speed_combo.setCurrentIndex(idx)`. This violates encapsulation — the tool layer shouldn't know about internal widgets.

## Findings

**Architecture Strategist (Low)**: The `playback_speed` setter only updates MPV, not the combo box. The gap forces the tool to reach into widget internals.

## Proposed Solutions

### Option A: Add set_speed() Method to VideoPlayer (Recommended)

Add a `set_speed(speed: float)` method that updates both MPV and the combo box:

```python
def set_speed(self, speed: float):
    self.playback_speed = speed  # updates MPV
    idx = ... # find matching index
    self.speed_combo.setCurrentIndex(idx)
```

Tool then just calls `player.set_speed(speed)`.

**Pros:** Restores encapsulation, single API for speed changes
**Cons:** New method
**Effort:** Small
**Risk:** Low

## Technical Details

- **File:** `core/chat_tools.py` line 5424
- **File:** `ui/video_player.py` (add `set_speed()`)

## Acceptance Criteria

- [ ] `set_playback_speed` tool doesn't access `speed_combo` directly
- [ ] VideoPlayer has a `set_speed()` method that updates both MPV and UI
- [ ] Speed combo and MPV stay in sync

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-28 | Created from PR #61 review — Architecture Strategist | Tools should use public API, not reach into widget internals |

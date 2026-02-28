---
status: complete
priority: p1
issue_id: "042"
tags: [code-review, agent-native, context-injection, gui-state]
dependencies: []
---

# play_preview/pause_preview Don't Update gui_state

## Problem Statement

When the agent calls `play_preview`, `gui_state.playback_is_playing` remains `False`. When calling `pause_preview`, it stays wherever it was. This means subsequent LLM turns see stale playback state in the context window.

## Findings

**Agent-Native Reviewer (Critical)**: The new `stop_playback` correctly calls `gui_state.clear_playback_state()` and `set_playback_speed` correctly updates speed, but play/pause — the most fundamental tools — don't update state.

## Proposed Solutions

### Option A: Add gui_state Updates (Recommended)

In `play_preview`: add `gui_state.update_playback_state(is_playing=True)`
In `pause_preview`: add `gui_state.update_playback_state(is_playing=False)`

**Pros:** Consistent with stop_playback and set_playback_speed patterns
**Cons:** None
**Effort:** Small
**Risk:** Low

## Technical Details

- **File:** `core/chat_tools.py` lines 1614-1660
- **Component:** Agent tool functions

## Acceptance Criteria

- [ ] `play_preview` sets `gui_state.playback_is_playing = True`
- [ ] `pause_preview` sets `gui_state.playback_is_playing = False`
- [ ] Agent context string reflects actual play/pause state

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-28 | Created from PR #61 review — Agent-Native Reviewer | Every tool that changes observable state must update gui_state |

---
status: complete
priority: p2
issue_id: "048"
tags: [code-review, agent-native, context-injection, gui-state]
dependencies: []
---

# frame_step Tools Don't Update gui_state.is_playing

## Problem Statement

When the agent steps frame by frame, `gui_state.playback_is_playing` is not updated to `False`. Frame stepping implies paused state, but the context could show "playing" when the video is actually paused on a frame.

## Findings

**Agent-Native Reviewer (Warning)**: After frame stepping, stale `is_playing` state in agent context.

## Proposed Solutions

### Option A: Add gui_state Update (Recommended)

Add `gui_state.update_playback_state(is_playing=False)` after successful frame step in both `frame_step_forward` and `frame_step_backward`.

**Pros:** Accurate state for agent context
**Cons:** None
**Effort:** Small
**Risk:** Low

## Technical Details

- **File:** `core/chat_tools.py` lines 5356-5390
- **Component:** frame_step_forward, frame_step_backward tools

## Acceptance Criteria

- [ ] Frame stepping sets `gui_state.playback_is_playing = False`
- [ ] Agent context shows paused state after frame step

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-28 | Created from PR #61 review — Agent-Native Reviewer | Frame stepping changes play state — gui_state must reflect it |

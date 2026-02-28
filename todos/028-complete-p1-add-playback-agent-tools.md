---
status: complete
priority: p1
issue_id: "028"
tags: [code-review, agent-native, action-parity, video-player]
dependencies: []
---

# Add Missing Playback Agent Tools (stop, frame_step, speed)

## Problem Statement

The MPV video player has rich playback controls (stop, frame stepping, playback speed, A/B loop) but the agent has no tools to use them. This is a direct Action Parity violation — users can frame-step and adjust speed through the UI, but the agent cannot.

## Findings

**Agent-Native Reviewer**: 5 missing playback tools identified:
1. `stop_playback` — stop video playback
2. `frame_step_forward` — advance one frame
3. `frame_step_backward` — go back one frame
4. `set_playback_speed` — change playback rate
5. `set_ab_loop` — set A/B loop markers

These tools also need to be added to the system prompt for discovery.

## Proposed Solutions

### Option A: Add All 5 Tools (Recommended)

Add thin tool wrappers in `core/chat_tools.py` that call the video player's existing public API:

```python
def stop_playback(main_window=None):
    """Stop video playback."""
    main_window.video_player.stop()
    return {"success": True}

def frame_step_forward(main_window=None):
    """Advance video by one frame."""
    main_window.video_player.frame_step_forward()
    return {"success": True}
# ... etc
```

Add to system prompt in `ui/chat_worker.py` under a PLAYBACK CONTROLS section.

**Pros:** Full action parity for playback
**Cons:** 5 new tools adds to tool count
**Effort:** Small (tools are thin wrappers)
**Risk:** Low

### Option B: Single `control_playback(action, **params)` Tool

Consolidate into one tool with an `action` enum parameter.

**Pros:** Fewer tools
**Cons:** Less discoverable, more complex validation
**Effort:** Small

## Acceptance Criteria

- [ ] Agent can stop playback via tool
- [ ] Agent can frame-step forward and backward via tool
- [ ] Agent can set playback speed via tool
- [ ] Agent can set A/B loop markers via tool
- [ ] Tools appear in system prompt
- [ ] Existing tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-27 | Created from branch review — Agent-Native Reviewer finding | Every UI control needs a corresponding agent tool for action parity |

---
status: complete
priority: p3
issue_id: "038"
tags: [code-review, agent-native, context-injection, gui-state]
dependencies: ["028"]
---

# Track Playback Speed and A/B Loop State in GUIState

## Problem Statement

`GUIState` tracks basic playback state (position, playing, clip_id) but not playback speed or A/B loop markers. The agent can't see or reason about these settings in its context.

## Findings

**Agent-Native Reviewer**: Playback speed and A/B loop state not tracked in GUIState context. Agent has partial playback awareness.

## Proposed Solutions

### Option A: Add Fields to GUIState (Recommended)

Add to `GUIState`:
```python
playback_speed: float = 1.0
ab_loop_start_ms: Optional[int] = None
ab_loop_end_ms: Optional[int] = None
```

Update `to_context_string()` to include speed and loop info.

**Pros:** Complete playback context for agent
**Cons:** More fields to maintain
**Effort:** Small
**Risk:** Low

## Acceptance Criteria

- [ ] GUIState includes playback speed
- [ ] GUIState includes A/B loop markers when set
- [ ] Agent context string shows speed and loop info
- [ ] Fields updated when user/agent changes speed or loop

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-27 | Created from branch review — Agent-Native Reviewer finding | Context injection should cover all observable UI state |

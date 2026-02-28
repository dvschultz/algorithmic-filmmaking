---
status: complete
priority: p2
issue_id: "047"
tags: [code-review, agent-native, ui-integration, video-player]
dependencies: []
---

# set_ab_loop Tool Bypasses UI Label Update (Silent Action)

## Problem Statement

When the agent calls `set_ab_loop`, it sets MPV loop properties via `player.set_ab_loop()` but doesn't update `_ab_a_seconds`, `_ab_b_seconds`, or call `_update_ab_label()`. The A/B loop label stays blank even though the loop is active — a silent action anti-pattern.

## Findings

**Agent-Native Reviewer (Warning)**: The A/B loop label is only updated through button handlers `_on_set_a`/`_on_set_b`. Agent-set loops are invisible in the UI.

## Proposed Solutions

### Option A: Update VideoPlayer.set_ab_loop() to Also Set Label State (Recommended)

Modify `set_ab_loop(a, b)` in `video_player.py` to also set `_ab_a_seconds`, `_ab_b_seconds`, and call `_update_ab_label()`.

**Pros:** Any caller (UI or agent) gets consistent label update
**Cons:** None
**Effort:** Small
**Risk:** Low

## Technical Details

- **File:** `ui/video_player.py` `set_ab_loop()` method
- **File:** `core/chat_tools.py` `set_ab_loop` tool

## Acceptance Criteria

- [ ] Agent-triggered A/B loop shows markers in UI label
- [ ] Agent-triggered `clear_ab_loop` clears the label
- [ ] Button-triggered loops still work correctly

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-28 | Created from PR #61 review — Agent-Native Reviewer | All state changes must be reflected in UI regardless of trigger source |

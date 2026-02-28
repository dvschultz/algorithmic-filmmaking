---
status: complete
priority: p2
issue_id: "044"
tags: [code-review, architecture, agent-native, dry]
dependencies: []
---

# Standardize Player Access Pattern in Agent Tools

## Problem Statement

5 new playback tools use raw `getattr` chains while 3 existing tools use `main_window.get_video_player()`. This inconsistency creates maintenance risk — if the player location changes, only one pattern gets updated.

## Findings

**Agent-Native Reviewer (Critical)**: Two patterns coexist for the same purpose.
**Architecture Strategist (Medium)**: `get_video_player()` already exists at `main_window.py:2341`, duplicating its logic is unnecessary.
**Security Sentinel (Medium)**: Pattern B bypasses the public API, missing any future safety checks.
**Code Simplicity Reviewer**: Extract `_get_video_player` helper to DRY up the 5-instance pattern.

## Proposed Solutions

### Option A: Use Existing get_video_player() (Recommended)

Refactor `stop_playback`, `frame_step_forward`, `frame_step_backward`, `set_playback_speed`, `set_ab_loop` to use `main_window.get_video_player()`.

**Pros:** Consistent, single point of change, ~15 LOC saved
**Cons:** None
**Effort:** Small
**Risk:** Low

## Technical Details

- **File:** `core/chat_tools.py` lines 5338-5342, 5362-5366, 5383-5387, 5415-5419, 5450-5454
- **Component:** Agent playback tools

## Acceptance Criteria

- [ ] All 8 playback tools use the same player access pattern
- [ ] No raw `getattr` chains for player access in tool functions
- [ ] All tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-28 | Created from PR #61 review — 4 agents flagged this | When a public API exists, tools should use it |

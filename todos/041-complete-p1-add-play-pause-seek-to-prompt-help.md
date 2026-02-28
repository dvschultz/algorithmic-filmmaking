---
status: complete
priority: p1
issue_id: "041"
tags: [code-review, agent-native, capability-discovery, system-prompt]
dependencies: []
---

# Add play_preview/pause_preview/seek_to_time to System Prompt and /help

## Problem Statement

The VIDEO PLAYBACK section in the system prompt and /help lists `stop_playback`, `frame_step_forward/backward`, `set_playback_speed`, and `set_ab_loop`, but omits the three most fundamental tools: `play_preview`, `pause_preview`, and `seek_to_time`. The agent may not discover these tools.

## Findings

**Agent-Native Reviewer (Critical)**: Play and pause are the most fundamental playback actions. An agent that doesn't know it can play/pause/seek is missing basic video control.

## Proposed Solutions

### Option A: Add to Existing Section (Recommended)

Update the VIDEO PLAYBACK section in `chat_worker.py` and `chat_panel.py` to include all 8 playback tools:

```
- play_preview(clip_id) — Start playing a clip
- pause_preview() — Pause playback
- seek_to_time(seconds) — Seek to a specific position
- stop_playback() — Stop playback and reset
- frame_step_forward/backward() — Step one frame
- set_playback_speed(speed) — Change playback speed
- set_ab_loop(a_seconds, b_seconds) — Set loop markers
```

**Pros:** Complete documentation, agent discovers all tools
**Cons:** None
**Effort:** Small
**Risk:** Low

## Technical Details

- **Files:** `ui/chat_worker.py` lines 638-643, `ui/chat_panel.py` lines 316-320
- **Component:** System prompt, /help command

## Acceptance Criteria

- [ ] System prompt VIDEO PLAYBACK section lists all 8 tools
- [ ] /help Video Playback section lists all 8 tools
- [ ] Tool names and parameter signatures are accurate

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-28 | Created from PR #61 review — Agent-Native Reviewer | Always document all tools in the prompt, especially fundamental ones |

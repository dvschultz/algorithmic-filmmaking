---
status: complete
priority: p2
issue_id: "030"
tags: [code-review, agent-native, capability-discovery, system-prompt]
dependencies: ["028"]
---

# Add New Tools to System Prompt and /help Command

## Problem Statement

Several new tools (`update_source`, `update_sequence_clip`, `start_clip_analysis`, playback controls) were added but not mentioned in the system prompt or the `/help` command. The agent cannot discover or use tools it doesn't know exist.

## Findings

**Agent-Native Reviewer**:
- `update_source` and `update_sequence_clip` not in system prompt
- Playback tools absent from system prompt
- `/help` command missing playback, update, and detail capabilities

**Simplicity Reviewer**: `/help` string is hardcoded — easy to miss when adding new tools.

## Proposed Solutions

### Option A: Update System Prompt + /help (Recommended)

1. Add sections to system prompt in `chat_worker.py`:
   - PLAYBACK CONTROLS: stop, frame step, speed, A/B loop
   - DATA UPDATES: update_source, update_sequence_clip
   - Mention `start_clip_analysis` as preferred over individual analysis tools

2. Update `/help` in `chat_panel.py` to include new capability groups

**Pros:** Complete discovery for all new tools
**Cons:** System prompt grows longer
**Effort:** Small
**Risk:** Low

## Acceptance Criteria

- [ ] System prompt mentions all new tools with usage guidance
- [ ] `/help` shows playback, update, and unified analysis capabilities
- [ ] Agent can discover and use new tools without user prompting

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-27 | Created from branch review — Agent-Native Reviewer + Simplicity findings | Tools without system prompt mention are invisible to the agent |

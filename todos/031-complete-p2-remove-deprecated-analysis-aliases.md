---
status: complete
priority: p2
issue_id: "031"
tags: [code-review, simplicity, dead-code, chat-tools]
dependencies: []
---

# Remove 6 Deprecated Analysis Tool Aliases (~200 LOC)

## Problem Statement

After consolidating 7 analysis tools into `start_clip_analysis`, the old tool names were kept as aliases for backward compatibility. These ~200 lines serve no purpose since the system prompt already directs the agent to use the unified tool.

## Findings

**Simplicity Reviewer**: 6 deprecated aliases (`analyze_colors_live`, `analyze_shots_live`, `transcribe_live`, `classify_content_live`, `detect_objects_live`, `count_people_live`) are thin wrappers that just call `start_clip_analysis`. ~200 LOC removable.

## Proposed Solutions

### Option A: Remove All Aliases (Recommended)

Delete the 6 alias functions from `chat_tools.py` and their registrations in `ToolExecutor`.

**Pros:** 200 fewer lines, simpler codebase
**Cons:** If any saved prompts reference old names, they'll break (unlikely — agent uses system prompt)
**Effort:** Small
**Risk:** Low

### Option B: Keep Aliases but Mark Deprecated

Add deprecation warnings to aliases, remove in next release.

**Pros:** Safer transition
**Cons:** More code to maintain
**Effort:** Small

## Acceptance Criteria

- [ ] Old alias tool names removed from chat_tools.py
- [ ] Tool registrations updated
- [ ] System prompt only references start_clip_analysis
- [ ] All tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-27 | Created from branch review — Simplicity Reviewer finding | Backward-compat aliases for agent tools are unnecessary when system prompt controls discovery |

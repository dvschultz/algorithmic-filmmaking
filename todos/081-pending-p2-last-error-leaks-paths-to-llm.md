---
status: pending
priority: p2
issue_id: "081"
tags: [code-review, security, privacy]
dependencies: []
---

# gui_state.last_error Exposes Internal File Paths to LLM Agent

## Problem Statement

Error messages from internal operations (transcription, analysis, scene detection) are stored in `gui_state.last_error` and included verbatim in the LLM agent's context via `to_context_string()`. These messages may contain file paths, system details, or other internal information that gets sent to the LLM provider.

Example: `"Detection error: ffprobe -v error -show_format /Users/derrick/private/sensitive_video.mp4"`

## Findings

- **Security Sentinel**: Medium finding #3
- Location: `core/gui_state.py` lines 375-381, `ui/main_window.py` multiple `set_last_error` call sites

## Proposed Solutions

### Option A: Sanitize error messages before storing
Strip file paths and system details in `set_last_error`, keeping only the error classification.

- **Pros**: Prevents information leakage
- **Cons**: Reduces debugging info available to agent
- **Effort**: Small
- **Risk**: Low

### Option B: Separate user-facing and internal error messages
Store both a sanitized message (for LLM context) and a detailed message (for logs only).

- **Pros**: Best of both worlds
- **Cons**: More complex API
- **Effort**: Medium
- **Risk**: Low

## Acceptance Criteria

- [ ] LLM context string does not contain absolute file paths
- [ ] Error classification is still available to the agent for helpful responses

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | Security sentinel flagged as Medium |

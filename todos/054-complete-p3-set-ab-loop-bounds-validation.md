---
status: complete
priority: p3
issue_id: "054"
tags: [code-review, security, validation, agent-tools]
dependencies: []
---

# Add Bounds Validation to set_ab_loop Tool

## Problem Statement

The `set_ab_loop` tool validates `b > a` but doesn't check for negative values or values exceeding the video duration. Compare with `seek_to_time` which validates both.

## Findings

**Security Sentinel (Low)**: MPV handles out-of-range values gracefully, but early validation with meaningful error messages is better for the LLM agent.

## Proposed Solutions

### Option A: Add Bounds Checks (Recommended)

```python
if a_seconds < 0:
    return {"success": False, "error": "a_seconds cannot be negative"}
duration_ms = player.duration_ms
if duration_ms > 0:
    duration_s = duration_ms / 1000.0
    if b_seconds > duration_s:
        return {"success": False, "error": f"b_seconds exceeds video duration ({duration_s:.2f}s)"}
```

**Pros:** Better error messages for LLM, consistent with seek_to_time
**Cons:** None
**Effort:** Small
**Risk:** Low

## Technical Details

- **File:** `core/chat_tools.py` lines 5440-5480

## Acceptance Criteria

- [ ] Negative values rejected with clear error
- [ ] Values beyond duration rejected with duration info
- [ ] Consistent with seek_to_time validation pattern

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-28 | Created from PR #61 review — Security Sentinel | Validate early with meaningful errors for LLM agents |

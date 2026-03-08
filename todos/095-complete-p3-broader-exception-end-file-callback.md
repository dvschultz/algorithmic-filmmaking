---
status: pending
priority: p3
issue_id: "095"
tags: [code-review, reliability]
dependencies: []
---

# Broader Exception Handling in end-file Callback

## Problem Statement

`on_end_file` callback in `ui/video_player.py` lines 506-516 catches ValueError and AttributeError but ctypes struct access can also raise RuntimeError during shutdown.

## Findings

- **Code Reviewer**: P3 reliability issue

## Proposed Solutions

### Option A: Broaden to except Exception
Since this is a best-effort handler, broaden to `except Exception` to prevent unhandled exceptions during shutdown.

- **Effort**: Small
- **Risk**: Low

## Acceptance Criteria

- [ ] No unhandled exceptions from end-file callback during shutdown

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | Narrow exception handling misses RuntimeError |

---
status: pending
priority: p3
issue_id: "097"
tags: [code-review, performance]
dependencies: []
---

# Sequential prerender_batch Misses Parallelism Opportunity

## Problem Statement

`prerender_batch` in `core/remix/prerender.py` lines 137-190 processes clips sequentially. Modern machines can run 2-4 FFmpeg processes concurrently. For 50 clips this means 100-250s instead of 25-60s.

## Findings

- **Code Reviewer**: P3 performance issue

## Proposed Solutions

### Option A: Use ThreadPoolExecutor
Use `ThreadPoolExecutor` with configurable worker count (default 2-4, capped by `cpu_count`). The existing `cancel_event` pattern already works across threads.

- **Effort**: Medium
- **Risk**: Low

## Acceptance Criteria

- [ ] Pre-rendering uses multiple concurrent FFmpeg processes

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | Sequential processing leaves CPU underutilized |

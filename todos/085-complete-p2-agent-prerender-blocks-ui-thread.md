---
status: pending
priority: p2
issue_id: "085"
tags: [code-review, performance, architecture]
dependencies: []
---

# Agent prerender_batch Blocks UI Thread

## Problem Statement

`generate_and_apply()` in `ui/tabs/sequence_tab.py` lines 1548-1575 calls `prerender_batch` synchronously. The DiceRollDialog correctly uses DiceRollWorker (CancellableWorker/QThread), but the agent/tool API path runs FFmpeg inline on whatever thread calls it. This freezes the UI when the chat agent triggers shuffle with transforms.

## Findings

- Location: `ui/tabs/sequence_tab.py` lines 1548-1575
- The dialog path correctly offloads to a background worker
- The agent path bypasses the worker and calls FFmpeg directly

## Proposed Solutions

### Option A: Route agent path through same DiceRollWorker pattern
Reuse the existing DiceRollWorker (CancellableWorker/QThread) for the agent code path.

- **Pros**: Reuses existing code, gets cancel+progress for free
- **Cons**: Requires async signal pattern in tool executor
- **Effort**: Medium
- **Risk**: Low

### Option B: Wrap prerender_batch call in a simple QThread worker with blocking wait
Create a minimal QThread wrapper that runs prerender_batch and blocks the calling thread until done.

- **Pros**: Simpler, keeps synchronous return semantics
- **Cons**: No progress/cancel feedback for agent
- **Effort**: Small
- **Risk**: Low

## Acceptance Criteria

- [ ] Agent can trigger shuffle with transforms without UI freeze
- [ ] Pre-rendering runs on a background thread when invoked via agent

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | Dialog path uses DiceRollWorker but agent path runs FFmpeg inline |

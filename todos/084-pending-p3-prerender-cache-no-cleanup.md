---
status: pending
priority: p3
issue_id: "084"
tags: [code-review, reliability]
dependencies: []
---

# Pre-Rendered Clip Cache Has No Size Bounds or Cleanup

## Problem Statement

Pre-rendered clips are stored in `settings.cache_dir / "transformed_clips"` with idempotency (skip if exists) but no maximum cache size limit, no LRU eviction, and no cleanup mechanism. Each pre-rendered clip is a full video segment (potentially hundreds of MB). Repeated use with different transforms accumulates files indefinitely.

## Findings

- **Security Sentinel**: Low finding #5
- Location: `core/remix/prerender.py`, `ui/dialogs/dice_roll_dialog.py` line 37

## Proposed Solutions

### Option A: Clean up when generating new sequence
Delete the `transformed_clips/` directory contents before each new pre-render batch.

- **Pros**: Simple, prevents accumulation
- **Cons**: Loses cached results from previous generations
- **Effort**: Small
- **Risk**: Low

### Option B: LRU cache with size limit
Track cache usage, evict oldest entries when total exceeds a configurable limit.

- **Pros**: Preserves useful cache, bounded disk usage
- **Cons**: More complex implementation
- **Effort**: Medium
- **Risk**: Low

## Acceptance Criteria

- [ ] Pre-rendered clip cache does not grow unbounded
- [ ] Disk usage from cached transforms is managed

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | Security sentinel flagged as Low |

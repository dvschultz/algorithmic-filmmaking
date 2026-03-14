---
status: pending
priority: p3
issue_id: "091"
tags: [code-review, architecture]
dependencies: []
---

# _copy_prerendered_clips Mutates State as Side Effect of Save

## Problem Statement

`_copy_prerendered_clips` in `core/project.py` lines 74-98 modifies `clip.prerendered_path` in-place during save, permanently changing in-memory paths from cache to project folder. Surprising side effect; partial failure leaves inconsistent state.

## Findings

- **Code Reviewer**: P3 architecture issue

## Proposed Solutions

### Option A: Return path mapping dict
Return path mapping dict, apply during `to_dict` serialization only. In-memory state remains unchanged.

- **Effort**: Medium
- **Risk**: Low

### Option B: Deep copy sequence before mutation
Deep copy the sequence before modifying paths so the original is preserved.

- **Effort**: Small
- **Risk**: Low

## Acceptance Criteria

- [ ] Save does not permanently alter in-memory SequenceClip paths

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | Save has surprising mutation side effect |

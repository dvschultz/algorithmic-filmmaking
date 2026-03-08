---
status: pending
priority: p3
issue_id: "093"
tags: [code-review, reliability]
dependencies: []
---

# Filename Collision Risk in _copy_prerendered_clips

## Problem Statement

`_copy_prerendered_clips` uses `src.name` for destination. While clip_id+transform encoding makes collisions unlikely, the code doesn't check for content mismatch before overwriting.

## Findings

- **Code Reviewer**: P3 reliability issue

## Proposed Solutions

### Option A: Content-aware collision check
Add content-aware collision check or include SequenceClip ID in destination name.

- **Effort**: Small
- **Risk**: Low

## Acceptance Criteria

- [ ] Different pre-rendered files with same filename are not silently overwritten

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | Potential silent overwrite on name collision |

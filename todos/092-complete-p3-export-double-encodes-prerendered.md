---
status: pending
priority: p3
issue_id: "092"
tags: [code-review, performance]
dependencies: []
---

# Export Double-Encodes Pre-Rendered Clips

## Problem Statement

`_export_prerendered_segment` in `core/sequence_export.py` re-encodes pre-rendered files (already CRF 18) through another encode pass, causing generation loss and wasted CPU time.

## Findings

- **Code Reviewer**: P3 performance issue

## Proposed Solutions

### Option A: Stream copy when possible
Use stream copy (`-c:v copy -c:a copy`) when no scale/chromatic bar is needed. Fall back to re-encode only when filters are required.

- **Effort**: Medium
- **Risk**: Low

## Acceptance Criteria

- [ ] Pre-rendered clips without scale/bar export via stream copy

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | Unnecessary re-encode of already-encoded clips |

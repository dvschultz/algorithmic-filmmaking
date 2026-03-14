---
status: pending
priority: p3
issue_id: "089"
tags: [code-review, reliability]
dependencies: []
---

# Prerender Idempotency Trusts File Existence Without Size Check

## Problem Statement

`prerender_clip` in `core/remix/prerender.py` lines 57-59 returns early if the output file exists, but doesn't check file size. An interrupted FFmpeg run could leave a 0-byte or truncated file that would be returned as valid.

## Findings

- **Code Reviewer**: P3 reliability issue

## Proposed Solutions

### Option A: Add file size check
Add `output_path.stat().st_size > 0` check before returning the cached path.

- **Effort**: Small
- **Risk**: Low

### Option B: Use atomic write pattern
Write to a temp file, rename on success. Incomplete files never appear at the final path.

- **Effort**: Medium
- **Risk**: Low

## Acceptance Criteria

- [ ] Truncated pre-render files are re-rendered

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | Early return trusts existence without validation |

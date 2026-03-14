---
status: pending
priority: p2
issue_id: "077"
tags: [code-review, quality, duplication]
dependencies: []
---

# Duplicated _REVERSE_MAX_DURATION Constant

## Problem Statement

The 15-second reverse safety limit is defined independently in two files:
- `core/remix/prerender.py` line 19: `_REVERSE_MAX_DURATION = 15.0`
- `core/sequence_export.py` line 172: `_REVERSE_MAX_DURATION = 15.0`

These will inevitably drift out of sync if one is updated without the other.

## Findings

- **Python Reviewer**: High-priority issue #4
- **Architecture Strategist**: Low risk #6
- Introduced by the Dice Roll pre-render feature (this session)

## Proposed Solutions

### Option A: Extract to shared constant
Move to `core/constants.py` or define once in `core/remix/prerender.py` and import in `sequence_export.py`.

- **Pros**: Single source of truth
- **Cons**: Minor import addition
- **Effort**: Small
- **Risk**: Low

## Acceptance Criteria

- [ ] `_REVERSE_MAX_DURATION` defined in exactly one location
- [ ] Both prerender.py and sequence_export.py reference the same value

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | New duplication introduced by prerender.py |

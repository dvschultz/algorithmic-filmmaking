---
status: pending
priority: p3
issue_id: "094"
tags: [code-review, agent-parity]
dependencies: []
---

# update_sequence_clip Cannot Set Transform Flags

## Problem Statement

`update_sequence_clip` in `core/chat_tools.py` lines 897-986 has no hflip/vflip/reverse parameters. Agent cannot modify transforms on individual clips after generation.

## Findings

- **Code Reviewer**: P3 agent-parity issue

## Proposed Solutions

### Option A: Add transform parameters
Add optional hflip, vflip, reverse params with pre-render trigger.

- **Effort**: Medium
- **Risk**: Low

## Acceptance Criteria

- [ ] Agent can toggle transforms on individual sequence clips

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | Agent lacks parity with GUI for transforms |

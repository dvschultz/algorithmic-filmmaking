---
status: pending
priority: p2
issue_id: "087"
tags: [code-review, agent-parity]
dependencies: []
---

# get_sequence_state Omits Transform Fields

## Problem Statement

`get_sequence_state` in `core/chat_tools.py` lines 1138-1150 does not include `hflip`, `vflip`, `reverse`, or `prerendered_path` in the clip data dict. After the agent runs `generate_remix` with transforms, it cannot verify which clips got transforms or whether pre-rendering succeeded. The agent is blind to the state it just created.

## Findings

- Location: `core/chat_tools.py` lines 1138-1150
- The four missing fields: `hflip`, `vflip`, `reverse`, `prerendered_path`
- Agent has no way to confirm transform application or pre-render success

## Proposed Solutions

### Option A: Add all 4 fields to the clip dict in get_sequence_state
Include `hflip`, `vflip`, `reverse`, and `prerendered_path` in the returned clip data.

- **Pros**: Complete visibility for the agent, trivial change
- **Cons**: Slightly larger response payload
- **Effort**: Small
- **Risk**: None

## Acceptance Criteria

- [ ] After `generate_remix` with `random_hflip=True`, `get_sequence_state` returns `hflip`/`vflip`/`reverse`/`prerendered_path` per clip
- [ ] Agent can verify which clips received transforms

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | Agent cannot verify transform state it creates |

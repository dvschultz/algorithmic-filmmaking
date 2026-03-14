---
status: pending
priority: p2
issue_id: "080"
tags: [code-review, architecture, maintainability]
dependencies: []
---

# Dialog Routing If-Chains Duplicated in sequence_tab.py

## Problem Statement

Two nearly identical if-chains in `ui/tabs/sequence_tab.py` route dialog-based algorithms:
- `_on_card_clicked()` lines ~511-532 (card click path)
- `_on_confirm_generate()` lines ~407-427 (confirm generate path)

At 6 dialog algorithms and growing, each new algorithm requires updates in both places. This is a maintenance risk — forgetting one path causes silent failure.

## Findings

- **Architecture Strategist**: Medium risk #3
- **Code Simplicity Reviewer**: Related to finding #2

## Proposed Solutions

### Option A: Dispatch table in ALGORITHM_CONFIG (Recommended)
Extend `ALGORITHM_CONFIG` with a `"dialog_factory"` key that maps to the dialog class or a `_show_<name>_dialog` method reference.

```python
# In algorithm_config.py
"rose_hobart": {..., "is_dialog": True, "dialog_key": "rose_hobart"},
```

Then in sequence_tab:
```python
_DIALOG_DISPATCH = {
    "rose_hobart": "_show_rose_hobart_dialog",
    "shuffle": "_show_dice_roll_dialog",
    ...
}
```

- **Pros**: Single routing table, easy to add new algorithms
- **Cons**: Indirection, slightly harder to trace
- **Effort**: Small
- **Risk**: Low

## Acceptance Criteria

- [ ] Algorithm-to-dialog routing defined in one place
- [ ] Both entry points (card click, confirm generate) use the same dispatch
- [ ] Adding a new dialog algorithm requires change in one location

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | 2 agents flagged independently |

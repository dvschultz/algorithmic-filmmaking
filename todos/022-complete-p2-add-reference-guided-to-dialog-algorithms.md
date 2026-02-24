---
status: complete
priority: p2
issue_id: "022"
tags: [code-review, agent-native, bug]
dependencies: []
---

# Add reference_guided to dialog_algorithms Tuple

## Problem Statement

The `dialog_algorithms` tuple in `set_sorting_algorithm()` at `sequence_tab.py:1127` does not include `"reference_guided"`. When the agent calls `set_sorting_algorithm("reference_guided")` while in timeline state, the code falls into the dropdown-based regeneration path instead of re-opening the dialog. Since `reference_guided` is not in the header dropdown, this silently fails or produces unexpected behavior.

## Findings

**Location:** `ui/tabs/sequence_tab.py` line 1127

```python
dialog_algorithms = ("exquisite_corpus", "storyteller")
```

Should be:

```python
dialog_algorithms = ("exquisite_corpus", "storyteller", "reference_guided")
```

All three algorithms bypass the `generate_sequence()` dispatcher and use their own configuration dialogs.

## Proposed Solutions

### Option A: Add to Tuple (Recommended)

One-line fix:

```python
dialog_algorithms = ("exquisite_corpus", "storyteller", "reference_guided")
```

**Pros:** Trivial fix, correct behavior
**Cons:** None
**Effort:** Small
**Risk:** None

## Acceptance Criteria

- [ ] `set_sorting_algorithm("reference_guided")` re-opens the dialog instead of using dropdown
- [ ] Existing tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-24 | Created from PR #58 architecture review | Dialog-based algorithms must be registered in both `ALGORITHM_CONFIG` and the `dialog_algorithms` tuple |

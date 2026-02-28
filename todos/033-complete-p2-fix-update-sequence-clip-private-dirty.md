---
status: complete
priority: p2
issue_id: "033"
tags: [code-review, code-quality, encapsulation, chat-tools]
dependencies: []
---

# Fix update_sequence_clip Accessing project._dirty Directly

## Problem Statement

The `update_sequence_clip` tool in `chat_tools.py` sets `project._dirty = True` directly instead of using a Project method. This breaks encapsulation — if the dirty-tracking mechanism changes, this tool silently stops working.

## Findings

**Python Reviewer**: `update_sequence_clip` accesses `project._dirty` directly (private attribute). Should use a Project method to mark state as changed.

## Proposed Solutions

### Option A: Add project.mark_dirty() Method (Recommended)

```python
# In project.py
def mark_dirty(self):
    """Mark project as having unsaved changes."""
    self._dirty = True

# In chat_tools.py
project.mark_dirty()
```

Or better: have the existing `_notify_observers("sequence_changed")` call also set `_dirty`.

**Pros:** Proper encapsulation, single responsibility
**Cons:** Minor method addition
**Effort:** Small
**Risk:** Low

## Acceptance Criteria

- [ ] `update_sequence_clip` does not access `project._dirty` directly
- [ ] Project state correctly marked dirty after sequence clip updates
- [ ] All tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-27 | Created from branch review — Python Reviewer finding | Private attributes should stay private, use public methods |

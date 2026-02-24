---
status: complete
priority: p3
issue_id: "021"
tags: [code-review, dead-code, simplicity]
dependencies: []
---

# Remove Unused project Parameter from ReferenceGuideDialog

## Problem Statement

The `project` parameter in `ReferenceGuideDialog.__init__` is always passed as `None` and never meaningfully used. It is stored as `self.project` but no method in the dialog reads it. This is dead code that misleads readers into thinking the dialog depends on the project object.

## Findings

**Location:** `ui/dialogs/reference_guide_dialog.py` lines 142-160

The constructor accepts `project`:

```python
def __init__(
    self,
    clips: list,
    sources_by_id: dict,
    project: Any,
    parent=None,
):
```

It is stored at line 160:

```python
self.project = project
```

But `self.project` is never referenced anywhere else in the file. A grep for `self.project` in the dialog file returns only the assignment at line 160.

**Caller:** `ui/tabs/sequence_tab.py` lines 788-792

The dialog is always instantiated with `project=None`:

```python
dialog = ReferenceGuideDialog(
    clips=clips,
    sources_by_id=sources_by_id,
    project=None,
    parent=self,
)
```

## Proposed Solutions

### Option A: Remove the Parameter (Recommended)
Delete `project: Any` from the constructor signature, remove `self.project = project`, and remove `project=None` from the caller.

**Pros:** Eliminates dead code, removes misleading `Any` import dependency
**Cons:** Breaking change if any external code passes `project` (none found)
**Effort:** Small

### Option B: Keep as Reserved for Future Use
Add a comment explaining it is reserved.

**Pros:** No API change
**Cons:** Dead code remains, YAGNI violation
**Effort:** None

## Acceptance Criteria

- [ ] `project` parameter removed from `ReferenceGuideDialog.__init__`
- [ ] `self.project` assignment removed
- [ ] Caller in `sequence_tab.py` no longer passes `project=None`
- [ ] `Any` import removed from dialog if no longer needed
- [ ] Existing tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-24 | Created from PR #58 code review | Remove unused parameters early to prevent misleading API surfaces |

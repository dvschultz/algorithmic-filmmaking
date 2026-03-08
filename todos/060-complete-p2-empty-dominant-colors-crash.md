---
status: complete
priority: p2
issue_id: "060"
tags: [code-review, quality, rose-hobart]
dependencies: []
---

# Empty dominant_colors List Crash in Color Sort

## Problem Statement

The color sort lambda in both the chat tool and dialog uses `x[0].dominant_colors[0]` with a guard `if x[0].dominant_colors`. An empty list `[]` is truthy but will raise `IndexError` on `[0]`. The `dominant_colors` field is `Optional[list]` and could be `[]`.

## Findings

**Python Reviewer (Critical)**: Latent crash that will hit real users. Both the dialog's `_order_clips` (line 177) and the chat tool (line ~3993) have the same pattern.

## Proposed Solutions

### Option A: Guard with Length Check (Recommended)

```python
if x[0].dominant_colors and len(x[0].dominant_colors) > 0
```

**Pros:** One-line fix per location
**Cons:** None
**Effort:** Small
**Risk:** Low

## Technical Details

- **Files:** `ui/dialogs/rose_hobart_dialog.py` line 177, `core/chat_tools.py` line ~3993

## Acceptance Criteria

- [ ] Color sort handles None, empty list [], and populated lists
- [ ] No IndexError when dominant_colors is []

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Python Reviewer | Empty lists are truthy in Python — always check len() before indexing |

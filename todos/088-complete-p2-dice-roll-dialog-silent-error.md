---
status: pending
priority: p2
issue_id: "088"
tags: [code-review, quality]
dependencies: []
---

# DiceRollDialog _on_error Gives No User Feedback

## Problem Statement

`_on_error` in `ui/dialogs/dice_roll_dialog.py` lines 322-325 logs the error and flips back to the config page silently. No QMessageBox, no error label — the user sees progress disappear with no explanation. Every other dialog (Storyteller, Exquisite Corpus) shows visible error feedback.

## Findings

- Location: `ui/dialogs/dice_roll_dialog.py` lines 322-325
- Error is logged but not surfaced to the user
- Other dialogs (Storyteller, Exquisite Corpus) show QMessageBox on error

## Proposed Solutions

### Option A: Show QMessageBox.critical with error message
Display a modal error dialog with the error message, consistent with other dialogs.

- **Pros**: Consistent with other dialogs, immediate and clear feedback
- **Cons**: Modal dialog on top of modal dialog (slightly awkward)
- **Effort**: Small
- **Risk**: None

### Option B: Show inline error label on config page
Add an error label widget to the config page that displays when an error occurs.

- **Pros**: Less disruptive UX, no stacked modals
- **Cons**: Need to add a label widget and manage its visibility
- **Effort**: Small
- **Risk**: None

## Acceptance Criteria

- [ ] When pre-rendering fails, user sees a clear error message
- [ ] Error feedback is consistent with other dialog error handling

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | Error handler silently swallows failures |

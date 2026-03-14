---
status: pending
priority: p3
issue_id: "082"
tags: [code-review, quality, ui-consistency]
dependencies: []
---

# DiceRollDialog Doesn't Use _apply_theme Pattern

## Problem Statement

`RoseHobartDialog` properly implements `_apply_theme()` and connects to `theme().changed` for dynamic theme updates. `DiceRollDialog` uses inline `setStyleSheet` calls with direct `theme()` references that will not update when the theme changes at runtime.

## Findings

- **Python Reviewer**: Medium issue #13

## Proposed Solutions

### Option A: Add _apply_theme pattern to DiceRollDialog
Follow the same pattern as RoseHobartDialog: define `_apply_theme()`, connect `theme().changed.connect(self._apply_theme)`, call on init.

- **Effort**: Small
- **Risk**: Low

## Acceptance Criteria

- [ ] DiceRollDialog responds to runtime theme changes
- [ ] Pattern matches RoseHobartDialog

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | Inconsistency between two new dialogs |

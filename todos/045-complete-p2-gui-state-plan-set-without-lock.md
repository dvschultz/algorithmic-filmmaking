---
status: complete
priority: p2
issue_id: "045"
tags: [code-review, thread-safety, gui-state, plan-controller]
dependencies: []
---

# GUIState current_plan Set Without Lock in PlanController

## Problem Statement

`PlanController.present()` sets `self._gui_state.current_plan = plan` directly, bypassing the lock. GUIState claims to be thread-safe ("All access to mutable state is protected by an internal lock") but this write is unprotected.

## Findings

**Python Reviewer (MEDIUM)**: `gui_state.set_plan()` exists and uses the lock, but PlanController doesn't call it.

## Proposed Solutions

### Option A: Use set_plan() Method (Recommended)

Change `PlanController.present()` to use `self._gui_state.set_plan(plan)` instead of direct assignment.

**Pros:** Consistent with GUIState's thread-safety contract
**Cons:** None
**Effort:** Small
**Risk:** Low

## Technical Details

- **File:** `core/plan_controller.py` line 91
- **Component:** PlanController

## Acceptance Criteria

- [ ] `PlanController.present()` uses `gui_state.set_plan()` not direct assignment
- [ ] All plan-related tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-28 | Created from PR #61 review — Python Reviewer | Always use locked accessors when a class promises thread-safety |

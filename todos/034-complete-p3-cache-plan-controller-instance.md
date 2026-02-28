---
status: complete
priority: p3
issue_id: "034"
tags: [code-review, performance, plan-controller]
dependencies: []
---

# Cache PlanController Instance Instead of Creating Per Call

## Problem Statement

`_get_plan_controller()` in `chat_tools.py` creates a new `PlanController(gui_state)` on every tool call. While `PlanController` is lightweight (just stores a reference), this is misleading — the docstring says "Get" but it always creates new. It could also cause issues if PlanController ever gains state.

## Findings

**Performance Oracle (OPT-1)**: `_get_plan_controller` creates new object per call.
**Python Reviewer**: Docstring says "Get the PlanController" but always creates a new instance.

## Proposed Solutions

### Option A: Cache on main_window (Recommended)

Create the PlanController once and store it on main_window:

```python
# In main_window.__init__
self.plan_controller = PlanController(self.gui_state)

# In _get_plan_controller
def _get_plan_controller(main_window):
    return main_window.plan_controller
```

**Pros:** Honest naming, slight perf improvement, future-proof
**Cons:** Trivial change
**Effort:** Small
**Risk:** Low

## Acceptance Criteria

- [ ] PlanController created once and cached
- [ ] All plan tools use the cached instance
- [ ] All tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-27 | Created from branch review — Performance Oracle + Python Reviewer findings | Factory functions should be honest about whether they create or retrieve |

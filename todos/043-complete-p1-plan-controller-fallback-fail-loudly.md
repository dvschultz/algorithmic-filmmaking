---
status: complete
priority: p1
issue_id: "043"
tags: [code-review, architecture, plan-controller]
dependencies: []
---

# `_get_plan_controller` Fallback Creates Orphan Instances

## Problem Statement

The `_get_plan_controller` fallback path silently creates a new `PlanController` on every call if `main_window.plan_controller` is missing. This bypasses the cached instance, creating multiple competing state machines.

## Findings

**Python Reviewer (HIGH)**: The comment says "shouldn't happen in normal flow" — if true, this should raise an error rather than silently degrading.

**Architecture Strategist (Medium)**: Latent correctness bug. Each new instance can't see prior plan state.

## Proposed Solutions

### Option A: Fail Loudly (Recommended)

Remove the fallback, raise `AttributeError`:

```python
def _get_plan_controller(main_window) -> PlanController:
    controller = getattr(main_window, 'plan_controller', None)
    if controller is None:
        raise AttributeError(
            "main_window.plan_controller not found — "
            "PlanController must be initialized during MainWindow setup"
        )
    return controller
```

**Pros:** Fails fast, no silent degradation
**Cons:** Crashes if init is wrong (but that's a bug worth catching)
**Effort:** Small
**Risk:** Low

## Technical Details

- **File:** `core/chat_tools.py` lines 43-52
- **Component:** Plan tool helper

## Acceptance Criteria

- [ ] Fallback path removed, raises AttributeError instead
- [ ] MainWindow always initializes plan_controller at init
- [ ] All plan tool tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-28 | Created from PR #61 review — Python Reviewer + Architecture Strategist | Fallbacks that silently degrade are worse than failures |

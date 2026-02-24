---
status: complete
priority: p3
issue_id: "020"
tags: [code-review, performance, ui]
dependencies: []
---

# Debounce Cost Estimation Refresh in ReferenceGuideDialog

## Problem Statement

Cost estimation (`_refresh_cost_estimates`) fires on every checkbox toggle in the Reference Guide dialog. During initialization, each of the 7 dimension rows calls `checkbox.setChecked(is_available)` which triggers `toggled` -> `_refresh_cost_estimates()`. With all 7 dimensions available, this means 7 redundant cost recomputations during dialog init. Additionally, rapid user toggling of checkboxes fires one recomputation per toggle with no debouncing.

## Findings

**Location:** `ui/dialogs/reference_guide_dialog.py`

Each dimension row connects checkbox toggle to cost refresh at line 331:

```python
checkbox.toggled.connect(lambda *_: self._refresh_cost_estimates())
```

The checkbox is set during row creation at line 311:

```python
checkbox.setChecked(is_available)
```

There are 7 dimensions defined in `DIMENSION_INFO` (lines 37-80): color, brightness, shot_scale, audio, embedding, movement, duration. Each `setChecked()` call emits `toggled` which triggers `_refresh_cost_estimates()`.

The `_refresh_cost_estimates` method (line 365) calls `estimate_sequence_cost()` which builds cost estimates from active dimensions -- not expensive individually, but 7 redundant calls during init is wasteful and could become noticeable if the estimation logic grows more complex.

The `_update_clip_counts` method (called on source combo change at line 363) also calls `_refresh_cost_estimates()`, adding another trigger path.

## Proposed Solutions

### Option A: QTimer Debounce (Recommended)
Add an 80ms single-shot `QTimer` so rapid toggles produce a single recomputation:

```python
def __init__(self, ...):
    ...
    self._cost_debounce = QTimer(self)
    self._cost_debounce.setSingleShot(True)
    self._cost_debounce.setInterval(80)
    self._cost_debounce.timeout.connect(self._refresh_cost_estimates)

# Replace direct connection:
checkbox.toggled.connect(lambda *_: self._cost_debounce.start())
```

**Pros:** Clean pattern, eliminates all redundant calls (init and user interaction), minimal code
**Cons:** 80ms delay before estimates update (imperceptible to user)
**Effort:** Small

### Option B: Block Signals During Init
Use `blockSignals(True)` on checkboxes during the dimension row creation loop, then unblock after all rows are created, followed by a single `_refresh_cost_estimates()` call.

**Pros:** No timer overhead, init fires exactly once
**Cons:** Does not help with rapid user toggling after init; signal blocking is error-prone
**Effort:** Small

## Acceptance Criteria

- [ ] Dialog initialization triggers `_refresh_cost_estimates` at most once (not 7 times)
- [ ] Rapid checkbox toggling produces at most one recomputation per debounce window
- [ ] Cost estimates still update correctly after user interaction settles
- [ ] Existing tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-24 | Created from PR #58 code review | QTimer debounce is standard Qt pattern for coalescing rapid signal bursts |

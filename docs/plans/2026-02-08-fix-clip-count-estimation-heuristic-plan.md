---
title: "fix: Update clip count estimation heuristic to 12 clips/minute"
type: fix
date: 2026-02-08
---

# fix: Update clip count estimation heuristic to 12 clips/minute

## Overview

The pre-detection clip count heuristic uses `_CLIPS_PER_MINUTE = 2.0` (30s average clip), but real-world scene detection on typical content produces clips averaging ~5 seconds. This causes cost/time estimates to be 6x too low, misleading users about what analysis will actually cost.

Change the constant to `12.0` (60s / 5s = 12 clips/minute).

## Acceptance Criteria

- [x] `_CLIPS_PER_MINUTE` constant updated from `2.0` to `12.0` in `ui/dialogs/intention_import_dialog.py:45`
- [x] Comment on the constant updated to document the 5-second derivation
- [x] Lightweight tests added to pin the heuristic value and verify edge cases
- [x] Existing `test_cost_estimates.py` tests still pass (they are parameterized and independent)
- [x] `~` prefix still displays for estimated clip counts

## Context

**Brainstorm:** `docs/brainstorms/2026-02-08-cost-estimate-clip-heuristic-brainstorm.md`

The heuristic was introduced in commit `d21cca1` (2026-02-07) with `_CLIPS_PER_MINUTE = 2.0`. User feedback confirms 5-second average clips are the norm for their content, making 12 clips/minute the correct heuristic.

**Impact on displayed estimates:**

| Video Duration | Old (2/min) | New (12/min) |
|---------------|-------------|--------------|
| 1 minute      | ~2 clips    | ~12 clips    |
| 5 minutes     | ~10 clips   | ~60 clips    |
| 30 minutes    | ~60 clips   | ~360 clips   |

## MVP

### 1. Update constant (`ui/dialogs/intention_import_dialog.py`)

```python
# Line 44-45: Update constant and comment
_CLIPS_PER_MINUTE = 12.0  # ~5 seconds per clip at default detection threshold
```

No other code changes needed — `_estimate_clip_count()` on line 719 already uses the constant via `max(1, round(total_minutes * _CLIPS_PER_MINUTE))`.

### 2. Add tests (`tests/test_intention_import_heuristic.py`)

```python
def test_clips_per_minute_constant():
    """Guard against accidental changes to the heuristic."""
    from ui.dialogs.intention_import_dialog import _CLIPS_PER_MINUTE
    assert _CLIPS_PER_MINUTE == 12.0

def test_clip_count_heuristic_5min_video():
    """A 5-minute video should estimate ~60 clips."""
    from ui.dialogs.intention_import_dialog import _CLIPS_PER_MINUTE, _DEFAULT_URL_DURATION_SECONDS
    total_minutes = _DEFAULT_URL_DURATION_SECONDS / 60.0
    expected = max(1, round(total_minutes * _CLIPS_PER_MINUTE))
    assert expected == 60

def test_clip_count_heuristic_short_video():
    """A 3-second video should estimate at least 1 clip."""
    from ui.dialogs.intention_import_dialog import _CLIPS_PER_MINUTE
    total_minutes = 3.0 / 60.0
    expected = max(1, round(total_minutes * _CLIPS_PER_MINUTE))
    assert expected == 1
```

### 3. Verify

```bash
pytest tests/test_cost_estimates.py tests/test_intention_import_heuristic.py -v
```

## Files Changed

| File | Change |
|------|--------|
| `ui/dialogs/intention_import_dialog.py` | Update `_CLIPS_PER_MINUTE` from `2.0` to `12.0`, update comment |
| `tests/test_intention_import_heuristic.py` | New file: 3 tests pinning the heuristic |

## References

- Brainstorm: `docs/brainstorms/2026-02-08-cost-estimate-clip-heuristic-brainstorm.md`
- Prior plan: `docs/plans/2026-02-07-feat-sequence-cost-estimate-panel-plan.md`
- Introducing commit: `d21cca1`

# Brainstorm: Fix Clip Count Estimation Heuristic

**Date:** 2026-02-08
**Status:** Ready for planning

## What We're Building

Update the clip count estimation heuristic from ~2 clips/minute (30s average) to ~12 clips/minute (5s average). The current heuristic dramatically underestimates clip counts, leading to misleadingly low cost/time estimates before analysis runs.

## Why This Approach

The current `_CLIPS_PER_MINUTE = 2.0` assumes 30-second average clips. Real-world scene detection on typical content produces clips averaging ~5 seconds. This 6x underestimate means users see cost/time figures that are far lower than reality.

The fix is a single constant change: `_CLIPS_PER_MINUTE = 12.0` (60s / 5s per clip).

## Key Decisions

1. **Average clip duration: 5 seconds** - Based on actual usage patterns with real content through scene detection
2. **Hardcoded constant** - No user-configurable setting; 5s is a reliable default
3. **Apply everywhere** - Any pre-detection clip count estimate uses duration/5, not just the intention dialog
4. **Keep URL default at 300s** - YouTube videos average ~5 minutes; 60 estimated clips is acceptable
5. **Keep `~` prefix** - Estimates remain approximate; the tilde indicator stays

## Scope

- Change `_CLIPS_PER_MINUTE` from `2.0` to `12.0` in `intention_import_dialog.py`
- Audit any other location that estimates clip counts before detection
- Update any tests that assert on the old heuristic value

## Impact

| Video Duration | Old Estimate (2/min) | New Estimate (12/min) |
|---------------|---------------------|----------------------|
| 1 minute      | ~2 clips            | ~12 clips            |
| 5 minutes     | ~10 clips           | ~60 clips            |
| 30 minutes    | ~60 clips           | ~360 clips           |
| 1 hour        | ~120 clips          | ~720 clips           |

## Open Questions

None - scope is clear and minimal.

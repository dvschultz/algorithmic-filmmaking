---
status: complete
priority: p3
issue_id: "052"
tags: [code-review, dry, simplicity, video-player]
dependencies: []
---

# Extract _safe_seek Helper in video_player.py

## Problem Statement

The try/except `mpv.ShutdownError`/Exception pattern for MPV seek operations appears 4 times with identical boilerplate in `seek_to`, `set_clip_range`, `stop`, and `_set_position`.

## Findings

**Code Simplicity Reviewer**: ~20 LOC saved by extracting a `_safe_seek` helper method.

## Proposed Solutions

### Option A: Extract Helper (Recommended)

```python
def _safe_mpv_command(self, fn, *args):
    if self._shutting_down:
        return
    try:
        fn(*args)
    except mpv.ShutdownError:
        pass
    except Exception:
        logger.warning("MPV command failed", exc_info=True)
```

**Pros:** DRY, single point of change for error handling
**Cons:** Slight indirection
**Effort:** Small
**Risk:** Low

## Technical Details

- **File:** `ui/video_player.py` — `seek_to`, `set_clip_range`, `stop`, `_set_position`

## Acceptance Criteria

- [ ] Seek error handling extracted to one helper
- [ ] All callers use the helper
- [ ] All tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-28 | Created from PR #61 review — Code Simplicity Reviewer | Repeated try/except patterns are good extraction candidates |

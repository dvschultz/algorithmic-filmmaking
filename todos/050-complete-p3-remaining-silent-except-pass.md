---
status: complete
priority: p3
issue_id: "050"
tags: [code-review, quality, error-handling, video-player]
dependencies: []
---

# Remaining Silent except:pass in Video Player Handlers

## Problem Statement

Several handlers in `video_player.py` still use bare `except Exception: pass` which silently swallows all errors. This makes debugging extremely difficult when something goes wrong.

## Findings

**Python Reviewer (HIGH)**: `_on_set_a` (line 576-579), `_on_set_b` (line 589-592), `playback_speed` setter (line 476-477), and `mute` setter (line 492-493) all silently swallow exceptions.

## Proposed Solutions

### Option A: Add Debug Logging (Recommended)

Replace `except Exception: pass` with:
```python
except Exception:
    if not self._shutting_down:
        logger.debug("_on_set_a failed", exc_info=True)
```

**Pros:** Diagnosable failures, minimal overhead
**Cons:** None
**Effort:** Small
**Risk:** Low

## Technical Details

- **File:** `ui/video_player.py` lines 476-477, 492-493, 576-579, 589-592

## Acceptance Criteria

- [ ] All `except Exception: pass` blocks have at minimum debug logging
- [ ] Shutdown guard prevents noise during normal shutdown
- [ ] All tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-28 | Created from PR #61 review — Python Reviewer | Silent exception swallowing makes debugging painful |

---
status: complete
priority: p3
issue_id: "036"
tags: [code-review, code-quality, video-player, error-handling]
dependencies: []
---

# Replace Broad except Exception in VideoPlayer

## Problem Statement

`VideoPlayer` uses `except Exception: pass` in several places (shutdown paths, MPV command wrappers). While this prevents crashes during shutdown, it also silently swallows legitimate errors during normal operation.

## Findings

**Python Reviewer**: Broad `except Exception: pass` in VideoPlayer. Should catch specific MPV exceptions or at least log.

## Proposed Solutions

### Option A: Catch Specific + Log (Recommended)

Replace broad catches with specific MPV exceptions plus logging:

```python
try:
    self._mpv.command(...)
except mpv.ShutdownError:
    pass  # Expected during shutdown
except Exception:
    if not self._shutting_down:
        logger.warning("MPV command failed", exc_info=True)
```

**Pros:** Errors visible in logs, shutdown still clean
**Cons:** Slightly more verbose
**Effort:** Small
**Risk:** Low

## Acceptance Criteria

- [ ] Shutdown paths still don't raise
- [ ] Non-shutdown errors are logged
- [ ] No test regressions

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-27 | Created from branch review — Python Reviewer finding | Broad except hides real bugs; catch specific + log others |

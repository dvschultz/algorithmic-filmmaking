---
status: complete
priority: p3
issue_id: "053"
tags: [code-review, thread-safety, security, video-player]
dependencies: []
---

# locale.setlocale Main-Thread Guard in VideoPlayer

## Problem Statement

`locale.setlocale()` is called in `VideoPlayer.__init__()` but is not thread-safe (modifies process-global state). If a VideoPlayer is ever constructed off-main-thread, this creates a race condition.

## Findings

**Security Sentinel (Medium)**: `locale.setlocale()` is documented as not thread-safe. The call in `main.py` is safe (startup), but the one in `VideoPlayer.__init__` could be problematic.

## Proposed Solutions

### Option A: Add Main-Thread Guard (Recommended)

```python
import threading
if threading.current_thread() is threading.main_thread():
    locale.setlocale(locale.LC_NUMERIC, 'C')
```

**Pros:** Prevents race condition while maintaining defense-in-depth
**Cons:** None
**Effort:** Small
**Risk:** Low

## Technical Details

- **File:** `ui/video_player.py` lines 108-111
- **File:** `main.py` line 65

## Acceptance Criteria

- [ ] VideoPlayer locale call only runs on main thread
- [ ] All tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-28 | Created from PR #61 review — Security Sentinel | locale.setlocale is process-global and not thread-safe |

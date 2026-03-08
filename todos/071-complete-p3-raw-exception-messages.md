---
status: complete
priority: p3
issue_id: "071"
tags: [code-review, security, rose-hobart]
dependencies: []
---

# Raw Exception Messages Leaked to GUI

## Problem Statement

`RoseHobartWorker.run()` catches all exceptions and emits `str(e)` directly to the error signal, displayed in a QMessageBox. Raw exception messages from InsightFace/OpenCV/numpy could leak internal paths, model configuration, or stack trace fragments.

## Findings

**Security Sentinel (Low)**: Sanitize error messages before display.

## Proposed Solutions

### Option A: Generic User Message, Log Details (Recommended)

```python
except Exception as e:
    if not self._cancelled:
        logger.error(f"Rose Hobart generation error: {e}", exc_info=True)
        self.error.emit("Face matching failed. Check logs for details.")
```

**Effort:** Small | **Risk:** Low

## Technical Details

- **File:** `ui/dialogs/rose_hobart_dialog.py` line 158

## Acceptance Criteria

- [ ] User sees generic error message
- [ ] Full details logged for debugging

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from Rose Hobart code review — Security Sentinel | Never show raw str(e) to users |

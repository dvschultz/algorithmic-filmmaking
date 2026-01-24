---
status: complete
priority: p2
issue_id: "008"
tags: [code-review, resource-leak, ui]
dependencies: ["002"]
---

# No Worker Cleanup on Window Close

## Problem Statement

When the main window is closed, any running workers (DownloadWorker, DetectionWorker, ThumbnailWorker) are not explicitly stopped. This can leave orphaned processes and threads running.

**Why it matters:** Users expect closing the application to stop all operations. Orphaned processes consume resources and can cause confusion.

## Findings

**Location:** `ui/main_window.py` - Missing `closeEvent` override

```python
class MainWindow(QMainWindow):
    # Workers are stored but never explicitly stopped
    self.detection_worker: Optional[DetectionWorker] = None
    self.thumbnail_worker: Optional[ThumbnailWorker] = None
    self.download_worker: Optional[DownloadWorker] = None

    # No closeEvent defined
```

## Proposed Solutions

### Option A: Override closeEvent (Recommended)
**Pros:** Ensures clean shutdown
**Cons:** Requires cancellation support in workers (see #003)
**Effort:** Medium
**Risk:** Low

```python
def closeEvent(self, event):
    """Clean up workers before closing."""
    workers = [
        self.detection_worker,
        self.thumbnail_worker,
        self.download_worker,
    ]

    for worker in workers:
        if worker and worker.isRunning():
            worker.terminate()  # Or worker.cancel() if implemented
            if not worker.wait(3000):  # Wait up to 3 seconds
                worker.terminate()

    event.accept()
```

## Recommended Action

Implement Option A after #002 (subprocess cleanup) and #003 (cancellation) are done.

## Technical Details

**Affected files:** `ui/main_window.py`

## Acceptance Criteria

- [ ] Closing window stops all running workers
- [ ] Workers have up to 3 seconds to finish gracefully
- [ ] Force terminate after timeout
- [ ] No orphan processes after close

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Override closeEvent for cleanup |

## Resources

- [QWidget.closeEvent](https://doc.qt.io/qt-6/qwidget.html#closeEvent)

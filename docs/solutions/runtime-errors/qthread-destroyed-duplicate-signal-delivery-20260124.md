---
module: Scene Ripper
date: 2026-01-24
problem_type: runtime_error
component: background_job
symptoms:
  - "QThread: Destroyed while thread is still running"
  - "Application crash during scene detection completion"
  - "Signal handler called multiple times unexpectedly"
root_cause: thread_violation
resolution_type: code_fix
severity: high
tags: [pyside6, qt, qthread, signal, duplicate-delivery, guard-pattern]
---

# Troubleshooting: QThread Destroyed While Running Due to Duplicate Qt Signal Delivery

## Problem
PySide6/Qt application crashed with "QThread: Destroyed while thread is still running" during scene detection completion. The crash was caused by Qt's `finished` signal being delivered twice to the same handler, creating duplicate background workers.

## Environment
- Module: Scene Ripper (algorithmic-filmmaking)
- Framework: PySide6 (Qt 6 for Python)
- Affected Component: QThread workers (DetectionWorker, ThumbnailWorker, ColorAnalysisWorker)
- Date: 2026-01-24

## Symptoms
- Qt warning: "QThread: Destroyed while thread is still running"
- Application crash during scene detection completion phase
- Logging revealed `_on_thumbnails_finished` handler called twice for single signal emission
- Multiple ColorAnalysisWorker instances created (should only be one)
- New MainWindow instance being initialized while old workers still running

## What Didn't Work

**Attempted Solution 1:** Parent QTimer to MainWindow for proper lifecycle
- **Why it failed:** QTimer lifecycle wasn't the issue - the duplicate signal delivery was

**Attempted Solution 2:** Move playback state initialization after `_setup_ui()`
- **Why it failed:** Initialization order wasn't the root cause

## Solution

Added guard flags and `Qt.UniqueConnection` to prevent duplicate signal handler execution.

**Code changes**:

```python
# Before (broken):
class MainWindow(QMainWindow):
    def __init__(self):
        # ... setup ...
        pass

    def _on_detection_finished(self, source, clips):
        # Start thumbnail generation
        self.thumbnail_worker = ThumbnailWorker(source, clips)
        self.thumbnail_worker.finished.connect(self._on_thumbnails_finished)
        self.thumbnail_worker.start()

    def _on_thumbnails_finished(self):
        # Start color analysis - THIS WAS CALLED TWICE!
        self.color_worker = ColorAnalysisWorker(self.clips)
        self.color_worker.start()

# After (fixed):
class MainWindow(QMainWindow):
    def __init__(self):
        # ... setup ...
        # Guards to prevent duplicate signal handling
        self._detection_finished_handled = False
        self._thumbnails_finished_handled = False

    def _on_detect_click(self):
        # Reset guards for new detection run
        self._detection_finished_handled = False
        self._thumbnails_finished_handled = False
        # ... rest of detection setup ...

    @Slot(object, list)
    def _on_detection_finished(self, source, clips):
        # Guard against duplicate calls
        if self._detection_finished_handled:
            logger.warning("_on_detection_finished already handled, ignoring duplicate call")
            return
        self._detection_finished_handled = True

        # Start thumbnail generation with UniqueConnection
        self.thumbnail_worker = ThumbnailWorker(source, clips)
        self.thumbnail_worker.finished.connect(
            self._on_thumbnails_finished,
            Qt.UniqueConnection  # Prevents duplicate connections
        )
        self.thumbnail_worker.start()

    @Slot()
    def _on_thumbnails_finished(self):
        # Guard against duplicate calls
        if self._thumbnails_finished_handled:
            logger.warning("_on_thumbnails_finished already handled, ignoring duplicate call")
            return
        self._thumbnails_finished_handled = True

        # Now safe to create single worker
        self.color_worker = ColorAnalysisWorker(self.clips)
        self.color_worker.finished.connect(
            self._on_color_analysis_finished,
            Qt.UniqueConnection
        )
        self.color_worker.start()
```

## Why This Works

1. **ROOT CAUSE**: Qt's signal-slot mechanism can deliver signals multiple times in certain scenarios (possibly due to event loop timing or internal Qt mechanics). The `finished` signal was being received twice by `_on_thumbnails_finished`.

2. **Guard flags** ensure the handler logic only executes once per operation cycle. Even if the signal is delivered multiple times, subsequent calls are ignored.

3. **`Qt.UniqueConnection`** prevents the same slot from being connected multiple times to the same signal. While this didn't fully prevent the issue, it's good defensive practice.

4. **`@Slot()` decorator** explicitly marks methods as Qt slots, improving signal-slot connection reliability.

5. **Reset guards on new operation** ensures the guards work correctly for subsequent detection runs.

## Prevention

When working with PySide6/Qt QThread workers:

1. **Always use guard flags** for signal handlers that create resources (workers, timers, etc.)
   ```python
   if self._handler_executed:
       return
   self._handler_executed = True
   ```

2. **Use `Qt.UniqueConnection`** when connecting signals:
   ```python
   worker.finished.connect(handler, Qt.UniqueConnection)
   ```

3. **Add `@Slot()` decorators** to signal handler methods:
   ```python
   @Slot()
   def _on_worker_finished(self):
       pass
   ```

4. **Add logging to track signal delivery** during development:
   ```python
   logger.info(f"Handler called, already_handled: {self._flag}")
   ```

5. **Reset guards appropriately** when starting new operation cycles

## Related Issues

No related issues documented yet.

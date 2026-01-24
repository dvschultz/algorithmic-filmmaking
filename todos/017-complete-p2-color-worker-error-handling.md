---
status: complete
priority: p2
issue_id: "017"
tags: [code-review, error-handling, pattern-consistency]
dependencies: []
---

# Add Error Handling and Cancel Support to ColorAnalysisWorker

## Problem Statement

`ColorAnalysisWorker` lacks try/except error handling and a `cancel()` method, making it inconsistent with other workers in the codebase. If color extraction fails, the worker crashes silently. If the user closes the window mid-analysis, it cannot be gracefully stopped.

**Why it matters:** Pattern inconsistency makes the code harder to maintain. Silent failures make debugging difficult.

## Findings

**Location:** `/Users/derrickschultz/repos/algorithmic-filmmaking/ui/main_window.py:159-177`

```python
class ColorAnalysisWorker(QThread):
    progress = Signal(int, int)
    color_ready = Signal(str, list)
    finished = Signal()
    # Missing: error = Signal(str)
    # Missing: _cancelled flag and cancel() method

    def run(self):
        # No try/except around extract_dominant_colors()
        for i, clip in enumerate(self.clips):
            if clip.thumbnail_path and clip.thumbnail_path.exists():
                colors = extract_dominant_colors(clip.thumbnail_path)  # Can throw
                self.color_ready.emit(clip.id, colors)
```

**Compare with ThumbnailWorker (lines 78-88):**
```python
try:
    thumb_path = generator.generate_clip_thumbnail(...)
    self.thumbnail_ready.emit(clip.id, str(thumb_path))
except Exception:
    pass  # Skip failed thumbnails
```

**Compare with DownloadWorker (lines 95-124):**
- Has `_cancelled` flag and `cancel()` method
- Has `error` signal

**Found by:** pattern-recognition-specialist, architecture-strategist agents

## Proposed Solutions

### Option A: Match ThumbnailWorker Pattern (Recommended)
Add try/except to skip failures silently:
```python
def run(self):
    for i, clip in enumerate(self.clips):
        try:
            if clip.thumbnail_path and clip.thumbnail_path.exists():
                colors = extract_dominant_colors(clip.thumbnail_path)
                self.color_ready.emit(clip.id, colors)
        except Exception:
            pass  # Skip failed color extraction
        self.progress.emit(i + 1, total)
    self.finished.emit()
```
- **Pros:** Consistent with ThumbnailWorker, simple
- **Cons:** Silent failure (but color extraction failure is non-critical)
- **Effort:** Small
- **Risk:** Low

### Option B: Full Worker Pattern with Cancel and Error
Add cancel support and error signal:
```python
class ColorAnalysisWorker(QThread):
    progress = Signal(int, int)
    color_ready = Signal(str, list)
    finished = Signal()
    error = Signal(str)

    def __init__(self, clips):
        super().__init__()
        self.clips = clips
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            for i, clip in enumerate(self.clips):
                if self._cancelled:
                    break
                # ... processing
        except Exception as e:
            self.error.emit(str(e))
```
- **Pros:** Most robust, consistent with DownloadWorker
- **Cons:** More code for a non-critical feature
- **Effort:** Medium
- **Risk:** Low

## Technical Details

**Affected files:**
- `ui/main_window.py` - ColorAnalysisWorker class

## Acceptance Criteria

- [ ] ColorAnalysisWorker has try/except in run() method
- [ ] Color extraction failures do not crash the worker
- [ ] (Optional) Worker has cancel() method for graceful shutdown
- [ ] closeEvent cleanup works properly

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Workers should handle errors consistently |

## Resources

- Existing worker patterns in main_window.py

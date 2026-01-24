---
status: complete
priority: p2
issue_id: "003"
tags: [code-review, ux, threading, downloader]
dependencies: []
---

# No Download Cancellation Support

## Problem Statement

Once a download starts, there is no way for the user to cancel it. The DownloadWorker runs until completion or error, with no mechanism to stop mid-download.

**Why it matters:** Downloads can take several minutes for long videos. Users should be able to cancel and download a different video without waiting or closing the application.

## Findings

**Location:** `ui/main_window.py:88-114` (DownloadWorker class)

```python
class DownloadWorker(QThread):
    def __init__(self, url: str):
        super().__init__()
        self.url = url
        # No cancellation flag

    def run(self):
        # No way to interrupt this
        result = downloader.download(...)
```

**Also affects:** `core/downloader.py:108-220` - download loop has no exit condition

## Proposed Solutions

### Option A: Add Cancellation Flag (Recommended)
**Pros:** Simple, follows Qt patterns
**Cons:** Requires changes to both worker and downloader
**Effort:** Medium
**Risk:** Low

```python
# In DownloadWorker
class DownloadWorker(QThread):
    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        downloader = VideoDownloader()
        result = downloader.download(
            self.url,
            progress_callback=...,
            cancel_check=lambda: self._cancelled,
        )

# In VideoDownloader.download()
def download(self, url, progress_callback=None, cancel_check=None):
    # ... in the loop:
    for line in process.stdout:
        if cancel_check and cancel_check():
            process.terminate()
            return DownloadResult(success=False, error="Download cancelled")
```

### Option B: Add Cancel Button to UI
**Pros:** Full UX improvement
**Cons:** More UI changes needed
**Effort:** Medium
**Risk:** Low

Add a "Cancel" button that appears during download and calls `self.download_worker.cancel()`.

## Recommended Action

Implement both Option A (mechanism) and Option B (UI) together.

## Technical Details

**Affected files:**
- `core/downloader.py` - Add cancel_check parameter
- `ui/main_window.py` - Add cancel method and UI button

## Acceptance Criteria

- [ ] Cancel button appears during download
- [ ] Clicking cancel stops the download within 2 seconds
- [ ] yt-dlp subprocess is properly terminated
- [ ] Partial downloaded files are cleaned up
- [ ] UI returns to ready state after cancel

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Long operations need cancellation |

## Resources

- [Qt Thread Cancellation](https://doc.qt.io/qt-6/qthread.html)

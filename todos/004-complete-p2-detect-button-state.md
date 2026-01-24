---
status: pending
priority: p2
issue_id: "004"
tags: [code-review, bug, ui]
dependencies: []
---

# Detect Button Not Re-enabled After Download

## Problem Statement

When a download completes or fails, the "Detect Scenes" button is not re-enabled, even though it was disabled at the start of the download. This leaves the UI in an inconsistent state.

**Why it matters:** Users cannot detect scenes after downloading a video without restarting the app or loading a different video.

## Findings

**Location:** `ui/main_window.py:286-327`

```python
def _download_video(self, url: str):
    self.import_btn.setEnabled(False)
    self.import_url_btn.setEnabled(False)
    self.detect_btn.setEnabled(False)  # DISABLED HERE
    # ...

def _on_download_finished(self, result):
    self.import_btn.setEnabled(True)
    self.import_url_btn.setEnabled(True)
    # detect_btn NOT re-enabled!

def _on_download_error(self, error: str):
    self.import_btn.setEnabled(True)
    self.import_url_btn.setEnabled(True)
    # detect_btn NOT re-enabled!
```

## Proposed Solutions

### Option A: Re-enable detect_btn in Both Handlers (Recommended)
**Pros:** Simple, direct fix
**Cons:** None
**Effort:** Small
**Risk:** Low

```python
def _on_download_finished(self, result):
    self.progress_bar.setVisible(False)
    self.import_btn.setEnabled(True)
    self.import_url_btn.setEnabled(True)
    self.detect_btn.setEnabled(True)  # Add this line
    # ...

def _on_download_error(self, error: str):
    self.progress_bar.setVisible(False)
    self.import_btn.setEnabled(True)
    self.import_url_btn.setEnabled(True)
    self.detect_btn.setEnabled(True)  # Add this line
    # ...
```

## Recommended Action

Implement Option A - add the missing line to both handlers.

## Technical Details

**Affected files:** `ui/main_window.py`

## Acceptance Criteria

- [ ] After successful download, Detect button is enabled
- [ ] After failed download, Detect button is enabled
- [ ] After cancelled download, Detect button is enabled

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Track button state changes symmetrically |

## Resources

- PR with the bug: Current uncommitted changes

---
status: complete
priority: p2
issue_id: "006"
tags: [code-review, reliability, downloader]
dependencies: []
---

# No Timeout on Subprocess Calls

## Problem Statement

Both `get_video_info()` and `download()` can hang indefinitely if yt-dlp freezes or the network becomes unresponsive. There is no timeout mechanism.

**Why it matters:** Users could be left waiting forever with no feedback if something goes wrong during the download.

## Findings

**Location:** `core/downloader.py:87-98` and `core/downloader.py:160-191`

```python
# get_video_info - no timeout
result = subprocess.run(cmd, capture_output=True, text=True)  # Can hang forever

# download - no timeout
process = subprocess.Popen(...)
for line in process.stdout:  # Can hang forever
    ...
```

## Proposed Solutions

### Option A: Add Timeouts (Recommended)
**Pros:** Prevents indefinite hangs
**Cons:** Need to choose reasonable timeout values
**Effort:** Small
**Risk:** Low

```python
# For get_video_info
result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

# For download loop - track elapsed time
import time
start_time = time.time()
MAX_DOWNLOAD_SECONDS = 3600  # 1 hour

for line in process.stdout:
    if time.time() - start_time > MAX_DOWNLOAD_SECONDS:
        process.kill()
        return DownloadResult(success=False, error="Download timed out")
    # ... rest of loop
```

## Recommended Action

Implement Option A with reasonable timeouts (30s for info, 1 hour for download).

## Technical Details

**Affected files:** `core/downloader.py`

## Acceptance Criteria

- [ ] get_video_info has 30 second timeout
- [ ] download has configurable max timeout (default 1 hour)
- [ ] TimeoutExpired exception is caught and returns error
- [ ] User sees meaningful error message on timeout

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Always add timeouts to external calls |

## Resources

- [subprocess timeout documentation](https://docs.python.org/3/library/subprocess.html#subprocess.run)

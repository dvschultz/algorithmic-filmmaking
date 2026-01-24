---
status: pending
priority: p1
issue_id: "002"
tags: [code-review, resource-leak, downloader]
dependencies: []
---

# Subprocess Resource Leak - No Cleanup on Exception

## Problem Statement

The `subprocess.Popen` call in `VideoDownloader.download()` is not properly cleaned up if an exception occurs during the stdout reading loop. This can leave orphaned yt-dlp processes running in the background.

**Why it matters:** Orphaned processes consume system resources and can cause issues like zombie downloads, disk space being consumed, and confusion when multiple downloads appear to be running.

## Findings

**Location:** `core/downloader.py:160-191`

```python
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

output_file = None
for line in process.stdout:  # If exception here, process is orphaned
    line = line.strip()
    # ... processing

process.wait()  # Never reached if exception thrown
```

**Problem scenarios:**
1. Exception in the loop leaves process running
2. User closes app while download in progress
3. Thread termination doesn't stop subprocess

## Proposed Solutions

### Option A: Use try/finally for Cleanup (Recommended)
**Pros:** Ensures cleanup in all cases
**Cons:** Slightly more code
**Effort:** Small
**Risk:** Low

```python
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
try:
    output_file = None
    for line in process.stdout:
        line = line.strip()
        # ... processing
finally:
    process.stdout.close()
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
```

### Option B: Use Context Manager
**Pros:** Cleaner syntax
**Cons:** Requires Python 3.2+
**Effort:** Small
**Risk:** Low

```python
with subprocess.Popen(cmd, stdout=subprocess.PIPE, ...) as process:
    for line in process.stdout:
        # ... processing
```

## Recommended Action

Implement Option A for explicit control over cleanup behavior.

## Technical Details

**Affected files:** `core/downloader.py`

## Acceptance Criteria

- [ ] Subprocess is always terminated, even on exception
- [ ] Cleanup waits briefly then force-kills if needed
- [ ] stdout pipe is closed to prevent resource leak
- [ ] Test: Close app during download - no orphan processes

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-01-24 | Created from code review | Always use try/finally with Popen |

## Resources

- [Python subprocess documentation](https://docs.python.org/3/library/subprocess.html)

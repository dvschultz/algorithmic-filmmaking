---
status: pending
priority: p2
issue_id: "086"
tags: [code-review, performance]
dependencies: []
---

# _copy_prerendered_clips Blocks Main Thread on Save

## Problem Statement

`_copy_prerendered_clips` in `core/project.py` lines 77-96 calls `shutil.copy2` synchronously during `save_project`, which runs on the main UI thread. Each pre-rendered clip is megabytes. Copying 50+ files can freeze the UI for seconds.

## Findings

- Location: `core/project.py` lines 77-96
- `shutil.copy2` is called per clip, no parallelism or async
- Save is triggered from the main thread with no background offload

## Proposed Solutions

### Option A: Use hard links instead of copy when same filesystem
Use `os.link()` instead of `shutil.copy2` when source and destination are on the same filesystem, falling back to copy on `OSError`.

- **Pros**: Near-instantaneous, simple implementation
- **Cons**: Hard links share inode — deleting source deletes data
- **Effort**: Small
- **Risk**: Low (fallback handles cross-filesystem case)

### Option B: Move save to background thread with progress dialog
Run the entire save operation on a background thread, showing a progress dialog.

- **Pros**: Consistent with other heavy operations in the app
- **Cons**: More complex, requires careful thread safety for project state
- **Effort**: Medium
- **Risk**: Medium (concurrent access to project state)

## Acceptance Criteria

- [ ] Saving a project with 50 pre-rendered clips does not freeze UI noticeably
- [ ] Pre-rendered clips are correctly persisted in the project directory

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-03-08 | Created from code review | shutil.copy2 per clip during synchronous save |

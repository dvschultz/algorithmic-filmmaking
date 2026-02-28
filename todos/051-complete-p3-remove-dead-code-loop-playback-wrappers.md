---
status: complete
priority: p3
issue_id: "051"
tags: [code-review, dead-code, simplicity, video-player]
dependencies: []
---

# Remove Dead _loop_playback Field and Thin Wrapper Methods

## Problem Statement

`_loop_playback` is set to `True` but never read anywhere. Three thin wrapper methods (`_stop`, `_on_frame_back`, `_on_frame_forward`) do nothing but call the public method — button signals could connect directly.

## Findings

**Code Simplicity Reviewer**: `_loop_playback` is dead code (1 LOC). Three wrappers are unnecessary (9 LOC). Total: 10 LOC to remove.

## Proposed Solutions

### Option A: Remove Dead Code (Recommended)

1. Delete `self._loop_playback: bool = True` (line 116)
2. Connect buttons directly: `self.stop_btn.clicked.connect(self.stop)` etc.
3. Delete `_stop`, `_on_frame_back`, `_on_frame_forward` methods

**Pros:** Less code, less indirection
**Cons:** None
**Effort:** Small
**Risk:** Low

## Technical Details

- **File:** `ui/video_player.py` lines 116, 164, 173, 182, 550-560

## Acceptance Criteria

- [ ] `_loop_playback` removed
- [ ] Button signals connect to public methods directly
- [ ] Three wrapper methods removed
- [ ] All tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-28 | Created from PR #61 review — Code Simplicity Reviewer | Remove dead code promptly |

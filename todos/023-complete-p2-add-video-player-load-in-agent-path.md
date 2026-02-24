---
status: complete
priority: p2
issue_id: "023"
tags: [code-review, agent-native, bug]
dependencies: []
---

# Add video_player.load_video Call in Agent generate_reference_guided Path

## Problem Statement

The agent path `generate_reference_guided()` in `sequence_tab.py` does not call `self.video_player.load_video(first_source.file_path)` after setting up the timeline. The dialog path (`_apply_reference_guide_sequence` at line 814) does. Without this call, after the agent generates a reference-guided sequence, the video player will not have the correct source loaded -- playback and scrubbing will be broken or show the wrong video.

## Findings

**Location:** `ui/tabs/sequence_tab.py` lines 1320-1324

Agent path (missing load_video):
```python
first_clip, first_source = matched[0]
self.timeline.set_fps(first_source.fps)
# Missing: self.video_player.load_video(first_source.file_path)
```

Dialog path (correct, line 813-814):
```python
self.timeline.set_fps(first_source.fps)
self.video_player.load_video(first_source.file_path)
```

## Proposed Solutions

### Option A: Add the Call (Recommended)

```python
first_clip, first_source = matched[0]
self.timeline.set_fps(first_source.fps)
self.video_player.load_video(first_source.file_path)  # Add this
```

**Pros:** One-line fix, matches dialog path
**Cons:** None
**Effort:** Small

## Acceptance Criteria

- [ ] Video player loads correct source after agent generates reference-guided sequence
- [ ] Playback and scrubbing work after agent-generated sequence
- [ ] Existing tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-24 | Created from PR #58 agent-native review | Agent code paths must replicate all UI setup steps, not just timeline population |

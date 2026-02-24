---
status: complete
priority: p3
issue_id: "017"
tags: [code-review, agent-native, signals]
dependencies: []
---

# Add clip_added Signal Emission in Agent generate_reference_guided Path

## Problem Statement

The agent path `generate_reference_guided()` does NOT emit `clip_added` signals when adding clips to the timeline, while every other clip-adding code path in SequenceTab does. Any observers listening to `clip_added` (e.g., for live preview updates, analytics, or future undo/redo tracking) will not be notified when the agent builds a reference-guided sequence.

## Findings

**Location:** `ui/tabs/sequence_tab.py` lines 1326-1328

The agent method's clip loop:

```python
current_frame = 0
for clip, source in matched:
    self.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
    current_frame += clip.duration_frames
```

Compare with the dialog path at lines 817-820, which correctly emits:

```python
for clip, source in sequence_clips:
    self.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
    current_frame += clip.duration_frames
    self.clip_added.emit(clip, source)
```

Other code paths that correctly emit `clip_added`:
- `_apply_sorted_sequence` (line 517)
- `_apply_exquisite_corpus_sequence` (line 609)
- `_apply_storyteller_sequence` (line 750)
- `_apply_reference_guide_sequence` (line 820)
- `_rebuild_with_params` (line 1434)
- `add_clip` (line 1071)

The agent path at line 1326 is the only one missing the signal emission.

## Proposed Solutions

### Option A: Add Signal Emission in Agent Loop (Recommended)

```python
current_frame = 0
for clip, source in matched:
    self.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
    current_frame += clip.duration_frames
    self.clip_added.emit(clip, source)  # <-- add this
```

**Pros:** One-line fix, matches all other code paths
**Cons:** None
**Effort:** Small

### Option B: Extract Common Clip-Adding Helper
Refactor all 7+ clip-adding loops into a single `_apply_clips_to_timeline(clips)` method that always emits the signal. This would prevent the pattern from diverging again.

**Pros:** DRY, prevents future omissions
**Cons:** Larger refactor, may warrant its own issue
**Effort:** Medium

## Acceptance Criteria

- [ ] `generate_reference_guided()` emits `clip_added(clip, source)` for each clip added
- [ ] Signal behavior matches the dialog path (`_apply_reference_guide_sequence`)
- [ ] Existing tests pass

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-24 | Created from PR #58 code review | Agent code paths must mirror dialog paths for signal emission parity |

---
status: complete
priority: p2
issue_id: "012"
tags: [code-review, duplication, architecture]
dependencies: []
---

# Extract Shared `_apply_matched_sequence()` Method in sequence_tab.py

## Problem Statement

Three nearly-identical `_apply_*_sequence` methods in `ui/tabs/sequence_tab.py` share approximately 80% of their code. The Exquisite Corpus, Storyteller, and Reference Guide apply methods all perform the same steps: clear timeline, set FPS, load video, add clips in order, update preview, update dropdown, set algorithm name, persist on sequence, and transition state. Additionally, the agent path `generate_reference_guided` duplicates this same logic a fourth time. This creates a maintenance risk where a fix applied to one path (e.g., a timeline update bug) may not be applied to the others.

## Findings

**Location:** `ui/tabs/sequence_tab.py`

### `_apply_exquisite_corpus_sequence` (lines 586-638)

```python
def _apply_exquisite_corpus_sequence(self, sequence_clips: list):
    # Clear and populate timeline
    self.timeline.clear_timeline()
    first_clip, first_source = sequence_clips[0]
    self.timeline.set_fps(first_source.fps)
    self.video_player.load_video(first_source.file_path)
    current_frame = 0
    for clip, source in sequence_clips:
        self.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
        current_frame += clip.duration_frames
        self.clip_added.emit(clip, source)
    self.timeline_preview.set_clips(sequence_clips, self._sources)
    self.timeline._on_zoom_fit()
    # ... dropdown update, algorithm set, state transition
```

### `_apply_storyteller_sequence` (lines 727-778)

```python
def _apply_storyteller_sequence(self, sequence_clips: list):
    # Identical structure: clear, fps, load, add clips, preview, zoom, dropdown, state
```

### `_apply_reference_guide_sequence` (lines 799-852)

```python
def _apply_reference_guide_sequence(self, sequence_clips: list):
    # Same pattern plus: storing dialog config on sequence
```

### `generate_reference_guided` (lines 1260-1369) - Agent path

```python
def generate_reference_guided(self, ...):
    # Lines 1320-1349: Same timeline population pattern duplicated again
```

**Shared steps across all four paths:**
1. `self.timeline.clear_timeline()`
2. Set FPS from first source
3. Load video into player
4. Iterate clips: `add_clip()` + emit `clip_added`
5. `self.timeline_preview.set_clips()`
6. `self.timeline._on_zoom_fit()`
7. Update algorithm dropdown (add item if missing, set current text)
8. Set `self._current_algorithm`
9. Set `sequence.algorithm`
10. `self._set_state(self.STATE_TIMELINE)`

**Differences:**
- Dropdown label text varies ("Exquisite Corpus", "Storyteller", "Reference Guide")
- Reference Guide stores extra metadata on sequence (`reference_source_id`, `dimension_weights`, etc.)
- Agent path returns a dict instead of being void

## Proposed Solutions

### Option A: Extract `_apply_matched_sequence()` Helper (Recommended)
**Pros:** Eliminates ~120 lines of duplication, single place for timeline population logic, easy to extend
**Cons:** Need to handle the metadata attachment for Reference Guide as a post-hook
**Effort:** Medium
**Risk:** Low

```python
def _apply_matched_sequence(
    self,
    sequence_clips: list,
    algorithm_key: str,
    algorithm_label: str,
    sequence_metadata: dict = None,
):
    """Apply a dialog-generated sequence to the timeline.

    Shared by Exquisite Corpus, Storyteller, Reference Guide, and agent paths.
    """
    if not sequence_clips:
        return

    self.timeline.clear_timeline()

    first_clip, first_source = sequence_clips[0]
    self.timeline.set_fps(first_source.fps)
    self.video_player.load_video(first_source.file_path)

    current_frame = 0
    for clip, source in sequence_clips:
        self.timeline.add_clip(clip, source, track_index=0, start_frame=current_frame)
        current_frame += clip.duration_frames
        self.clip_added.emit(clip, source)

    self.timeline_preview.set_clips(sequence_clips, self._sources)
    self.timeline._on_zoom_fit()

    self.algorithm_dropdown.blockSignals(True)
    if self.algorithm_dropdown.findText(algorithm_label) == -1:
        self.algorithm_dropdown.addItem(algorithm_label)
    self.algorithm_dropdown.setCurrentText(algorithm_label)
    self.algorithm_dropdown.blockSignals(False)

    self._current_algorithm = algorithm_key

    sequence = self.timeline.get_sequence()
    sequence.algorithm = algorithm_key
    if sequence_metadata:
        for key, value in sequence_metadata.items():
            setattr(sequence, key, value)

    self._set_state(self.STATE_TIMELINE)
```

Then each apply method becomes a thin wrapper:
```python
def _apply_exquisite_corpus_sequence(self, sequence_clips):
    self._apply_matched_sequence(sequence_clips, "exquisite_corpus", "Exquisite Corpus")

def _apply_reference_guide_sequence(self, sequence_clips):
    metadata = {}
    dialog = self.sender()
    if dialog and hasattr(dialog, '_last_ref_source_id'):
        metadata = {
            "reference_source_id": dialog._last_ref_source_id,
            "dimension_weights": dialog._last_weights,
            ...
        }
    self._apply_matched_sequence(sequence_clips, "reference_guided", "Reference Guide", metadata)
```

### Option B: Template Method Pattern
**Pros:** More OOP, allows overriding specific steps
**Cons:** Overkill for 3-4 callers, adds class complexity
**Effort:** Medium
**Risk:** Medium (harder to reason about)

### Option C: Leave As-Is with Documentation
**Pros:** No code changes, explicit per-algorithm code
**Cons:** Duplication remains, maintenance risk persists
**Effort:** None
**Risk:** Bug divergence between paths over time

## Recommended Action

Option A - Extract a shared `_apply_matched_sequence()` method with a `sequence_metadata` dict for algorithm-specific extras.

## Technical Details

**Affected Files:**
- `ui/tabs/sequence_tab.py` - Extract shared method, refactor 4 callers

**Verification:**
1. Run existing tests: `pytest tests/`
2. Manually test each algorithm path: Exquisite Corpus, Storyteller, Reference Guide
3. Test agent tool: `generate_reference_guided` via chat
4. Verify sequence metadata is correctly persisted for Reference Guide

## Acceptance Criteria

- [ ] Single `_apply_matched_sequence()` method handles all timeline population
- [ ] `_apply_exquisite_corpus_sequence` delegates to shared method
- [ ] `_apply_storyteller_sequence` delegates to shared method
- [ ] `_apply_reference_guide_sequence` delegates to shared method (with metadata)
- [ ] `generate_reference_guided` agent path delegates to shared method
- [ ] Reference Guide sequence metadata (source_id, weights, etc.) still persisted
- [ ] All existing tests pass
- [ ] No functional behavior change

## Work Log

| Date | Action | Learnings |
|------|--------|-----------|
| 2026-02-24 | Created from PR #58 code review | Dialog-based algorithms accumulate near-identical apply methods; extract shared logic before a fourth copy appears |

## Resources

- PR #58: Reference-Guided Remixing
- DRY principle: https://en.wikipedia.org/wiki/Don%27t_repeat_yourself

---
title: Fix Intention Workflow Tab Population
type: fix
date: 2026-01-31
---

# Fix Intention Workflow Tab Population

## Overview

The intention-based workflow (starting from Sequence tab) does not populate the Cut and Analyze tabs with clips, even though the clips are successfully created, thumbnails generated, and analysis performed. This breaks the expectation that all tabs should be populated regardless of which entry point the user starts from.

## Problem Statement

When using the intention workflow:
1. **Cut tab is not populated** - Despite all clips being detected and having thumbnails, they don't appear in the Cut tab browser
2. **Analyze tab is not populated** - Despite OCR/color analysis being performed on clips (e.g., Exquisite Corpus workflow), they don't appear in the Analyze tab

This creates an inconsistent user experience where the workflow produces results (clips appear in Sequence tab) but users can't browse/filter clips in other tabs.

## Root Cause Analysis

### Normal Flow (Collect → Cut → Analyze → Sequence)

In the normal flow, tabs are populated through these mechanisms:

1. **Cut tab population** via `_on_thumbnail_ready`:
   ```python
   # main_window.py:4180-4190
   def _on_thumbnail_ready(self, clip_id: str, thumb_path: str):
       clip = self.clips_by_id.get(clip_id)
       if clip:
           clip_source = self.sources_by_id.get(clip.source_id)
           if clip_source:
               self.cut_tab.add_clip(clip, clip_source)  # ✓ Adds to Cut tab
   ```

2. **Cut tab state sync** via `_on_thumbnails_finished`:
   ```python
   # main_window.py:4199-4217
   def _on_thumbnails_finished(self):
       self.analyze_tab.set_lookups(self.clips_by_id, self.sources_by_id)
       self.cut_tab.set_clips(self.clips)  # ✓ Syncs state
   ```

3. **Analyze tab population** - User manually selects clips and clicks "Analyze Selected" button

### Intention Workflow (Sequence tab starting point)

In the intention workflow:

1. **`_on_thumbnail_ready` IS connected** - Clips should be added to Cut tab
   ```python
   # main_window.py:6101
   self.thumbnail_worker.thumbnail_ready.connect(self._on_thumbnail_ready)
   ```

2. **BUT `_on_intention_thumbnails_finished` does NOT sync state**:
   ```python
   # main_window.py:6114-6123
   def _on_intention_thumbnails_finished(self):
       if self._thumbnails_finished_handled:
           return
       self._thumbnails_finished_handled = True
       logger.info("Intention thumbnails finished")
       if self.intention_workflow:
           self.intention_workflow.on_thumbnails_finished()
       # ❌ MISSING: self.analyze_tab.set_lookups(...)
       # ❌ MISSING: self.cut_tab.set_clips(...)
   ```

3. **`_finalize_intention_workflow` does NOT populate tabs**:
   ```python
   # main_window.py:5853-5889 (simplified)
   def _finalize_intention_workflow(self):
       for source in all_sources:
           self.project.add_source(source)
           self.collect_tab.add_source(source)  # ✓ Collect tab populated
       for clip in all_clips:
           self.project.add_clips([clip])
       # ❌ MISSING: Cut tab population
       # ❌ MISSING: Analyze tab population (for analyzed clips)
   ```

### Why Cut Tab Shows Partial Clips

The `_on_thumbnail_ready` signal IS connected in the intention workflow, so thumbnails DO trigger `cut_tab.add_clip()`. However:

1. **`set_lookups` not called** - `self.clips_by_id` may not include intention workflow clips when `_on_thumbnail_ready` runs
2. **`set_clips` not called** - The Cut tab's internal `_clips` list isn't synced with the clip browser contents
3. **Source not set** - `cut_tab.set_source()` may not be called for intention workflow sources

## Proposed Solution

Add tab population calls to the intention workflow to match the normal flow's behavior.

### Fix 1: Update `_on_intention_thumbnails_finished`

Add the missing state synchronization calls that the normal flow has:

```python
def _on_intention_thumbnails_finished(self):
    """Handle thumbnail generation completion during intention workflow."""
    if self._thumbnails_finished_handled:
        return
    self._thumbnails_finished_handled = True

    logger.info("Intention thumbnails finished")

    # Sync lookups for Analyze tab (same as normal flow)
    self.analyze_tab.set_lookups(self.clips_by_id, self.sources_by_id)

    # Sync Cut tab clips state (same as normal flow)
    self.cut_tab.set_clips(self.clips)

    if self.intention_workflow:
        self.intention_workflow.on_thumbnails_finished()
```

### Fix 2: Update `_finalize_intention_workflow`

Add Analyze tab population for workflows that perform analysis:

```python
def _finalize_intention_workflow(self):
    """Finalize the intention workflow and apply results."""
    # ... existing source/clip addition code ...

    # Ensure Cut tab has all clips from the workflow
    for clip in all_clips:
        source = self.sources_by_id.get(clip.source_id)
        if source and clip not in self.cut_tab._clips:
            self.cut_tab.add_clip(clip, source)
    self.cut_tab.set_clips(self.clips)

    # If workflow performed analysis, add clips to Analyze tab
    if algorithm == "exquisite_corpus":
        # Exquisite Corpus does OCR - add analyzed clips
        analyzed_clip_ids = [c.id for c in all_clips if c.extracted_texts]
        if analyzed_clip_ids:
            self.analyze_tab.add_clips(analyzed_clip_ids)
    elif algorithm == "color":
        # Color algorithm does color analysis
        analyzed_clip_ids = [c.id for c in all_clips if c.dominant_colors]
        if analyzed_clip_ids:
            self.analyze_tab.add_clips(analyzed_clip_ids)

    # ... rest of existing code ...
```

### Fix 3: Ensure Cut Tab Source is Set

In `_on_intention_detection_completed`, ensure the Cut tab knows about the source:

```python
def _on_intention_detection_completed(self, source, clips):
    """Handle detection completion during intention workflow."""
    # ... existing code ...

    # Ensure Cut tab knows about this source
    if source.id not in [s.id for s in self.cut_tab._sources if hasattr(self.cut_tab, '_sources')]:
        self.cut_tab.set_source(source)

    # ... rest of existing code ...
```

## Acceptance Criteria

- [x] After intention workflow completes, all detected clips appear in Cut tab
- [x] After intention workflow completes, analyzed clips appear in Analyze tab
- [x] Clip counts in Cut tab match total clips created
- [x] Analyze tab shows clips with their analysis results (OCR text, colors)
- [ ] Normal flow behavior is unchanged
- [ ] No duplicate clips in any tab

## Technical Considerations

### Signal Timing
- `_on_thumbnail_ready` runs during thumbnail generation (one call per clip)
- `_on_intention_thumbnails_finished` runs once when all thumbnails complete
- `_finalize_intention_workflow` runs last, after dialog closes

### State Dependencies
- `analyze_tab.add_clips()` requires `set_lookups()` to be called first
- `cut_tab.add_clip()` requires clip to exist in `clips_by_id`

### Guard Flags
- `_thumbnails_finished_handled` prevents duplicate processing
- Ensure guard is reset before intention workflow starts

## Files to Modify

1. **`ui/main_window.py`**
   - `_on_intention_thumbnails_finished`: Add `set_lookups` and `set_clips` calls
   - `_finalize_intention_workflow`: Add Analyze tab population based on algorithm
   - `_on_intention_detection_completed`: Ensure Cut tab source is set

## Testing

1. Start from Sequence tab → Import Videos → Run Exquisite Corpus
   - Verify: All clips appear in Cut tab
   - Verify: All clips with OCR text appear in Analyze tab

2. Start from Sequence tab → Import Videos → Run Color algorithm
   - Verify: All clips appear in Cut tab
   - Verify: All clips with colors appear in Analyze tab

3. Normal flow still works
   - Collect → Cut → Analyze → Sequence unchanged

## References

- Recent workflow fixes: commits b71998e, a837a5b
- Tab encapsulation pattern from learnings-researcher
- Signal timing pattern: use `step_started` for state-dependent work

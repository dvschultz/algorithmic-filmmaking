---
title: "refactor: Move Analysis Features to On-Demand Triggering"
type: refactor
date: 2026-01-25
status: completed
---

# Move Analysis Features to On-Demand Triggering

## Overview

Currently, color detection, shot type classification, and speech-to-text transcription run automatically after scene detection completes (controlled by `auto_analyze_colors`, `auto_classify_shots`, `auto_transcribe` settings). This is incorrect behavior that should be changed.

The correct behavior: analysis features should only run when the user explicitly triggers them from the Analyze tab after selecting clips.

## Problem Statement

**Current Flow (Incorrect):**
```
Detection → Thumbnails → [auto] Colors → [auto] Shots → [auto] Transcribe → Ready
```

**Target Flow (Correct):**
```
Detection → Thumbnails → Ready
                ↓
    [User selects clips in Cut tab]
                ↓
    [User clicks "Analyze Selected"]
                ↓
    [Clips appear in Analyze tab]
                ↓
    [User clicks analysis buttons manually]
```

### Why This Matters

1. **User Control**: Users should decide which clips to analyze, not analyze everything automatically
2. **Resource Efficiency**: Auto-analysis wastes CPU/GPU cycles on clips the user may not care about
3. **Workflow Clarity**: Clear separation between "detect scenes" and "analyze clips"
4. **Faster Initial Results**: Detection completes faster without waiting for analysis chain

## Proposed Solution

### Phase 1: Remove Auto-Chaining Logic

Remove the automatic analysis chain that triggers after thumbnail generation completes.

**File:** `ui/main_window.py`

**Current code (lines 1729-1759):**
```python
# After thumbnails finish, currently chains to:
if self.clips and self.settings.auto_analyze_colors:
    # Creates ColorAnalysisWorker...
elif self.clips and not self.settings.auto_analyze_colors:
    if self.settings.auto_classify_shots:
        # Creates ShotTypeWorker...
    else:
        self._maybe_start_transcription()
```

**New code:**
```python
# After thumbnails finish, just mark as ready
self.progress_bar.setVisible(False)
self.status_bar.showMessage(f"Ready - {len(self.clips)} scenes detected")
# Advance batch queue immediately
QTimer.singleShot(0, self._start_next_analysis)
```

### Phase 2: Add "Analyze All" Button

Add a new button to the Analyze tab that runs all three operations sequentially.

**File:** `ui/tabs/analyze_tab.py`

```python
# New signal
analyze_all_requested = Signal()

# In _create_controls():
self.analyze_all_btn = QPushButton("Analyze All")
self.analyze_all_btn.setToolTip(
    "Run all analysis operations sequentially:\n"
    "1. Extract colors\n"
    "2. Classify shot types\n"
    "3. Transcribe speech"
)
self.analyze_all_btn.setEnabled(False)
self.analyze_all_btn.clicked.connect(self._on_analyze_all_click)
controls.addWidget(self.analyze_all_btn)
```

### Phase 3: Implement Sequential "Analyze All" Handler

**File:** `ui/main_window.py`

```python
def _on_analyze_all_from_tab(self):
    """Handle 'Analyze All' request - run colors, shots, transcribe sequentially."""
    clips = self.analyze_tab.get_clips()
    if not clips:
        return

    # Store state for sequential processing
    self._analyze_all_pending = ["colors", "shots", "transcribe"]
    self._analyze_all_clips = clips

    # Start with colors
    self._start_next_analyze_all_step()

def _start_next_analyze_all_step(self):
    """Start the next step in the Analyze All sequence."""
    if not self._analyze_all_pending:
        # All done
        self.analyze_tab.set_analyzing(False)
        self.status_bar.showMessage(
            f"Analysis complete - {len(self._analyze_all_clips)} clips"
        )
        return

    next_step = self._analyze_all_pending.pop(0)

    if next_step == "colors":
        self._start_color_analysis_for_analyze_all()
    elif next_step == "shots":
        self._start_shot_analysis_for_analyze_all()
    elif next_step == "transcribe":
        self._start_transcription_for_analyze_all()
```

### Phase 4: Support Multi-Source Transcription

Currently transcription requires `current_source` for audio extraction. To support clips from multiple sources in Analyze tab:

**File:** `ui/main_window.py`

```python
def _start_transcription_for_analyze_all(self):
    """Start transcription, handling clips from multiple sources."""
    clips = self._analyze_all_clips

    # Group clips by source_id
    clips_by_source = {}
    for clip in clips:
        if clip.source_id not in clips_by_source:
            clips_by_source[clip.source_id] = []
        clips_by_source[clip.source_id].append(clip)

    # Queue transcription for each source
    self._transcription_source_queue = list(clips_by_source.items())
    self._start_next_source_transcription()

def _start_next_source_transcription(self):
    """Start transcription for the next source in queue."""
    if not self._transcription_source_queue:
        # All sources done
        self._on_analyze_all_transcription_complete()
        return

    source_id, clips = self._transcription_source_queue.pop(0)
    source = self.sources_by_id.get(source_id)

    if not source:
        logger.warning(f"Source {source_id} not found, skipping transcription")
        self._start_next_source_transcription()
        return

    # Create worker for this source's clips
    self.transcription_worker = TranscriptionWorker(
        clips, source,
        self.settings.transcription_model,
        self.settings.transcription_language
    )
    # ... connect signals and start
```

### Phase 5: Remove Auto-Analysis Settings

**File:** `core/settings.py`

Remove these fields from `Settings` dataclass:
```python
# DELETE these lines:
auto_analyze_colors: bool = True
auto_classify_shots: bool = True
auto_transcribe: bool = True
```

Remove from JSON config loading/saving:
```python
# DELETE from _load_from_json:
if "auto_analyze_colors" in detection:
    settings.auto_analyze_colors = bool(detection["auto_analyze_colors"])
# ... similar for auto_classify_shots, auto_transcribe

# DELETE from _save_to_json:
"auto_analyze_colors": settings.auto_analyze_colors,
# ... similar for others
```

**File:** `ui/settings_dialog.py`

Remove the "Automatic Analysis" group box and related checkboxes:
- `self.auto_colors_check`
- `self.auto_shots_check`
- `self.auto_transcribe_check`

### Phase 6: Update Guard Flag Resets

Ensure manual analysis handlers properly reset guard flags to prevent issues from documented PySide6 duplicate signal delivery.

**File:** `ui/main_window.py`

Each manual handler already does this (confirmed in research), but verify:
```python
def _on_analyze_colors_from_tab(self):
    # Reset guard
    self._color_analysis_finished_handled = False  # Already present
    # ...

def _on_analyze_shots_from_tab(self):
    # Reset guard
    self._shot_type_finished_handled = False  # Already present
    # ...
```

## Technical Considerations

### Documented Gotchas to Avoid

Based on `docs/solutions/`:

1. **QThread Duplicate Signal Delivery** (`qthread-destroyed-duplicate-signal-delivery-20260124.md`)
   - Always use guard flags on finished handlers
   - Use `Qt.UniqueConnection` for worker signal connections
   - Reset guards when starting new operations

2. **Source ID Mismatch** (`pyside6-thumbnail-source-id-mismatch.md`)
   - Workers may create new Source objects with different IDs
   - Sync IDs before storing results
   - Use `clips_by_id` and `sources_by_id` lookups consistently

3. **State Synchronization** (`timeline-widget-sequence-mismatch-20260124.md`)
   - Don't create duplicate state objects between tabs
   - Use property delegation for shared state
   - MainWindow remains single source of truth

### Batch Queue Behavior

With auto-analysis removed, the batch queue should advance immediately after thumbnails complete:

```
Source 1: Import → Detect → Thumbnails → Ready
                                          ↓
Source 2: Import → Detect → Thumbnails → Ready  (starts immediately)
                                          ↓
Source 3: Import → Detect → Thumbnails → Ready
```

User can manually analyze clips from any source at any time via Analyze tab.

## Acceptance Criteria

### Functional Requirements

- [x] Scene detection completes without triggering any analysis operations
- [x] Batch queue advances immediately after thumbnails complete
- [x] "Analyze Selected" button sends selected clips to Analyze tab
- [x] Each analysis button (Colors, Shots, Transcribe) works independently
- [x] New "Analyze All" button runs all three operations sequentially
- [x] Multi-source clips in Analyze tab can be transcribed (grouped by source)
- [x] Progress bar shows correct progress during analysis operations
- [x] Status bar shows appropriate messages for each operation

### Settings Requirements

- [x] `auto_analyze_colors` setting removed from Settings dataclass
- [x] `auto_classify_shots` setting removed from Settings dataclass
- [x] `auto_transcribe` setting removed from Settings dataclass
- [x] Settings dialog no longer shows "Automatic Analysis" group
- [x] Existing config files with deprecated settings don't cause errors

### Guard Flag Requirements

- [x] Guard flags reset at start of each manual analysis operation
- [x] `Qt.UniqueConnection` used for all worker signal connections
- [x] `@Slot()` decorators on all handler methods
- [x] No duplicate handler executions observed

## Files Affected

| File | Changes |
|------|---------|
| `ui/main_window.py` | Remove auto-chain logic (~130 lines), add "Analyze All" handler, add multi-source transcription support |
| `ui/tabs/analyze_tab.py` | Add "Analyze All" button and signal |
| `core/settings.py` | Remove `auto_analyze_colors`, `auto_classify_shots`, `auto_transcribe` fields |
| `ui/settings_dialog.py` | Remove "Automatic Analysis" group and related UI elements |

## Implementation Phases

### Phase 1: Remove Auto-Chaining (Core Change)
1. Modify `_on_thumbnails_finished` to skip analysis chain
2. Update batch queue to advance after thumbnails
3. Remove conditional auto-analysis logic

### Phase 2: Add "Analyze All" Feature
1. Add button to `AnalyzeTab`
2. Add signal and handler in `MainWindow`
3. Implement sequential orchestration

### Phase 3: Multi-Source Transcription
1. Group clips by source_id
2. Queue transcription per source
3. Handle completion across all sources

### Phase 4: Settings Cleanup
1. Remove settings from dataclass
2. Remove from JSON config handling
3. Remove from Settings dialog UI

### Phase 5: Testing & Verification
1. Test detection without auto-analysis
2. Test batch queue advancement
3. Test manual analysis operations
4. Test "Analyze All" sequence
5. Test multi-source transcription
6. Verify no guard flag issues

## References

### Internal References
- `ui/main_window.py:1730-1870` - Current auto-chain implementation
- `ui/tabs/analyze_tab.py:85-126` - Analyze tab controls
- `core/settings.py:226-237` - Auto-analysis settings
- `ui/settings_dialog.py:340-355` - Auto-analysis checkboxes
- `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md` - Guard flag pattern
- `docs/solutions/ui-bugs/pyside6-thumbnail-source-id-mismatch.md` - Source ID sync pattern

### Related Work
- Previous refactor: `docs/plans/2026-01-24-refactor-split-cut-analyze-tabs-plan.md` - Split tabs architecture

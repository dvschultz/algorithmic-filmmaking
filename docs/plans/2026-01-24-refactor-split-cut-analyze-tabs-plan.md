---
title: "refactor: Split AnalyzeTab into Cut and Analyze tabs"
type: refactor
date: 2026-01-24
---

# Split AnalyzeTab into Cut and Analyze Tabs

## Overview

Separate the monolithic AnalyzeTab into two focused tabs:
- **Cut Tab**: Scene detection, clip browsing (grid/list/preview), clip selection
- **Analyze Tab**: Analysis features (color extraction, shot type, transcription, future ML)

This separation of concerns improves workflow clarity and enables independent evolution of cutting and analysis features.

## Problem Statement / Motivation

The current AnalyzeTab combines two distinct workflows:
1. **Cutting**: Detecting scenes, adjusting thresholds, browsing detected clips
2. **Analyzing**: Extracting metadata (colors, shot types, transcripts) from clips

Users need to:
- Focus on scene detection without analysis UI clutter
- Analyze specific clips, not necessarily all detected clips
- Work with clips from multiple sources in analysis

## Proposed Solution

### Tab Structure

```
CURRENT:
Collect → Analyze → Generate → Sequence → Render

PROPOSED:
Collect → Cut → Analyze → Generate → Sequence → Render
              │              ▲
              │ "Analyze     │
              │  Selected"   │
              └──────────────┘
```

### Data Model

**Clip References (not copies):**
- Analyze tab maintains a `Set[str]` of clip IDs
- Clips are resolved via `MainWindow.clips_by_id`
- Single source of truth remains in MainWindow

```python
# AnalyzeTab stores references, not clips
class AnalyzeTab(BaseTab):
    def __init__(self):
        self._clip_ids: set[str] = set()  # References to MainWindow clips
```

### Key Behaviors

| Behavior | Decision |
|----------|----------|
| Clip transfer | Select clips in Cut → "Analyze Selected" → appears in Analyze |
| Multiple sends | Merge with deduplication (new clips added, duplicates skipped) |
| Re-detection | Orphaned clips automatically removed from Analyze with warning |
| Multi-source | Analyze tab can contain clips from different sources |
| Clip ownership | MainWindow owns all clips; tabs hold references |

## Technical Approach

### Phase 1: Create Cut Tab

**New file:** `ui/tabs/cut_tab.py`

```python
class CutTab(BaseTab):
    """Tab for scene detection and clip browsing."""

    # Signals
    detect_requested = Signal(float)  # threshold
    clip_selected = Signal(object)  # Clip
    clip_double_clicked = Signal(object)  # Clip
    clips_sent_to_analyze = Signal(list)  # list[str] clip IDs

    # UI Components
    # - Sensitivity slider + Detect Scenes button (from current AnalyzeTab)
    # - ClipBrowser with grid/list/preview modes
    # - VideoPlayer for clip preview
    # - "Analyze Selected" button
    # - Selection counter label
```

**Cut Tab UI Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│ Sensitivity: [====] 3.0  [Detect Scenes]  [Analyze Selected]│
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────┐ ┌─────────────────────────┐ │
│ │                             │ │                         │ │
│ │    ClipBrowser              │ │    VideoPlayer          │ │
│ │    (grid/list/preview)      │ │                         │ │
│ │                             │ │                         │ │
│ │    Multi-select enabled     │ │                         │ │
│ │                             │ │                         │ │
│ └─────────────────────────────┘ └─────────────────────────┘ │
│                                                 3 selected  │
└─────────────────────────────────────────────────────────────┘
```

**Tasks:**
- [x] Create `ui/tabs/cut_tab.py` extending BaseTab
- [x] Move sensitivity slider and detect button from AnalyzeTab
- [x] Add ClipBrowser with multi-select support
- [x] Add VideoPlayer for preview
- [x] Add "Analyze Selected" button (disabled when no selection)
- [x] Add selection counter label
- [x] Emit `clips_sent_to_analyze` signal with selected clip IDs

### Phase 2: Refactor Analyze Tab

**Modify:** `ui/tabs/analyze_tab.py`

```python
class AnalyzeTab(BaseTab):
    """Tab for clip analysis features."""

    # Signals
    analyze_colors_requested = Signal()
    analyze_shots_requested = Signal()
    transcribe_requested = Signal()
    clip_selected = Signal(object)  # Clip
    clip_removed = Signal(str)  # clip_id

    # State
    _clip_ids: set[str]  # References to MainWindow clips

    # UI Components
    # - Analysis controls (Extract Colors, Classify Shots, Transcribe)
    # - ClipBrowser (displays referenced clips with analysis data)
    # - VideoPlayer for preview
    # - Clear/Remove buttons
```

**Analyze Tab UI Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│ [Extract Colors] [Classify Shots] [Transcribe]  [Clear All] │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────┐ ┌─────────────────────────┐ │
│ │                             │ │                         │ │
│ │    ClipBrowser              │ │    VideoPlayer          │ │
│ │    (shows analysis data)    │ │                         │ │
│ │                             │ │                         │ │
│ │    Filter by color/shot     │ │                         │ │
│ │    Search transcripts       │ │                         │ │
│ └─────────────────────────────┘ └─────────────────────────┘ │
│                                             12 clips        │
└─────────────────────────────────────────────────────────────┘
```

**Tasks:**
- [x] Remove scene detection UI (moved to Cut tab)
- [x] Add `_clip_ids: set[str]` to track referenced clips
- [x] Add public method `add_clips(clip_ids: list[str])`
- [x] Add public method `remove_clip(clip_id: str)`
- [x] Add public method `clear_clips()`
- [x] Add public method `remove_orphaned_clips(valid_clip_ids: set[str])`
- [x] Add analysis control buttons
- [x] Keep existing filter/sort/search functionality
- [x] Update empty state message: "Send clips from Cut tab to analyze"

### Phase 3: Update MainWindow Coordination

**Modify:** `ui/main_window.py`

```python
# New tab creation
self.cut_tab = CutTab()
self.analyze_tab = AnalyzeTab()  # Modified version

# Tab order
self.tab_widget.addTab(self.collect_tab, "Collect")
self.tab_widget.addTab(self.cut_tab, "Cut")  # NEW
self.tab_widget.addTab(self.analyze_tab, "Analyze")  # Modified
# ... rest of tabs

# New signal connections
self.cut_tab.detect_requested.connect(self._on_detect_from_tab)
self.cut_tab.clips_sent_to_analyze.connect(self._on_clips_sent_to_analyze)
self.analyze_tab.analyze_colors_requested.connect(self._on_analyze_colors)
self.analyze_tab.analyze_shots_requested.connect(self._on_analyze_shots)
self.analyze_tab.transcribe_requested.connect(self._on_transcribe_from_tab)
```

**New handlers:**

```python
def _on_clips_sent_to_analyze(self, clip_ids: list[str]):
    """Handle clips being sent from Cut to Analyze tab."""
    self.analyze_tab.add_clips(clip_ids)
    self.tab_widget.setCurrentWidget(self.analyze_tab)
    self.status_bar.showMessage(f"Sent {len(clip_ids)} clips to Analyze")

def _on_detection_finished(self, source: Source, clips: list[Clip]):
    # ... existing code ...

    # NEW: Remove orphaned clips from Analyze tab
    valid_clip_ids = set(self.clips_by_id.keys())
    removed_count = self.analyze_tab.remove_orphaned_clips(valid_clip_ids)
    if removed_count > 0:
        logger.warning(f"Removed {removed_count} orphaned clips from Analyze tab")
```

**Tasks:**
- [x] Add CutTab instantiation and tab insertion
- [x] Update tab list in `_on_tab_changed()`
- [x] Connect Cut tab signals
- [x] Connect modified Analyze tab signals
- [x] Add `_on_clips_sent_to_analyze()` handler
- [x] Update `_on_detection_finished()` to remove orphaned clips
- [x] Move detection-related methods to work with Cut tab
- [x] Update `_on_source_selected()` to set source on Cut tab

### Phase 4: Update Tab Exports

**Modify:** `ui/tabs/__init__.py`

```python
from .base_tab import BaseTab
from .collect_tab import CollectTab
from .cut_tab import CutTab  # NEW
from .analyze_tab import AnalyzeTab  # Modified
from .generate_tab import GenerateTab
from .sequence_tab import SequenceTab
from .render_tab import RenderTab

__all__ = [
    "BaseTab",
    "CollectTab",
    "CutTab",  # NEW
    "AnalyzeTab",
    "GenerateTab",
    "SequenceTab",
    "RenderTab",
]
```

### Phase 5: Project Save/Load Updates

**Modify:** `core/project.py` and `ui/main_window.py`

Add Analyze tab clip IDs to ui_state:

```python
# In save
ui_state["analyze_clip_ids"] = list(self.analyze_tab.get_clip_ids())

# In load
if "analyze_clip_ids" in ui_state:
    self.analyze_tab.add_clips(ui_state["analyze_clip_ids"])
```

**Tasks:**
- [x] Add `get_clip_ids()` method to AnalyzeTab
- [x] Save analyze_clip_ids in ui_state
- [x] Restore analyze_clip_ids on project load
- [x] Validate clip IDs exist before restoring

## Acceptance Criteria

### Functional Requirements

- [x] Cut tab shows scene detection controls and clip browser
- [x] Cut tab supports multi-select for clips
- [x] "Analyze Selected" button sends clips to Analyze tab
- [x] Analyze tab shows only clips sent from Cut
- [x] Analyze tab has color, shot type, and transcription buttons
- [x] Analysis operations work on clips in Analyze tab
- [x] Re-detection removes orphaned clips from Analyze tab
- [x] Multiple sends merge clips with deduplication
- [x] Tab switching during operations works correctly
- [x] Project save/load preserves Analyze tab clips

### Non-Functional Requirements

- [x] No duplicate state objects between tabs (use references)
- [x] Guard flags prevent duplicate signal handling
- [x] Empty states provide clear user guidance
- [x] Performance: no regression in clip browsing speed

### Quality Gates

- [ ] All existing tests pass
- [ ] Manual testing of all user flows
- [ ] No crashes on tab switching during operations
- [ ] Project round-trip (save/load) preserves state

## Success Metrics

- Clear separation: Detection workflow in Cut, analysis workflow in Analyze
- User can select specific clips for analysis (not forced to analyze all)
- Multi-source analysis workflow enabled

## Dependencies & Prerequisites

- Existing ClipBrowser widget (reusable)
- Existing VideoPlayer widget (reusable)
- MainWindow coordination pattern (established)

## Risk Analysis & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Duplicate state objects | Medium | High | Use clip ID references, not copies |
| Orphaned clips crash | High | High | Remove orphaned clips on re-detection |
| Signal duplication | Medium | Medium | Guard flags + Qt.UniqueConnection |
| Worker coordination | Low | Medium | Workers use MainWindow.clips_by_id |

## References & Research

### Internal References

- Tab pattern: `ui/tabs/base_tab.py`
- ClipBrowser: `ui/clip_browser.py`
- Current AnalyzeTab: `ui/tabs/analyze_tab.py`
- MainWindow coordination: `ui/main_window.py:587-618`
- Documented learnings: `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`
- QThread guard pattern: `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`

### Key Patterns to Follow

**State Delegation (avoid duplication):**
```python
# CORRECT: Reference via property
@property
def clips(self) -> list[Clip]:
    return [self._main_window.clips_by_id[cid] for cid in self._clip_ids
            if cid in self._main_window.clips_by_id]
```

**Guard Flags (prevent duplicate handling):**
```python
def _on_analysis_finished(self, results):
    if self._analysis_finished_handled:
        return
    self._analysis_finished_handled = True
    # ... handle results
```

---
title: "feat: Redesign Sequence Tab UI"
type: feat
date: 2026-01-28
---

# feat: Redesign Sequence Tab UI

## Overview

Simplify the Sequence tab experience by removing the intermediate parameter view. The tab will have two clear states: **Cards** (empty/selection state) and **Timeline** (working state). Algorithm cards apply directly using selected clips from Analyze/Cut tabs, and a dedicated header row provides "redo" functionality.

## Problem Statement / Motivation

The current Sequence tab has three states (empty, card selection, parameter view) with a complex flow:
1. User selects algorithm card
2. Parameter panel appears with clip count, direction, preview strip
3. User configures and clicks "Apply"

This creates friction for the common case where users just want to quickly apply an algorithm. The parameter view adds cognitive overhead without proportional value—most users accept defaults.

**Pain points:**
- Too many clicks to apply a sequence
- Timeline hidden until deep in the flow
- Redundant remix combobox in timeline toolbar
- No direct way to "redo" with different algorithm once in timeline view

## Proposed Solution

### New Two-State Model

| State | When | Visible Components |
|-------|------|-------------------|
| `STATE_CARDS` | No clips on timeline | Card grid only (no timeline, no preview) |
| `STATE_TIMELINE` | Clips on timeline | Timeline + VideoPlayer + Thumbnail preview + Header row |

### User Flow

```
┌─────────────────────────────────────────────────────────────┐
│  STATE_CARDS (Initial / Empty)                              │
│  ┌─────────┐ ┌─────────┐                                    │
│  │ 🎨 Color│ │ ⏱ Dur   │   ← Click card to apply           │
│  └─────────┘ └─────────┘                                    │
│  ┌─────────┐ ┌─────────┐                                    │
│  │ 🎲 Shuff│ │ 📋 Seq  │   (Uses selected clips from        │
│  └─────────┘ └─────────┘    Analyze or Cut tab)             │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ User clicks card → Clips applied
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  STATE_TIMELINE                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Algorithm: [Color ▼]  [Clear Sequence]                  ││ ← NEW Header
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Video Player                               ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │         Timeline Preview (Thumbnails)                   ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Timeline Widget                            ││
│  │  [Play] [Export] [+ Track] [Zoom]                       ││ ← Remix combobox REMOVED
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                    │
                    │ User selects different algorithm from dropdown
                    │ OR clicks "Clear Sequence"
                    ▼
            Regenerate in place / Return to cards
```

### Key Behaviors

1. **Card Click → Immediate Apply**
   - Get selected clips from GUIState (prefer Analyze tab, fallback to Cut tab)
   - If no clips selected → Show error toast: "Select clips in Analyze or Cut tab first"
   - Generate sequence using all selected clips
   - Transition to STATE_TIMELINE

2. **Redo via Header Dropdown**
   - Dropdown shows: Color, Duration, Shuffle, Sequential
   - Selecting regenerates timeline in-place with same clips
   - No confirmation needed (fast iteration)

3. **Clear Sequence**
   - Button in header returns to STATE_CARDS
   - Clears timeline completely

4. **Tab Activation**
   - Preserve state across tab switches
   - If clips exist on timeline → STATE_TIMELINE
   - If no clips on timeline → STATE_CARDS

## Technical Considerations

### State Management (Critical)

Based on documented learnings from `pyside6-stacked-widget-programmatic-state-sync`:

```python
def _set_state(self, state: int):
    """Unified state setter - ALWAYS use this, never set index directly."""
    self._current_state = state
    self.state_stack.setCurrentIndex(state)

    # Update visibility
    if state == self.STATE_CARDS:
        self.header_widget.setVisible(False)
        self.content_widget.setVisible(False)  # Timeline + preview
    else:  # STATE_TIMELINE
        self.header_widget.setVisible(True)
        self.content_widget.setVisible(True)
```

**Agent tool compatibility**: All tools that modify sequence must call `_set_state()`:
- `add_to_sequence()` → If clips added, ensure STATE_TIMELINE
- `clear_sequence()` → Transition to STATE_CARDS
- `generate_and_apply()` → Ensure STATE_TIMELINE

### Widget Hierarchy Changes

**Current:**
```
state_stack
├── [0] no_clips_widget (EmptyStateWidget)
├── [1] card_grid (SortingCardGrid)
└── [2] parameter_view (QWidget)
content_widget
├── video_player
└── timeline
```

**Proposed:**
```
state_stack
├── [0] card_grid (SortingCardGrid)  # Cards only - STATE_CARDS
└── [1] timeline_view (QWidget)       # Everything else - STATE_TIMELINE
         ├── header_widget (NEW)
         │   ├── algorithm_dropdown (QComboBox)
         │   └── clear_btn (QPushButton)
         ├── video_player
         ├── timeline_preview (moved from parameter_view)
         └── timeline
```

### Clip Selection Source

Cards must read selected clips from GUIState:

```python
def _on_card_clicked(self, algorithm: str):
    # Get selected clips from GUI state
    selected_ids = self._gui_state.analyze_selected_ids or self._gui_state.cut_selected_ids

    if not selected_ids:
        self._show_error("Select clips in Analyze or Cut tab first")
        return

    # Get actual clip objects
    clips = [self.clips_by_id[cid] for cid in selected_ids if cid in self.clips_by_id]

    if not clips:
        self._show_error("Selected clips not found")
        return

    # Generate and apply
    self._generate_sequence(algorithm, clips)
    self._set_state(self.STATE_TIMELINE)
```

### Removed Components

- `SortingParameterPanel` - No longer needed
- `EmptyStateWidget` in sequence tab - Cards serve as empty state
- Remix combobox in `TimelineWidget` toolbar - Replaced by header dropdown

### Performance

- Sequence generation is fast (in-memory sorting)
- Thumbnail generation for TimelinePreview may need optimization for large clip counts
- Consider lazy-loading preview thumbnails if > 50 clips

## Acceptance Criteria

### Functional Requirements

- [ ] STATE_CARDS: Only card grid visible, no timeline or preview
- [ ] STATE_TIMELINE: Header + video player + thumbnail preview + timeline visible
- [ ] Card click generates sequence from Analyze/Cut selected clips
- [ ] Error shown if no clips selected when clicking card
- [ ] Header dropdown allows changing algorithm (regenerates in-place)
- [ ] "Clear Sequence" button returns to STATE_CARDS
- [ ] State preserved across tab switches
- [ ] Agent tools properly sync UI state after modifications

### Non-Functional Requirements

- [ ] Smooth state transitions (no flicker)
- [ ] Header dropdown matches application theme
- [ ] Remove all dead code from parameter view

## Dependencies & Risks

### Dependencies

- GUIState must track selected clips in Analyze and Cut tabs
- Verify GUIState already has `analyze_selected_ids` and `cut_selected_ids` (or add them)

### Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Agent tools don't update UI state | Timeline shows stale state | Audit all sequence-modifying tools |
| Color algorithm unavailable (no analysis) | User frustration | Disable card with tooltip |
| Large clip selections cause slow preview | Poor UX | Lazy-load thumbnails |

## Files to Modify

| File | Changes |
|------|---------|
| `ui/tabs/sequence_tab.py` | Major restructure: 2-state model, remove parameter view, add header |
| `ui/timeline/timeline_widget.py` | Remove remix combobox from toolbar |
| `ui/widgets/sorting_card_grid.py` | Update to emit click signal for immediate apply |
| `ui/widgets/sorting_card.py` | Minor: ensure disabled state works for unavailable algorithms |
| `ui/widgets/timeline_preview.py` | Move to be child of timeline_view |
| `core/chat_tools.py` | Update `get_sorting_state()` return values for new states |
| `core/gui_state.py` | Verify/add `analyze_selected_ids`, `cut_selected_ids` tracking |

## Files to Delete

| File | Reason |
|------|--------|
| `ui/widgets/sorting_parameter_panel.py` | Entire widget no longer needed |

## MVP

### ui/tabs/sequence_tab.py (key changes)

```python
class SequenceTab(BaseTab):
    # New state constants (2 states instead of 3)
    STATE_CARDS = 0      # Show card grid only
    STATE_TIMELINE = 1   # Show header + timeline + preview

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # State stack for cards vs timeline
        self.state_stack = QStackedWidget()

        # STATE_CARDS: Just the card grid
        self.card_grid = SortingCardGrid()
        self.card_grid.card_clicked.connect(self._on_card_clicked)
        self.state_stack.addWidget(self.card_grid)

        # STATE_TIMELINE: Header + content
        self.timeline_view = QWidget()
        timeline_layout = QVBoxLayout(self.timeline_view)

        # New header row
        self.header_widget = self._create_header()
        timeline_layout.addWidget(self.header_widget)

        # Video player
        self.video_player = VideoPlayer()
        timeline_layout.addWidget(self.video_player)

        # Timeline preview (moved from parameter view)
        self.timeline_preview = TimelinePreview()
        timeline_layout.addWidget(self.timeline_preview)

        # Timeline widget
        self.timeline = TimelineWidget()
        timeline_layout.addWidget(self.timeline)

        self.state_stack.addWidget(self.timeline_view)

        main_layout.addWidget(self.state_stack)

        # Start in cards state
        self._set_state(self.STATE_CARDS)

    def _create_header(self) -> QWidget:
        header = QWidget()
        layout = QHBoxLayout(header)

        layout.addWidget(QLabel("Algorithm:"))

        self.algorithm_dropdown = QComboBox()
        self.algorithm_dropdown.addItems(["Color", "Duration", "Shuffle", "Sequential"])
        self.algorithm_dropdown.currentTextChanged.connect(self._on_algorithm_changed)
        layout.addWidget(self.algorithm_dropdown)

        layout.addStretch()

        self.clear_btn = QPushButton("Clear Sequence")
        self.clear_btn.clicked.connect(self._on_clear_clicked)
        layout.addWidget(self.clear_btn)

        return header

    def _on_card_clicked(self, algorithm: str):
        """Handle card click - generate sequence from selected clips."""
        # Get selected clips from GUI state
        selected_ids = (
            self._gui_state.analyze_selected_ids
            or self._gui_state.cut_selected_ids
            or []
        )

        if not selected_ids:
            QMessageBox.warning(
                self,
                "No Clips Selected",
                "Select clips in the Analyze or Cut tab first."
            )
            return

        # Get clip objects
        clips = [self.clips_by_id.get(cid) for cid in selected_ids]
        clips = [c for c in clips if c is not None]

        if not clips:
            QMessageBox.warning(self, "Error", "Selected clips not found.")
            return

        # Generate and apply
        self._apply_algorithm(algorithm, clips)

    def _apply_algorithm(self, algorithm: str, clips: list):
        """Generate sequence and transition to timeline state."""
        from core.remix.shuffle import generate_sequence

        # Generate sorted clips
        sorted_clips = generate_sequence(
            clips=clips,
            sources=self._sources,
            algorithm=algorithm.lower(),
            clip_count=len(clips),
        )

        # Clear and populate timeline
        self.timeline.clear_timeline()
        for i, (clip, source) in enumerate(sorted_clips):
            self.timeline.add_clip(clip, source, track_index=0, start_frame=i * 100)

        # Update preview
        self.timeline_preview.set_clips(sorted_clips)

        # Update dropdown to show current algorithm
        self.algorithm_dropdown.blockSignals(True)
        self.algorithm_dropdown.setCurrentText(algorithm.capitalize())
        self.algorithm_dropdown.blockSignals(False)

        # Transition to timeline state
        self._set_state(self.STATE_TIMELINE)

    def _on_algorithm_changed(self, algorithm: str):
        """Handle algorithm dropdown change - regenerate in place."""
        # Use clips currently on timeline
        sequence = self.timeline.get_sequence()
        clip_ids = [sc.source_clip_id for sc in sequence.get_all_clips()]
        clips = [self.clips_by_id.get(cid) for cid in clip_ids]
        clips = [c for c in clips if c is not None]

        if clips:
            self._apply_algorithm(algorithm, clips)

    def _on_clear_clicked(self):
        """Clear sequence and return to cards."""
        self.timeline.clear_timeline()
        self.timeline_preview.clear()
        self._set_state(self.STATE_CARDS)

    def _set_state(self, state: int):
        """Unified state setter."""
        self._current_state = state
        self.state_stack.setCurrentIndex(state)
```

## References

### Internal References

- Sequence tab implementation: `ui/tabs/sequence_tab.py`
- QStackedWidget state sync learning: `~/.claude/skills/pyside6-stacked-widget-programmatic-state-sync/SKILL.md`
- Timeline sequence mismatch fix: `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`
- GUIState tracking: `core/gui_state.py`

### Related Work

- Previous state management fixes: commit `12abad4`

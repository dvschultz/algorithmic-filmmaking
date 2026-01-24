---
title: "refactor: Five-Tab Workflow UI (DaVinci Resolve-style)"
type: refactor
date: 2026-01-24
priority: high
---

# refactor: Five-Tab Workflow UI

## Overview

Restructure Scene Ripper from a single-window layout into a 5-tab workflow interface inspired by DaVinci Resolve. Each tab focuses on a specific stage of the video collage workflow, giving users more screen space and a clearer mental model.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Scene Ripper                                              [─][□][×]│
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┐               │
│  │ COLLECT │ ANALYZE │GENERATE │SEQUENCE │ RENDER  │               │
│  └─────────┴─────────┴─────────┴─────────┴─────────┘               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                      [ TAB CONTENT AREA ]                           │
│                                                                     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  [████████████████░░░░░░░░░░] 65% - Analyzing colors...            │
└─────────────────────────────────────────────────────────────────────┘
```

## Problem Statement

The current single-window layout crams everything into one view:
- Toolbar, clip browser, video player, and timeline compete for space
- Users can't focus on one task without distractions
- No clear workflow progression (import → analyze → arrange → export)
- Professional tools (DaVinci Resolve, Premiere) use tab/page-based workflows

## Proposed Solution

Create 5 dedicated tabs, each owning a stage of the workflow:

| Tab | Purpose | Key Components |
|-----|---------|----------------|
| **Collect** | Import/download videos | Import buttons, URL input, drop zone |
| **Analyze** | Detect scenes, browse clips | Sensitivity slider, ClipBrowser, VideoPlayer |
| **Generate** | Algorithmic remix | Stub: "Coming Soon" placeholder |
| **Sequence** | Arrange clips on timeline | TimelineWidget, VideoPlayer |
| **Render** | Export final video | Quality/resolution settings, export button |

## Technical Approach

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          MainWindow                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Shared State: current_source, clips, sequence, settings       │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Workers: Detection, Thumbnail, Download, Export, Color, Shot  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  ┌───────────────────────────┴───────────────────────────────────┐  │
│  │                      QTabWidget                                │  │
│  │  ┌──────────┬──────────┬──────────┬──────────┬──────────┐    │  │
│  │  │ Collect  │ Analyze  │ Generate │ Sequence │  Render  │    │  │
│  │  │   Tab    │   Tab    │   Tab    │   Tab    │   Tab    │    │  │
│  │  └──────────┴──────────┴──────────┴──────────┴──────────┘    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                   Global Progress Bar                          │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Multi-source support? | No (single source) | Matches current behavior, avoids scope creep |
| State sharing? | Signal/slot via MainWindow | Matches existing patterns, documented learnings |
| Auto-navigation? | Only Import → Analyze | Minimizes disorientation |
| Progress bar location? | Global (bottom of window) | Visible regardless of current tab |
| Generate tab content? | Stub "Coming Soon" | User specified this |
| VideoPlayer visibility? | Analyze + Sequence only | Where preview is needed |
| Worker ownership? | MainWindow (unchanged) | Avoid complex refactor |

### Files to Create

```
ui/tabs/
├── __init__.py
├── base_tab.py           # Base class with common functionality
├── collect_tab.py        # Import/download interface
├── analyze_tab.py        # Detection + ClipBrowser + VideoPlayer
├── generate_tab.py       # Placeholder stub
├── sequence_tab.py       # Timeline + VideoPlayer
└── render_tab.py         # Export settings + button
```

### Files to Modify

| File | Changes |
|------|---------|
| `ui/main_window.py` | Replace splitter layout with QTabWidget, keep workers |
| `ui/clip_browser.py` | Minor: remove from main_window, embed in analyze_tab |
| `ui/video_player.py` | No changes (reused in multiple tabs) |
| `ui/timeline/timeline_widget.py` | Remove remix controls (move to generate_tab later) |

### Data Model

No changes to existing models. Shared state stays in MainWindow:

```python
# MainWindow instance variables (unchanged)
self.current_source: Optional[Source] = None
self.clips: list[Clip] = []
self.clips_by_id: dict[str, Clip] = {}
self.sequence: Sequence = Sequence(...)
self.settings: Settings = load_settings()
```

## Implementation Phases

### Phase 1: Tab Infrastructure

**Goal**: Create tab widget with stub tabs, keep all functionality working

**Deliverables**:
- [x] Create `ui/tabs/base_tab.py` with shared signals/methods
- [x] Create `ui/tabs/__init__.py` exporting all tab classes
- [x] Create 5 stub tab classes (empty QWidget subclasses)
- [x] Modify `MainWindow._setup_ui()` to use QTabWidget
- [x] Move global progress bar below tabs
- [x] Verify app launches with new structure

**Files**:
```python
# ui/tabs/base_tab.py
class BaseTab(QWidget):
    """Base class for workflow tabs."""

    # Common signals
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Override in subclasses."""
        pass

    def on_tab_activated(self):
        """Called when tab becomes visible."""
        pass

    def on_tab_deactivated(self):
        """Called when switching away from tab."""
        pass
```

### Phase 2: Collect Tab

**Goal**: Move import functionality to dedicated tab

**Deliverables**:
- [x] Create `CollectTab` with import buttons and drop zone
- [x] Move "Import Video" button from toolbar
- [x] Move "Import URL" button and dialog from toolbar
- [x] Add drag-drop visual feedback
- [x] Show "No videos imported" empty state
- [x] After import, auto-switch to Analyze tab

**UI Mockup**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                           COLLECT                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                                                              │   │
│   │              ┌──────────────────────────┐                   │   │
│   │              │                          │                   │   │
│   │              │     📁 Drop video here   │                   │   │
│   │              │                          │                   │   │
│   │              │   or click to browse     │                   │   │
│   │              │                          │                   │   │
│   │              └──────────────────────────┘                   │   │
│   │                                                              │   │
│   │                         — or —                               │   │
│   │                                                              │   │
│   │              [  Import from URL...  ]                        │   │
│   │                                                              │   │
│   │              Supported: YouTube, Vimeo                       │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 3: Analyze Tab

**Goal**: Move detection controls and clip browser to dedicated tab

**Deliverables**:
- [x] Create `AnalyzeTab` with sensitivity slider and detect button
- [x] Embed existing `ClipBrowser` component
- [x] Embed `VideoPlayer` for clip preview
- [x] Wire `clip_selected` signal to video player seek
- [x] Wire `clip_double_clicked` to play clip range
- [x] Show "Import a video first" when no source
- [x] Show "Click Detect to find scenes" after import

**UI Mockup**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                           ANALYZE                                    │
├─────────────────────────────────────────────────────────────────────┤
│  Sensitivity: [──●──────] 3.0   [Detect Scenes]                     │
├───────────────────────────────────┬─────────────────────────────────┤
│                                   │                                  │
│   ┌─────┐ ┌─────┐ ┌─────┐       │    ┌───────────────────────┐     │
│   │ 001 │ │ 002 │ │ 003 │       │    │                       │     │
│   │ 2.3s│ │ 4.1s│ │ 1.8s│       │    │     VIDEO PREVIEW     │     │
│   └─────┘ └─────┘ └─────┘       │    │                       │     │
│   ┌─────┐ ┌─────┐ ┌─────┐       │    │                       │     │
│   │ 004 │ │ 005 │ │ 006 │       │    └───────────────────────┘     │
│   │ 3.2s│ │ 2.0s│ │ 5.1s│       │     00:12 / 02:34  [▶]           │
│   └─────┘ └─────┘ └─────┘       │                                  │
│                                   │  Sort: [Color ▼] [Shot Type ▼]  │
│   Filter: [All ▼]                │                                  │
│                                   │                                  │
└───────────────────────────────────┴─────────────────────────────────┘
```

### Phase 4: Generate Tab (Stub)

**Goal**: Create placeholder for future algorithmic remix features

**Deliverables**:
- [x] Create `GenerateTab` with "Coming Soon" message
- [x] Add brief description of planned features
- [ ] Disable tab when no clips detected

**UI Mockup**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                          GENERATE                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                                                                      │
│                        🚧 Coming Soon 🚧                             │
│                                                                      │
│            Algorithmic remix features will appear here.              │
│                                                                      │
│                    Planned capabilities:                             │
│                    • Shuffle with constraints                        │
│                    • Similarity chaining                             │
│                    • Beat-synced editing                             │
│                    • Color-based sequencing                          │
│                                                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 5: Sequence Tab

**Goal**: Move timeline and playback to dedicated tab

**Deliverables**:
- [x] Create `SequenceTab` with timeline and video player
- [x] Remove remix controls from TimelineWidget (for now)
- [x] Keep drag-from-Analyze functionality (cross-tab drag-drop)
- [x] Wire playback signals between timeline and player
- [x] Show "Drag clips from Analyze tab" when timeline empty

**UI Mockup**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                          SEQUENCE                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                                                              │   │
│   │                      VIDEO PREVIEW                           │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│    00:00:12 / 00:02:34    [⏮] [▶] [⏭]    ──●────────────────       │
├─────────────────────────────────────────────────────────────────────┤
│  0:00    0:10    0:20    0:30    0:40    0:50    1:00               │
│  │────────│────────│────────│────────│────────│────────│            │
│  ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │V1│ [Clip 001 ▓▓▓] [Clip 003] [Clip 002 ▓▓▓▓▓]                │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  [+ Track]                                          [Clear Timeline] │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 6: Render Tab

**Goal**: Create export configuration and rendering interface

**Deliverables**:
- [x] Create `RenderTab` with export settings
- [x] Add quality preset selector (High/Medium/Low)
- [x] Add resolution selector (Original/1080p/720p/480p)
- [x] Add "Export Sequence" button
- [x] Add "Export Selected Clips" button
- [x] Add "Export Dataset (JSON)" button
- [x] Show export progress in this tab
- [x] Disable when timeline is empty

**UI Mockup**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                           RENDER                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─ Export Settings ─────────────────────────────────────────────┐ │
│   │                                                                │ │
│   │  Quality:     [▼ Medium (balanced)    ]                       │ │
│   │                                                                │ │
│   │  Resolution:  [▼ Original             ]                       │ │
│   │                                                                │ │
│   │  Frame Rate:  [▼ Original             ]                       │ │
│   │                                                                │ │
│   └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│   ┌─ Sequence ────────────────────────────────────────────────────┐ │
│   │                                                                │ │
│   │  Duration: 00:02:34    Clips: 12                              │ │
│   │                                                                │ │
│   │                    [ Export Sequence ]                         │ │
│   │                                                                │ │
│   └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│   ┌─ Other Exports ───────────────────────────────────────────────┐ │
│   │                                                                │ │
│   │  [ Export Selected Clips ]    [ Export Dataset (JSON) ]       │ │
│   │                                                                │ │
│   └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 7: Signal Routing & Polish

**Goal**: Wire all cross-tab communication and polish UX

**Deliverables**:
- [ ] Connect Collect import → refresh Analyze tab
- [ ] Connect detection complete → enable Generate/Sequence tabs
- [ ] Connect Analyze clip drag → Sequence timeline add
- [ ] Add tab enable/disable based on state
- [ ] Add keyboard shortcuts for tab switching (Ctrl+1-5)
- [ ] Test tab switching during background operations
- [ ] Apply QThread guard pattern from documented learnings

## Acceptance Criteria

### Functional Requirements

- [ ] App opens with 5 tabs visible at top
- [ ] Collect tab: Can import local video via drag-drop or button
- [ ] Collect tab: Can import from YouTube/Vimeo URL
- [ ] Analyze tab: Can adjust sensitivity and detect scenes
- [ ] Analyze tab: Can browse detected clips with thumbnails
- [ ] Analyze tab: Can preview clips in video player
- [ ] Generate tab: Shows "Coming Soon" placeholder
- [ ] Sequence tab: Can drag clips to timeline
- [ ] Sequence tab: Can play timeline sequence
- [ ] Render tab: Can configure and export sequence
- [ ] Progress bar visible at bottom regardless of tab
- [ ] Tab switching works during background operations

### Non-Functional Requirements

- [ ] No regression in existing functionality
- [ ] App startup time unchanged (<3s)
- [ ] Memory usage unchanged
- [ ] All documented QThread patterns applied

## Dependencies & Risks

### Dependencies

- PySide6 QTabWidget (already available)
- Existing components (ClipBrowser, VideoPlayer, TimelineWidget)

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Signal routing complexity | Medium | Medium | Use documented patterns, add guards |
| Cross-tab drag-drop issues | Medium | Medium | Test thoroughly, fallback to copy-paste |
| Worker state confusion | Low | High | Keep workers in MainWindow, document state |
| UI performance with tabs | Low | Low | QTabWidget is lightweight |

## Success Metrics

| Metric | Target |
|--------|--------|
| All existing features work | 100% parity |
| Tab switch time | <100ms |
| No duplicate signals | 0 (verified by logs) |
| No orphaned workers | 0 (verified by shutdown) |

## References

### Internal References

- Current layout: `ui/main_window.py:340-380`
- Tab pattern: `ui/settings_dialog.py:116-124`
- QThread guards: `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`
- State duplication: `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`
- Component init: `docs/solutions/runtime-errors/qgraphicsscene-missing-items-20260124.md`

### External References

- [DaVinci Resolve Pages](https://www.blackmagicdesign.com/products/davinciresolve) - UI inspiration
- [PySide6 QTabWidget](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QTabWidget.html)

---

*Generated: 2026-01-24*

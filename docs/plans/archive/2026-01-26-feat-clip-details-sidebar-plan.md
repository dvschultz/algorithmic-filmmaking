---
title: "feat: Add clip details sidebar"
type: feat
date: 2026-01-26
---

# feat: Add Clip Details Sidebar

## Overview

A dismissable sidebar that opens on the left side of the app to display detailed information about a selected clip. Includes video preview, metadata, and analysis results.

## Problem Statement / Motivation

Currently, clip information is scattered:
- Thumbnails show limited metadata (duration badge, color swatches, shot type)
- Detailed analysis data (full transcript, all colors) requires navigating elsewhere
- No unified view of all clip information in one place

Users need a quick way to inspect clip details without leaving their current workflow.

## Proposed Solution

Create a `ClipDetailsSidebar` as a `QDockWidget` (following the `ChatPanel` pattern) that:

1. **Opens via multiple triggers:**
   - Right-click context menu → "View Details"
   - Double-click on clip card
   - Keyboard shortcut (Enter or 'i' when clip selected)

2. **Displays clip information:**
   - Video preview at top (using embedded `VideoPlayer`)
   - Clip title (source filename + timecode range)
   - Basic metadata (duration, frames, resolution)
   - Analysis data sections (colors, shot type, transcript)

3. **Updates dynamically:**
   - When a different clip is selected, sidebar content updates
   - Single sidebar instance (not per-clip)

4. **Dismissable:**
   - X button in header
   - Escape key
   - Toggle via View menu

## Technical Approach

### Architecture

```
ui/
├── clip_details_sidebar.py    # New: QDockWidget with clip info
└── widgets/
    └── color_palette.py       # New: Reusable color display widget
```

### Key Components

**ClipDetailsSidebar (QDockWidget)**
- Header: "Clip Details" title + close button
- Video preview section (embedded VideoPlayer, 16:9 aspect)
- Metadata section (title, duration, frame range, resolution)
- Analysis section (collapsible groups for colors, shot type, transcript)
- Scroll area for overflow content

**State Management**
- Sidebar references clip data from source of truth (don't duplicate state)
- Listen to clip selection signals from ClipBrowser/SortingCard
- Property delegation pattern per documented learnings

### Signal Flow

```
ClipThumbnail.double_clicked ──┐
SortingCard.double_clicked ────┼──► MainWindow._on_clip_details_requested
Context menu "View Details" ───┤                    │
Keyboard shortcut (Enter/i) ───┘                    ▼
                                        ClipDetailsSidebar.show_clip(clip, source)
                                                    │
                                                    ▼
                                        VideoPlayer.load_video()
                                        Update metadata labels
                                        Update analysis sections
```

### Integration Points

| Location | Change |
|----------|--------|
| `ui/main_window.py` | Add dock widget, connect signals |
| `ui/clip_browser.py` | Add context menu option, connect double-click |
| `ui/widgets/sorting_card.py` | Add context menu option, connect double-click |
| `ui/tabs/analyze_tab.py` | Connect to sidebar signals |
| `ui/tabs/cut_tab.py` | Connect to sidebar signals |
| `ui/tabs/sequence_tab.py` | Connect to sidebar signals |

## Acceptance Criteria

### Functional Requirements

- [ ] Sidebar opens via right-click context menu "View Details"
- [ ] Sidebar opens via double-click on clip card (ClipThumbnail, SortingCard)
- [ ] Sidebar opens via keyboard (Enter or 'i' when clip selected)
- [ ] Video preview plays clip range (start_frame to end_frame)
- [ ] Displays clip title as "source_filename - HH:MM:SS"
- [ ] Displays duration, frame range, source resolution
- [ ] Displays dominant colors as swatches (if analyzed)
- [ ] Displays shot type badge (if analyzed)
- [ ] Displays transcript text (if transcribed)
- [ ] Sidebar content updates when different clip is selected
- [ ] Dismissable via X button
- [ ] Dismissable via Escape key
- [ ] Sidebar persists across tab changes

### Non-Functional Requirements

- [ ] Uses theme colors (light/dark mode support)
- [ ] Sidebar width: 350px default, resizable via dock widget
- [ ] Video preview maintains aspect ratio
- [ ] Empty states for missing analysis data ("Not analyzed")
- [ ] Error state for missing source file

## Implementation Phases

### Phase 1: Core Sidebar Structure

**Files:**
- `ui/clip_details_sidebar.py` - New file

**Tasks:**
- [ ] Create `ClipDetailsSidebar` class extending `QDockWidget`
- [ ] Add header with title and close button
- [ ] Add scroll area for content
- [ ] Implement `show_clip(clip: Clip, source: Source)` method
- [ ] Add to `MainWindow` as left-docked widget
- [ ] Add View menu toggle "Clip Details" (Ctrl+D)

### Phase 2: Metadata Display

**Tasks:**
- [ ] Add video preview section (embed `VideoPlayer` component)
- [ ] Add title label (computed from source filename + timecode)
- [ ] Add metadata labels (duration, frames, resolution, fps)
- [ ] Apply theme styling

### Phase 3: Analysis Display

**Tasks:**
- [ ] Add collapsible "Colors" section with color swatches
- [ ] Add "Shot Type" badge display
- [ ] Add collapsible "Transcript" section with text area
- [ ] Handle empty states ("Not analyzed" / "No transcript")

### Phase 4: Trigger Integration

**Tasks:**
- [ ] Add context menu to `ClipThumbnail` with "View Details"
- [ ] Add context menu to `SortingCard` with "View Details"
- [ ] Connect double-click signals to open sidebar
- [ ] Add keyboard shortcut handler (Enter/i on selected clip)
- [ ] Wire up signals in `MainWindow`

### Phase 5: Polish

**Tasks:**
- [ ] Handle missing source file gracefully
- [ ] Add Escape key to dismiss
- [ ] Ensure sidebar updates on clip selection change
- [ ] Test across all tabs (Cut, Analyze, Sequence)
- [ ] Theme change refresh

## Gotchas from Documented Learnings

### State Management (CRITICAL)

From `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`:
- Sidebar must be a VIEW of clip state, not maintain its own copy
- Use property delegation to access clip from source of truth
- Never duplicate clip/source data

```python
# WRONG
class ClipDetailsSidebar(QDockWidget):
    def __init__(self):
        self._current_clip = None  # BAD: duplicate state

# RIGHT
class ClipDetailsSidebar(QDockWidget):
    def show_clip(self, clip: Clip, source: Source):
        # Store references only, don't copy
        self._clip_ref = clip
        self._source_ref = source
```

### Signal Handler Guards (CRITICAL)

From `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`:
- Add guard flags to prevent duplicate signal handling
- Use `Qt.UniqueConnection` when connecting signals
- Use `@Slot()` decorator on all handlers

```python
@Slot(str)
def _on_clip_selected(self, clip_id: str):
    if self._loading:
        return  # Guard against duplicate delivery
    self._loading = True
    # ... load clip data ...
    self._loading = False
```

### Source ID Lookups (MEDIUM)

From `docs/solutions/ui-bugs/pyside6-thumbnail-source-id-mismatch.md`:
- Pass clip objects directly, don't rely on ID lookups
- If lookup needed, use source_id + frame range as key
- Log warnings on empty lookup results

## References

### Internal Code Patterns

| Pattern | File | Lines |
|---------|------|-------|
| QDockWidget sidebar | `ui/chat_panel.py` | 51-673 |
| Video player embed | `ui/video_player.py` | 22-162 |
| Clip model | `models/clip.py` | 114-209 |
| Source model | `models/clip.py` | 13-111 |
| Color swatches | `ui/clip_browser.py` | 31-67 |
| Theme system | `ui/theme.py` | 1-608 |
| Panel layout pattern | `ui/widgets/sorting_parameter_panel.py` | 77-320 |

### Documented Learnings

- `docs/solutions/ui-bugs/pyside6-thumbnail-source-id-mismatch.md`
- `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`
- `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`

## MVP Implementation

### ui/clip_details_sidebar.py

```python
"""Clip details sidebar widget."""

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QScrollArea, QFrame, QSizePolicy
)

from models.clip import Clip, Source
from ui.theme import theme
from ui.video_player import VideoPlayer


class ClipDetailsSidebar(QDockWidget):
    """Sidebar displaying detailed clip information."""

    def __init__(self, parent=None):
        super().__init__("Clip Details", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setMinimumWidth(300)
        self.setMaximumWidth(450)

        # State references (not copies)
        self._clip_ref: Clip | None = None
        self._source_ref: Source | None = None
        self._loading = False  # Guard flag

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Build the sidebar UI."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Video preview
        self.video_player = VideoPlayer()
        self.video_player.setMaximumHeight(200)
        layout.addWidget(self.video_player)

        # Scroll area for metadata
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(16)

        # Title section
        self.title_label = QLabel("No clip selected")
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {theme().text_primary};")
        content_layout.addWidget(self.title_label)

        # Metadata section
        self.metadata_label = QLabel("")
        self.metadata_label.setWordWrap(True)
        self.metadata_label.setStyleSheet(f"color: {theme().text_secondary};")
        content_layout.addWidget(self.metadata_label)

        # Colors section
        self.colors_label = QLabel("Colors")
        self.colors_label.setStyleSheet(f"font-weight: bold; color: {theme().text_primary};")
        content_layout.addWidget(self.colors_label)

        self.color_swatches = QWidget()
        self.color_swatches_layout = QHBoxLayout(self.color_swatches)
        self.color_swatches_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.color_swatches)

        # Shot type section
        self.shot_type_label = QLabel("")
        self.shot_type_label.setStyleSheet(f"color: {theme().text_secondary};")
        content_layout.addWidget(self.shot_type_label)

        # Transcript section
        self.transcript_label = QLabel("Transcript")
        self.transcript_label.setStyleSheet(f"font-weight: bold; color: {theme().text_primary};")
        content_layout.addWidget(self.transcript_label)

        self.transcript_text = QLabel("")
        self.transcript_text.setWordWrap(True)
        self.transcript_text.setStyleSheet(f"color: {theme().text_secondary};")
        content_layout.addWidget(self.transcript_text)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        self.setWidget(container)

    def _connect_signals(self):
        """Connect theme change signal."""
        theme().changed.connect(self._refresh_theme)

    @Slot()
    def _refresh_theme(self):
        """Update colors on theme change."""
        self.title_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {theme().text_primary};")
        self.metadata_label.setStyleSheet(f"color: {theme().text_secondary};")
        self.colors_label.setStyleSheet(f"font-weight: bold; color: {theme().text_primary};")
        self.shot_type_label.setStyleSheet(f"color: {theme().text_secondary};")
        self.transcript_label.setStyleSheet(f"font-weight: bold; color: {theme().text_primary};")
        self.transcript_text.setStyleSheet(f"color: {theme().text_secondary};")

    @Slot(object, object)
    def show_clip(self, clip: Clip, source: Source):
        """Display details for the given clip."""
        if self._loading:
            return  # Guard against duplicate calls
        self._loading = True

        self._clip_ref = clip
        self._source_ref = source

        # Title: filename - timecode
        start_time = clip.start_time(source.fps)
        title = f"{source.filename} - {self._format_time(start_time)}"
        self.title_label.setText(title)

        # Metadata
        duration = clip.duration_seconds(source.fps)
        metadata = (
            f"Duration: {self._format_time(duration)}\n"
            f"Frames: {clip.start_frame} - {clip.end_frame}\n"
            f"Resolution: {source.width}x{source.height}\n"
            f"FPS: {source.fps:.2f}"
        )
        self.metadata_label.setText(metadata)

        # Colors
        self._update_colors(clip.dominant_colors)

        # Shot type
        if clip.shot_type:
            self.shot_type_label.setText(f"Shot Type: {clip.shot_type}")
            self.shot_type_label.show()
        else:
            self.shot_type_label.setText("Shot Type: Not analyzed")

        # Transcript
        if clip.transcript:
            self.transcript_text.setText(clip.get_transcript_text())
            self.transcript_label.show()
            self.transcript_text.show()
        else:
            self.transcript_text.setText("No transcript available")

        # Load video
        if source.file_path.exists():
            self.video_player.load_video(source.file_path)
            self.video_player.seek_to(start_time)
        else:
            # Handle missing file
            self.video_player.clear()

        self.show()
        self._loading = False

    def _update_colors(self, colors: list[tuple[int, int, int]] | None):
        """Update the color swatches display."""
        # Clear existing swatches
        while self.color_swatches_layout.count():
            item = self.color_swatches_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not colors:
            no_colors = QLabel("Not analyzed")
            no_colors.setStyleSheet(f"color: {theme().text_muted};")
            self.color_swatches_layout.addWidget(no_colors)
            return

        for r, g, b in colors[:5]:
            swatch = QFrame()
            swatch.setFixedSize(32, 32)
            swatch.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border-radius: 4px;")
            self.color_swatches_layout.addWidget(swatch)

        self.color_swatches_layout.addStretch()

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"
```

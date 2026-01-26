---
title: "feat: Editable Clip Details Sidebar"
type: feat
date: 2026-01-26
---

# feat: Editable Clip Details Sidebar

## Overview

Make the clip details sidebar editable so users can modify clip attributes directly. Users will be able to edit:
- **Clip name** (custom name, falling back to auto-generated if empty)
- **Shot type** (dropdown with predefined categories)
- **Transcript** (multi-line text area)

Changes save automatically on blur (clicking away from field).

## Problem Statement / Motivation

Currently, the clip details sidebar is read-only. Users cannot:
- Give clips meaningful names for easier identification
- Correct or override AI-detected shot types
- Edit or correct transcription errors

This limits the usefulness of the metadata and forces users to accept auto-generated values.

## Proposed Solution

Transform specific fields in the clip details sidebar from read-only labels to inline-editable fields:

1. **Clip Name**: Click to edit inline, auto-save on blur/Enter
2. **Shot Type**: Click to open dropdown, save on selection
3. **Transcript**: Click to expand multi-line text area, save on blur

### What IS Editable

| Field | Input Type | Save Trigger | Notes |
|-------|------------|--------------|-------|
| Clip name | Inline text input | Blur or Enter | Falls back to auto-generated if empty |
| Shot type | Dropdown (QComboBox) | Selection | Categories: Wide, Medium, Close-up, Extreme CU |
| Transcript | Multi-line text area | Blur | Enter inserts newline |

### What is NOT Editable

| Field | Reason |
|-------|--------|
| Duration | Derived from video frames |
| Frames | Derived from scene detection |
| Resolution | Derived from source video |
| FPS | Derived from source video |
| Dominant Colors | Skip for now (future feature) |
| Source filename | Read-only reference |

## Technical Approach

### Architecture

```
ui/clip_details_sidebar.py
├── ClipDetailsSidebar (QDockWidget)
│   ├── video_player (VideoPlayer) - unchanged
│   ├── EditableLabel - NEW: name editing
│   ├── metadata_label (QLabel) - unchanged (read-only fields)
│   ├── ShotTypeDropdown - NEW: shot type editing
│   ├── EditableTextArea - NEW: transcript editing
│   └── color_swatches - unchanged
│
models/clip.py
├── Clip (dataclass)
│   ├── name: str = "" - NEW FIELD
│   └── (existing fields)
```

### Key Components

**EditableLabel Widget (New)**
- Dual-widget approach: QLabel (display) + QLineEdit (edit)
- Click to enter edit mode
- Auto-save on blur or Enter
- Follows existing PlanStepWidget pattern from `ui/chat_widgets.py:654`

**ShotTypeDropdown Widget (New)**
- QComboBox with predefined shot types
- Options: Wide, Medium, Close-up, Extreme CU, (Not set)
- Save immediately on selection change

**EditableTextArea Widget (New)**
- QLabel (display) + QTextEdit (edit)
- Click to enter edit mode
- Enter inserts newline, blur saves
- Escape cancels edit

### Signal Flow

```
User edits field
       ↓
Widget emits value_changed signal
       ↓
ClipDetailsSidebar._on_field_changed(field, value)
       ↓
Update clip reference (single source of truth)
       ↓
Call project.update_clips([clip]) to notify observers
       ↓
ProjectSignalAdapter emits clips_updated signal
       ↓
Other views (clip browser, sequence) refresh as needed
```

### Integration Points

| Location | Change |
|----------|--------|
| `models/clip.py` | Add `name: str = ""` field to Clip dataclass |
| `ui/clip_details_sidebar.py` | Replace QLabels with editable widgets for name, shot type, transcript |
| `ui/clip_browser.py` | Listen to `clips_updated` signal to refresh display |
| `core/project.py` | Already has `update_clips()` method - no changes needed |

## Acceptance Criteria

### Functional Requirements

- [x] Clicking on clip name field enters edit mode (text input)
- [x] Clip name saves on blur or Enter key
- [x] Empty clip name falls back to auto-generated name (source filename + timecode)
- [x] Clicking shot type field opens dropdown with options
- [x] Shot type dropdown has options: Wide, Medium, Close-up, Extreme CU, (Not set)
- [x] Selecting shot type saves immediately
- [x] Clicking transcript segment text enters edit mode (per-segment)
- [x] Transcript segment saves on blur
- [x] Escape key cancels edit and reverts to previous value
- [x] Non-editable fields (duration, frames, resolution, FPS) do not respond to clicks
- [x] Changes persist to project (saved with project file)

### Multi-Selection Behavior

- [x] When multiple clips selected, sidebar shows "X clips selected"
- [x] Editing is disabled when multiple clips selected
- [x] Switching to single selection re-enables editing

### Visual Requirements

- [x] Editable fields show hover state (cursor change, subtle highlight)
- [x] Edit mode shows clear visual distinction (border, background change)
- [x] Non-editable fields appear visually different (no hover state)

### Non-Functional Requirements

- [x] Signal handlers have guard flags to prevent duplicate execution
- [x] Clip reference follows single source of truth pattern (no duplicate state)
- [x] Edit saves are debounced/guarded to prevent rapid-fire saves
- [x] Theme changes update editable field styles correctly

## Implementation Phases

### Phase 1: Model Update

**Files:**
- `models/clip.py`

**Tasks:**
- [x] Add `name: str = ""` field to Clip dataclass
- [x] Update `to_dict()` to include name field
- [x] Update `from_dict()` to load name field (with default for backwards compatibility)
- [x] Add `display_name` property that returns name or auto-generated fallback

### Phase 2: EditableLabel Widget

**Files:**
- `ui/widgets/editable_label.py` (NEW)

**Tasks:**
- [x] Create EditableLabel class with dual-widget approach (QLabel + QLineEdit)
- [x] Implement click-to-edit behavior
- [x] Implement auto-save on blur/Enter
- [x] Implement Escape to cancel
- [x] Add `value_changed` signal
- [x] Add hover state styling
- [x] Add edit mode styling

### Phase 3: ShotTypeDropdown Widget

**Files:**
- `ui/widgets/shot_type_dropdown.py` (NEW)

**Tasks:**
- [x] Create ShotTypeDropdown class extending QComboBox
- [x] Populate with shot type options from `core/analysis/shots.py`
- [x] Add "(Not set)" option for null/unset state
- [x] Implement immediate save on selection change
- [x] Add `value_changed` signal
- [x] Style to match theme

### Phase 4: EditableTextArea Widget

**Files:**
- `ui/widgets/editable_text_area.py` (NEW)

**Tasks:**
- [x] Create EditableTranscriptWidget class with per-segment editing
- [x] Display timestamps (read-only) with editable text for each segment
- [x] Click segment text to edit, blur saves
- [x] Implement Escape to cancel
- [x] Add `segments_changed` signal
- [x] Handle multiple segments with scrolling

### Phase 5: Integrate into Sidebar

**Files:**
- `ui/clip_details_sidebar.py`

**Tasks:**
- [x] Replace title QLabel with EditableLabel for clip name
- [x] Replace shot type QLabel with ShotTypeDropdown
- [x] Replace transcript QLabel with EditableTranscriptWidget
- [x] Connect value_changed signals to handlers
- [x] Implement `_on_name_changed()` handler with guard flag
- [x] Implement `_on_shot_type_changed()` handler with guard flag
- [x] Implement `_on_transcript_changed()` handler with guard flag
- [x] Emit `clip_edited` signal for parent to update project
- [x] Handle multi-selection (disable editing, show count)

### Phase 6: View Synchronization

**Files:**
- `ui/clip_browser.py`
- `ui/main_window.py`

**Tasks:**
- [x] Connect ClipBrowser to `clips_updated` signal via main_window
- [x] Added `update_clips()` method to ClipBrowser for refreshing thumbnails
- [x] Connected ProjectSignalAdapter.clips_updated to forward to clip browsers

## Gotchas from Documented Learnings

### Signal Handler Guards (CRITICAL)

From `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`:

```python
# WRONG - no guard, may fire multiple times
@Slot(str)
def _on_name_changed(self, text: str):
    self._clip_ref.name = text
    self.project.update_clips([self._clip_ref])

# RIGHT - guard prevents duplicate execution
@Slot(str)
def _on_name_changed(self, text: str):
    if self._name_change_in_progress:
        return
    self._name_change_in_progress = True

    self._clip_ref.name = text
    self.project.update_clips([self._clip_ref])

    self._name_change_in_progress = False
```

### Single Source of Truth (CRITICAL)

From `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`:

- Sidebar stores `self._clip_ref` - a REFERENCE, not a copy
- All edits modify the reference directly
- Never duplicate clip attributes in sidebar's own state
- Use `@property` to delegate if needed

### Block Signals During Programmatic Updates

When updating widget values programmatically (e.g., when showing a different clip), block signals to prevent unwanted saves:

```python
def show_clip(self, clip: Clip, source: Source):
    # Block signals while updating UI
    self.name_edit.blockSignals(True)
    self.shot_type_dropdown.blockSignals(True)
    self.transcript_edit.blockSignals(True)

    # Update widget values
    self.name_edit.setText(clip.display_name)
    self.shot_type_dropdown.set_value(clip.shot_type)
    self.transcript_edit.setText(clip.get_transcript_text())

    # Unblock signals
    self.name_edit.blockSignals(False)
    self.shot_type_dropdown.blockSignals(False)
    self.transcript_edit.blockSignals(False)
```

## References

### Internal Code Patterns

| Pattern | File | Lines |
|---------|------|-------|
| Inline editing (dual widget) | `ui/chat_widgets.py` | 654-876 (PlanStepWidget) |
| Clip model | `models/clip.py` | 114-209 |
| Shot type categories | `core/analysis/shots.py` | 1-20 |
| Project update signal | `core/project.py` | update_clips() method |
| Signal adapter | `ui/project_adapter.py` | clips_updated signal |
| ClipDetailsSidebar | `ui/clip_details_sidebar.py` | Full file |

### Documented Learnings

- `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md` - Signal guard pattern
- `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md` - Single source of truth
- `docs/solutions/ui-bugs/pyside6-thumbnail-source-id-mismatch.md` - ID synchronization

## MVP Implementation Sketch

### models/clip.py - Add name field

```python
@dataclass
class Clip:
    id: str
    source_id: str
    start_frame: int
    end_frame: int
    name: str = ""  # NEW: custom clip name
    thumbnail_path: Optional[Path] = None
    dominant_colors: Optional[list[tuple[int, int, int]]] = None
    shot_type: Optional[str] = None
    transcript: Optional[list["TranscriptSegment"]] = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def display_name(self) -> str:
        """Return custom name or auto-generated fallback."""
        return self.name if self.name else None  # Caller provides fallback
```

### ui/widgets/editable_label.py

```python
"""Inline editable label widget."""

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QLineEdit

from ui.theme import theme


class EditableLabel(QWidget):
    """Label that can be clicked to edit inline."""

    value_changed = Signal(str)

    def __init__(self, text: str = "", placeholder: str = "", parent=None):
        super().__init__(parent)
        self._text = text
        self._placeholder = placeholder
        self._is_editing = False

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Display label
        self.label = QLabel(self._text or self._placeholder)
        self.label.setCursor(Qt.PointingHandCursor)
        self.label.mousePressEvent = self._start_editing
        layout.addWidget(self.label)

        # Edit field (hidden initially)
        self.edit = QLineEdit(self._text)
        self.edit.setPlaceholderText(self._placeholder)
        self.edit.returnPressed.connect(self._finish_editing)
        self.edit.hide()
        layout.addWidget(self.edit)

    def _connect_signals(self):
        pass  # Signals connected in _setup_ui

    def _start_editing(self, event):
        if self._is_editing:
            return
        self._is_editing = True

        self.label.hide()
        self.edit.setText(self._text)
        self.edit.show()
        self.edit.setFocus()
        self.edit.selectAll()

    def _finish_editing(self):
        if not self._is_editing:
            return

        new_text = self.edit.text().strip()
        if new_text != self._text:
            self._text = new_text
            self.label.setText(new_text or self._placeholder)
            self.value_changed.emit(new_text)

        self.edit.hide()
        self.label.show()
        self._is_editing = False

    def focusOutEvent(self, event):
        if self._is_editing:
            self._finish_editing()
        super().focusOutEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self._is_editing:
            # Cancel edit
            self.edit.hide()
            self.label.show()
            self._is_editing = False
            event.accept()
        else:
            super().keyPressEvent(event)

    def setText(self, text: str):
        """Set text programmatically (doesn't emit signal)."""
        self._text = text
        self.label.setText(text or self._placeholder)
        self.edit.setText(text)

    def text(self) -> str:
        return self._text
```

---
title: "feat: Add shot type filter to sequence tab"
type: feat
date: 2026-02-02
---

# feat: Add Shot Type Filter to Sequence Tab

## Overview

Add a shot type filter to the intention modal that allows users to select a specific shot type (wide, medium, close-up, etc.) and only show clips matching that type in the sequence. This feature is accessible via both the GUI and the chat agent.

## Problem Statement / Motivation

When building sequences, users often want to work with only certain types of shots. Currently, the sequence tab shows all clips regardless of shot type. Users must manually identify and select clips of the desired type, which is tedious for large projects.

Adding a shot type filter enables:
- **Focused editing**: Work with only close-ups for interview sequences, or only wide shots for establishing shots
- **Faster iteration**: Quickly explore how different shot types look in a sequence
- **Agent-assisted workflows**: Ask the chat agent to "show only wide shots" without manual selection

## Proposed Solution

Add a shot type combobox to the existing intention modal (`IntentionImportDialog`). When a user selects a shot type and imports, only clips matching that shot type appear in the sequence grid.

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Filter location | Intention modal | Consistent with existing algorithm/direction controls |
| Unanalyzed clips | Include in "All" only | Prevents silent exclusion of most clips |
| Empty state | Show message | Non-blocking, lets users understand and adjust |
| Order of operations | Filter → Algorithm → Direction | Mental model: "filter first, then sort" |
| Agent tool | New `set_sequence_shot_filter` | Dedicated tool for clarity |

## Technical Approach

### Files to Modify

| File | Changes |
|------|---------|
| `ui/dialogs/intention_import_dialog.py` | Add shot type dropdown, update signal |
| `ui/tabs/sequence_tab.py` | Filter clips by shot type before applying algorithm |
| `core/chat_tools.py` | Add `set_sequence_shot_filter` tool |
| `core/gui_state.py` | Track current shot type filter |

### UI Changes

Add shot type dropdown to intention modal, below the direction dropdown:

```
┌─────────────────────────────────────────┐
│        Import for [Algorithm]           │
├─────────────────────────────────────────┤
│                                         │
│  Direction:   [Shortest First    ▼]     │
│                                         │
│  Shot Type:   [All               ▼]  ◄── NEW
│               (Optional filter)         │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │     Drag & drop videos here     │   │
│  └─────────────────────────────────┘   │
│                                         │
│  [Cancel]              [Import]         │
└─────────────────────────────────────────┘
```

Dropdown options:
- "All" (default) - shows all clips including unanalyzed
- "Wide" - `shot_type == "wide shot"`
- "Full" - `shot_type == "full shot"`
- "Medium" - `shot_type == "medium shot"`
- "Close-up" - `shot_type == "close-up"`
- "Extreme CU" - `shot_type == "extreme close-up"`

### Signal Contract Change

Current signal:
```python
import_requested = Signal(list, list, str, str)  # local_paths, urls, algorithm, direction
```

Updated signal:
```python
import_requested = Signal(list, list, str, str, str)  # local_paths, urls, algorithm, direction, shot_type
```

The `shot_type` parameter is `None` when "All" is selected.

### Filter Logic

In `sequence_tab.py`, filter clips before applying the algorithm:

```python
# ui/tabs/sequence_tab.py

def _build_sequence(self, algorithm: str, direction: str, shot_type: str | None):
    """Build sequence with optional shot type filter."""
    clips = self._get_available_clips()

    # Apply shot type filter
    if shot_type:
        clips = [c for c in clips if c.shot_type == shot_type]

    if not clips:
        self._show_empty_state("No clips match the selected shot type")
        return

    # Apply algorithm and direction
    sorted_clips = self._apply_algorithm(clips, algorithm, direction)
    self._display_clips(sorted_clips)
```

### Agent Tool

Add new tool in `core/chat_tools.py`:

```python
# core/chat_tools.py

@tools.register(
    description="Filter sequence to show only clips of a specific shot type. "
                "Valid shot types: 'wide shot', 'full shot', 'medium shot', 'close-up', 'extreme close-up'. "
                "Use shot_type=None to show all clips.",
    requires_project=True,
    modifies_gui_state=True
)
def set_sequence_shot_filter(
    project,
    gui_state,
    main_window,
    shot_type: str | None = None,
) -> dict:
    """Set the shot type filter for the sequence tab."""
    from core.analysis.shots import SHOT_TYPES

    # Validate shot type
    if shot_type and shot_type not in SHOT_TYPES:
        return {
            "success": False,
            "error": f"Invalid shot type '{shot_type}'. Valid types: {SHOT_TYPES}"
        }

    # Get sequence tab and apply filter
    sequence_tab = main_window.get_tab("sequence")
    filtered_count = sequence_tab.apply_shot_type_filter(shot_type)

    # Update GUI state
    gui_state.sequence_shot_filter = shot_type

    return {
        "success": True,
        "result": {
            "shot_type": shot_type or "all",
            "clip_count": filtered_count,
            "message": f"Showing {filtered_count} clips" +
                      (f" of type '{shot_type}'" if shot_type else " (all types)")
        }
    }
```

### GUI State Tracking

Add to `core/gui_state.py`:

```python
# core/gui_state.py

class GUIState:
    def __init__(self):
        # ... existing fields ...
        self.sequence_shot_filter: str | None = None  # Current shot type filter
```

## Acceptance Criteria

### Functional Requirements

- [x] Shot type dropdown appears in intention modal below direction dropdown
- [x] Dropdown shows "All" as default, plus all shot types from `SHOT_TYPES`
- [x] Selecting a shot type filters clips to only show matching types
- [x] "All" shows all clips including those without shot_type metadata
- [x] Empty state message shows when no clips match selected filter
- [x] Signal includes shot_type parameter
- [x] Agent tool `set_sequence_shot_filter` filters existing sequence
- [x] Agent tool validates shot type input

### Edge Cases

- [ ] Modal with no analyzed clips: "All" works, specific types show empty state
- [ ] Modal with mixed analysis: specific filter shows only analyzed matches
- [ ] Agent requests invalid shot type: returns error with valid options

## Success Metrics

- Users can filter sequence by shot type without leaving the sequence tab
- Agent can respond to "show only close-ups" type requests
- No regressions in existing intention modal functionality

## Dependencies & Risks

**Dependencies:**
- Clips must have `shot_type` metadata from Analyze tab classification
- Existing `ShotTypeDropdown` widget can be reused

**Risks:**
- Users may not realize clips need analysis first → Mitigate with help text
- Signal change requires updating all consumers → Only `sequence_tab.py` consumes it

## Implementation Notes

### Reuse Existing Widget

The `ShotTypeDropdown` widget at `ui/widgets/shot_type_dropdown.py` already handles:
- Populating options from `SHOT_TYPES`
- Display name mapping
- Theme integration
- Guard flags for signal handling

However, it doesn't have an "All" option. Either:
1. Add "All" option to `ShotTypeDropdown` (affects other uses)
2. Create a local dropdown in the modal (simpler, isolated)

**Recommendation:** Create local dropdown in modal to avoid affecting other widget uses.

### Gotchas from Learnings

From `docs/solutions/`:
- Use guard flags (`_change_in_progress`) when handling dropdown signals
- Use `blockSignals(True/False)` when programmatically setting dropdown value
- Connect to `theme().changed` for dynamic theming

## References & Research

### Internal References

- Intention modal: `ui/dialogs/intention_import_dialog.py:157-611`
- Shot types: `core/analysis/shots.py:19-25`
- Shot type dropdown: `ui/widgets/shot_type_dropdown.py:18-131`
- Filter clips tool: `core/chat_tools.py:763-892`
- UI sizes: `ui/theme.py:99-124`

### Related Files

- Sequence tab: `ui/tabs/sequence_tab.py`
- GUI state: `core/gui_state.py`
- Tool executor: `core/tool_executor.py`

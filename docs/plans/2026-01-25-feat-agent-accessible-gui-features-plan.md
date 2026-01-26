---
title: "feat: Agent-Accessible GUI Features"
type: feat
date: 2026-01-25
---

# Agent-Accessible GUI Features Enhancement

## Overview

Make GUI features more apparent and usable by the chat agent through comprehensive tool expansion. Currently, there are significant gaps where users can perform actions in the GUI that the agent cannot access, limiting the agent's ability to assist with video editing workflows.

**Current state**: 11 tools exist (4 GUI state tools, 7 CLI tools)
**Target state**: ~26 tools covering full video editing workflow

## Problem Statement / Motivation

The Scene Ripper agent is limited compared to what users can do through the GUI:

1. **Sequence manipulation is one-way**: Agent can add clips but cannot remove, reorder, or clear
2. **No content-aware search**: Cannot search transcripts or find similar clips
3. **No project persistence**: Cannot save/load projects
4. **No clip metadata management**: Cannot add tags or notes
5. **Export tools incomplete**: EDL export exists in code but not exposed to agent
6. **No multi-step workflow coordination**: Complex workflows require manual intervention

This creates friction where users must switch between agent and GUI to complete tasks.

## Proposed Solution

Implement tools in four phases, prioritized by user impact and dependency order:

### Phase 1: Sequence & Selection Foundation
Enable full sequence manipulation and clip selection.

### Phase 2: Content-Aware Tools
Add search and similarity tools for intelligent clip discovery.

### Phase 3: Export & Project Management
Complete export capabilities and project lifecycle management.

### Phase 4: Workflow Automation
Multi-step workflow coordination with progress tracking.

## Technical Considerations

### Architecture Pattern
Follow the existing dual-pattern architecture:
- **GUI State Tools**: Direct project mutation for immediate UI updates (e.g., `add_to_sequence`)
- **CLI Tools**: Subprocess execution for heavy operations (e.g., `detect_scenes`)

New sequence/selection tools should be GUI State tools. Heavy export operations should be CLI tools.

### Thread Safety
GUI State tools must:
1. Use `modifies_gui_state=True` in registration
2. Be executed via `gui_tool_requested` signal on main thread
3. Follow guard flag pattern to prevent duplicate signal delivery

### Data Model Changes
The `Clip` dataclass needs extension for tags/notes:
```python
# models/clip.py
@dataclass
class Clip:
    # ... existing fields ...
    tags: list[str] = field(default_factory=list)
    notes: str = ""
```

### Signal Flow for GUI Updates
```
Tool Execution → Project.notify_observers() → ProjectSignalAdapter
    → Qt Signal → UI Component Update
```

## Acceptance Criteria

### Phase 1: Sequence & Selection
- [x] `remove_from_sequence(clip_ids: list[str])` - Remove clips by ID
- [x] `clear_sequence()` - Reset sequence to empty
- [x] `reorder_sequence(clip_ids: list[str])` - Set clip order
- [x] `get_sequence_state()` - Return current timeline state
- [x] `select_clips(clip_ids: list[str])` - Update GUI selection
- [x] `navigate_to_tab(tab_name: str)` - Switch active tab

### Phase 2: Content-Aware Tools
- [ ] `search_transcripts(query: str)` - Find clips by spoken content
- [ ] `find_similar_clips(clip_id: str, criteria: list[str])` - Visual similarity
- [ ] `group_clips_by(criteria: str)` - Group by color/shot_type/duration
- [ ] Extended `filter_clips` with transcript content filtering

### Phase 3: Export & Project Management
- [ ] `export_edl(output_path: Optional[str])` - CMX 3600 EDL
- [ ] `save_project(path: Optional[str])` - Save project state
- [ ] `load_project(path: str)` - Load existing project
- [ ] `new_project(name: str)` - Create fresh project
- [ ] `add_tags(clip_ids: list[str], tags: list[str])` - Tag clips
- [ ] `remove_tags(clip_ids: list[str], tags: list[str])` - Remove tags
- [ ] `add_note(clip_id: str, note: str)` - Add clip note
- [ ] `get_project_summary()` - Human-readable project overview

### Phase 4: Workflow Automation
- [ ] Multi-video pipeline support ("download 5 videos and detect scenes")
- [ ] Progress reporting for compound operations
- [ ] Failure recovery/resume capability
- [ ] Conflict queuing when GUI is busy

### Testing Requirements
- [ ] Unit tests for each new tool
- [ ] Integration tests for GUI state synchronization
- [ ] Test duplicate signal delivery guards

## Implementation Plan

### Phase 1: Sequence & Selection Foundation

#### 1.1 `remove_from_sequence` Tool

**File**: `core/chat_tools.py`

```python
@tools.register(
    description="Remove clips from the timeline sequence by their IDs.",
    requires_project=True,
    modifies_gui_state=True
)
def remove_from_sequence(clip_ids: list[str]) -> dict:
    """Remove clips from sequence."""
    # Uses existing Track.clips list manipulation
    # Emits sequence_changed via observer pattern
```

**Depends on**: Existing `Sequence` and `Track` models in `models/sequence.py`

#### 1.2 `clear_sequence` Tool

```python
@tools.register(
    description="Clear all clips from the timeline sequence.",
    requires_project=True,
    modifies_gui_state=True
)
def clear_sequence() -> dict:
    """Reset sequence to empty state."""
```

#### 1.3 `reorder_sequence` Tool

```python
@tools.register(
    description="Reorder clips in the sequence. Provide clip IDs in desired order.",
    requires_project=True,
    modifies_gui_state=True
)
def reorder_sequence(clip_ids: list[str]) -> dict:
    """Reorder sequence clips to match provided order."""
```

#### 1.4 `get_sequence_state` Tool

```python
@tools.register(
    description="Get the current state of the timeline sequence including all clips and their order.",
    requires_project=True,
    modifies_gui_state=False
)
def get_sequence_state() -> dict:
    """Return detailed sequence state."""
    # Returns: clips list with position, duration, source info
```

#### 1.5 `select_clips` Tool

```python
@tools.register(
    description="Select clips in the browser by their IDs. This updates the GUI selection state.",
    requires_project=True,
    modifies_gui_state=True
)
def select_clips(clip_ids: list[str]) -> dict:
    """Update GUI selection to specified clips."""
    # Updates GUIState.selected_clip_ids
    # Emits signal for ClipBrowser to update selection
```

#### 1.6 `navigate_to_tab` Tool

```python
@tools.register(
    description="Switch to a specific tab in the application. Valid tabs: collect, cut, analyze, sequence, generate, render",
    requires_project=False,
    modifies_gui_state=True
)
def navigate_to_tab(tab_name: str) -> dict:
    """Switch active tab."""
```

### Phase 2: Content-Aware Tools

#### 2.1 `search_transcripts` Tool

```python
@tools.register(
    description="Search clip transcripts for specific words or phrases. Returns clips containing the search term with timestamp and context.",
    requires_project=True,
    modifies_gui_state=False
)
def search_transcripts(query: str, case_sensitive: bool = False) -> dict:
    """Search transcripts for matching content."""
    # Returns: list of {clip_id, match_text, timestamp, context}
```

#### 2.2 `find_similar_clips` Tool

```python
@tools.register(
    description="Find clips visually similar to a reference clip based on color, shot type, or duration.",
    requires_project=True,
    modifies_gui_state=False
)
def find_similar_clips(
    clip_id: str,
    criteria: list[str] = ["color", "shot_type"],
    limit: int = 10
) -> dict:
    """Find similar clips by visual/temporal criteria."""
```

#### 2.3 `group_clips_by` Tool

```python
@tools.register(
    description="Group clips by a specific criteria: 'color' (dominant color), 'shot_type', 'duration' (short/medium/long), or 'source'.",
    requires_project=True,
    modifies_gui_state=False
)
def group_clips_by(criteria: str) -> dict:
    """Group clips by specified criteria."""
    # Returns: dict of {group_name: [clip_ids]}
```

### Phase 3: Export & Project Management

#### 3.1 Data Model Update

**File**: `models/clip.py`

```python
@dataclass
class Clip:
    id: str
    source_id: str
    start_frame: int
    end_frame: int
    dominant_colors: Optional[list[tuple[int, int, int]]] = None
    shot_type: Optional[str] = None
    transcript: Optional[list[TranscriptSegment]] = None
    # NEW FIELDS
    tags: list[str] = field(default_factory=list)
    notes: str = ""
```

#### 3.2 `export_edl` Tool

```python
@tools.register(
    description="Export the current sequence as an EDL (Edit Decision List) file for use in external video editors.",
    requires_project=True,
    modifies_gui_state=False
)
def export_edl(output_path: Optional[str] = None) -> dict:
    """Export sequence to EDL format."""
    # Uses existing core/edl_export.py
    # Default path: settings.export_dir / "{project_name}.edl"
```

#### 3.3 `save_project` Tool

```python
@tools.register(
    description="Save the current project to disk. Uses existing path if project was previously saved, or settings.project_dir for new projects.",
    requires_project=True,
    modifies_gui_state=False
)
def save_project(path: Optional[str] = None) -> dict:
    """Save project state to JSON file."""
```

#### 3.4 `load_project` Tool

```python
@tools.register(
    description="Load a project from a .sceneripper file.",
    requires_project=False,
    modifies_gui_state=True
)
def load_project(path: str) -> dict:
    """Load project from file."""
    # Clears current state, loads new project
    # Triggers full UI refresh via observers
```

#### 3.5 Tag Management Tools

```python
@tools.register(
    description="Add tags to one or more clips for organization and filtering.",
    requires_project=True,
    modifies_gui_state=True
)
def add_tags(clip_ids: list[str], tags: list[str]) -> dict:
    """Add tags to specified clips."""

@tools.register(
    description="Remove tags from one or more clips.",
    requires_project=True,
    modifies_gui_state=True
)
def remove_tags(clip_ids: list[str], tags: list[str]) -> dict:
    """Remove tags from specified clips."""
```

#### 3.6 `add_note` Tool

```python
@tools.register(
    description="Add or update a note on a clip.",
    requires_project=True,
    modifies_gui_state=True
)
def add_note(clip_id: str, note: str) -> dict:
    """Set note text for a clip."""
```

#### 3.7 `get_project_summary` Tool

```python
@tools.register(
    description="Generate a human-readable summary of the current project including sources, clips, analysis status, and sequence.",
    requires_project=True,
    modifies_gui_state=False
)
def get_project_summary() -> dict:
    """Generate project summary."""
    # Returns: markdown-formatted summary
```

### Phase 4: Workflow Automation

#### 4.1 Multi-Step Workflow Support

Update `chat_worker.py` system prompt to handle compound requests:

```python
# In _build_system_prompt():
"""
WORKFLOW AUTOMATION:
For multi-step requests like "download 5 videos and detect scenes in each":
1. Process each step sequentially
2. Report progress after each step
3. If a step fails, continue with remaining items and report failures at end
4. Provide a summary when complete
"""
```

#### 4.2 Progress Reporting Signal

**File**: `ui/chat_worker.py`

```python
# Add new signal
workflow_progress = Signal(str, int, int)  # step_name, current, total

# Emit during multi-step operations
self.workflow_progress.emit("Downloading videos", 2, 5)
```

## Success Metrics

1. **Tool coverage**: Increase from 11 to ~26 tools
2. **Workflow completion**: User can complete full editing workflow via agent without GUI switching
3. **User satisfaction**: Agent can assist with previously impossible tasks

## Dependencies & Risks

### Dependencies
- Existing `Project` observer pattern must handle new event types
- `edl_export.py` module exists and is functional
- GUI components have selection APIs

### Risks

| Risk | Mitigation |
|------|------------|
| Race conditions with GUI | Use guard flags, queue agent operations during active GUI interaction |
| Schema migration for tags/notes | Tags/notes default to empty, backwards compatible |
| Large projects may have slow searches | Add pagination to search results, limit defaults |

## References & Research

### Internal References
- Tool registration pattern: `core/chat_tools.py:265`
- GUI State tool example: `add_to_sequence` at `core/chat_tools.py:303`
- Observer pattern: `core/project.py` and `ui/project_adapter.py`
- EDL export: `core/edl_export.py`
- Sequence model: `models/sequence.py:9`
- Clip model: `models/clip.py:104`
- Guard flag pattern: `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`
- Agent-GUI sync: `docs/plans/2026-01-25-feat-agent-gui-bidirectional-sync-plan.md`

### Institutional Learnings Applied
- Guard flags for signal handlers to prevent duplicate delivery
- Single source of truth for state objects
- Model/view synchronization after state changes
- Source ID matching between workers and UI

### Related Work
- Agent-native architecture plan: `docs/plans/agent-native-architecture-plan.md`

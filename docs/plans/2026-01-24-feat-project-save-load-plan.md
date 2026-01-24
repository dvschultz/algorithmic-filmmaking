---
title: "feat: Project Save and Load"
type: feat
date: 2026-01-24
priority: P1
---

# feat: Project Save and Load

## Overview

Add project persistence to Scene Ripper, allowing users to save their work (sources, clips, sequences) to JSON files and restore them later. Projects use relative paths for portability and regenerate thumbnails on load.

## Problem Statement

Users currently lose all work when closing the app:
1. Detected scenes and their metadata (colors, shot types, transcripts) are lost
2. Assembled sequences on the timeline disappear
3. No way to resume work on a project across sessions
4. Cannot share projects between machines

## Proposed Solution

Implement save/load functionality following existing patterns from `core/dataset_export.py`:
- **Save**: Serialize sources, clips, and sequences to versioned JSON
- **Load**: Parse JSON, resolve relative paths, rebuild application state
- **Portability**: Store video paths relative to project file location
- **Thumbnails**: Store clip IDs only; regenerate from cache or video on load

## Technical Approach

### Project File Format (JSON)

```json
{
  "version": "1.0",
  "project_name": "My Project",
  "created_at": "2026-01-24T10:30:00Z",
  "modified_at": "2026-01-24T14:15:00Z",

  "sources": [{
    "id": "uuid",
    "file_path": "../videos/footage.mp4",
    "duration_seconds": 120.5,
    "fps": 30.0,
    "width": 1920,
    "height": 1080
  }],

  "clips": [{
    "id": "uuid",
    "source_id": "source-uuid",
    "start_frame": 0,
    "end_frame": 900,
    "dominant_colors": [{"r": 255, "g": 128, "b": 64}],
    "shot_type": "wide",
    "transcript": [{
      "start_time": 0.5,
      "end_time": 2.3,
      "text": "Hello world",
      "confidence": -0.3
    }]
  }],

  "sequence": {
    "id": "uuid",
    "name": "Main Sequence",
    "fps": 30.0,
    "tracks": [{
      "id": "uuid",
      "name": "Video 1",
      "clips": [{
        "id": "uuid",
        "source_clip_id": "clip-uuid",
        "source_id": "source-uuid",
        "track_index": 0,
        "start_frame": 0,
        "in_point": 100,
        "out_point": 200
      }]
    }]
  },

  "ui_state": {
    "sensitivity": 3.0
  }
}
```

### Path Resolution Strategy

- **Base directory**: Project file's parent directory
- **Relative paths**: Always store relative paths for sources
- **Absolute fallback**: If relative path fails, try the stored absolute path (saved as `_absolute_path`)
- **Cross-platform**: Use forward slashes in JSON; resolve with `pathlib` on load

### Files to Create/Modify

| File | Change |
|------|--------|
| `core/project.py` | **New** - Project dataclass, save/load functions |
| `models/clip.py` | Add `to_dict()` / `from_dict()` methods |
| `models/sequence.py` | Add `to_dict()` / `from_dict()` methods |
| `core/transcription.py` | Add serialization for TranscriptSegment |
| `ui/main_window.py` | Add menu items, dirty state tracking, save/load handlers |

### New Module: `core/project.py`

```python
@dataclass
class ProjectMetadata:
    id: str
    name: str
    created_at: str
    modified_at: str
    version: str = "1.0"

def save_project(
    filepath: Path,
    sources: dict[str, Source],
    clips: list[Clip],
    sequence: Optional[Sequence],
    settings: dict,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """Save project to JSON file with relative paths."""

def load_project(
    filepath: Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> tuple[list[Source], list[Clip], Optional[Sequence], ProjectMetadata]:
    """Load project from JSON file, resolving paths and validating sources."""
```

### UI Integration

#### Menu Items
- **File > Save Project** (Cmd+S / Ctrl+S) - Save to current file or prompt for location
- **File > Save Project As...** (Cmd+Shift+S / Ctrl+Shift+S) - Always prompt for location
- **File > Open Project...** (Cmd+O / Ctrl+O) - Open file dialog
- **File > Recent Projects** - Submenu with last 10 projects

#### Window Title
Format: `Scene Ripper - [filename]` with `*` suffix for unsaved changes
Example: `Scene Ripper - movie.json*`

#### Dirty State Tracking
Track changes to:
- [x] Sources added/removed
- [x] Clips modified (colors, shot type, transcript)
- [x] Sequence clips added/removed/moved
- [ ] Playhead position (not tracked)
- [ ] UI selections (not tracked)

### Error Handling

#### Missing Source Video
When a referenced video file doesn't exist:
1. Show dialog: "Source video not found: [filename]"
2. Options: "Locate..." / "Skip" / "Cancel Load"
3. If "Locate": Open file picker, remap path
4. If "Skip": Load project without that source (orphan clips become unavailable)

#### Corrupted Project File
1. Validate JSON syntax
2. Validate required fields exist
3. Validate version is supported (≤ current app version)
4. Show specific error message if validation fails

#### Schema Version Migration
- v1.0 → future: Add migration functions as needed
- Newer file in older app: Warn user, attempt best-effort load

## Implementation Steps

### Step 1: Data Model Serialization

Add `to_dict()` and `from_dict()` methods to:
- [x] `Source` in `models/clip.py`
- [x] `Clip` in `models/clip.py`
- [x] `TranscriptSegment` in `core/transcription.py`
- [x] `Sequence`, `Track`, `SequenceClip` in `models/sequence.py`

### Step 2: Core Project Module

Create `core/project.py`:
- [x] `ProjectMetadata` dataclass
- [x] `save_project()` function
- [x] `load_project()` function
- [x] Path resolution helpers (relative ↔ absolute)
- [x] Schema validation

### Step 3: MainWindow Integration

Add to `ui/main_window.py`:
- [x] `self.current_project_path: Optional[Path]`
- [x] `self._is_dirty: bool` flag
- [x] `_mark_dirty()` method called on state changes
- [x] `_on_save_project()` handler
- [x] `_on_save_project_as()` handler
- [x] `_on_open_project()` handler
- [x] `_check_unsaved_changes()` method with dialog
- [x] Recent projects list (use QSettings)

### Step 4: Menu Items

Add to File menu:
- [x] Save Project action with Cmd+S shortcut
- [x] Save Project As action with Cmd+Shift+S shortcut
- [x] Open Project action with Cmd+O shortcut
- [x] Recent Projects submenu
- [x] Separator before Quit

### Step 5: State Restoration

On load, restore:
- [x] Sources and clips to `MainWindow`
- [x] Clips to `AnalyzeTab.clip_browser`
- [x] Sequence to `SequenceTab.timeline`
- [x] Queue thumbnail regeneration for clips missing thumbnails

### Step 6: Unsaved Changes Warning

Add to:
- [x] `closeEvent()` - before app closes
- [x] `_on_open_project()` - before loading another project
- [ ] Any "New Project" action (not implemented - no New Project action exists)

## Acceptance Criteria

- [ ] Can save project with Cmd+S (prompts for location if new)
- [ ] Can save project to new location with Cmd+Shift+S
- [ ] Can open project with Cmd+O
- [ ] Recent projects shows last 10 projects
- [ ] Window title shows project name with * for unsaved changes
- [ ] Prompted to save before closing with unsaved changes
- [ ] Missing source videos show "Locate..." dialog
- [ ] Projects are portable (relative paths work when moved with videos)
- [ ] Thumbnails regenerate automatically if missing
- [ ] Schema version in file for future migration support

## Design Decisions

### Why JSON over binary formats?
- Human-readable for debugging
- Git-friendly for version control
- Platform-independent
- Matches existing `dataset_export.py` pattern

### Why relative paths?
- Projects can be moved with their video files
- Can share projects between machines
- Fallback to absolute path if relative fails

### Why not save thumbnails?
- Thumbnails can be large (megabytes per project)
- Regeneration is fast from existing cache
- Cache keyed by clip ID makes regeneration deterministic

### Why single sequence per project?
- Matches current UI (one timeline)
- Simpler mental model
- Can expand to multiple sequences in future version

## Future Enhancements (Out of Scope)

- Auto-save with recovery
- Project templates
- Multiple sequences per project
- Cloud sync
- Undo/redo history persistence

## References

- Existing pattern: `core/dataset_export.py` (JSON serialization)
- Existing pattern: `core/settings.py` (QSettings for recent projects)
- Data models: `models/clip.py`, `models/sequence.py`
- MainWindow state: `ui/main_window.py:348-365`

---

*Generated: 2026-01-24*

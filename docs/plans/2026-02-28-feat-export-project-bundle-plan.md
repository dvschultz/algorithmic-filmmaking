---
title: "feat: Export project as self-contained bundle"
type: feat
status: completed
date: 2026-02-28
origin: docs/brainstorms/2026-02-28-export-project-brainstorm.md
---

# feat: Export project as self-contained bundle

## Overview

Add a project export feature that bundles a Scene Ripper project file and all referenced assets into a self-contained folder. The bundle can be opened on another machine or shared with collaborators and loads seamlessly using the existing relative-path resolution mechanism.

## Problem Statement / Motivation

Currently, a `.sceneripper` project file stores paths relative to its parent directory. If the user moves the project file away from its original location, or wants to share it with someone else, all source/frame references break. There's no way to package a project for archival or collaboration without manually organizing files.

## Proposed Solution

Add **File > Export Project Bundle...** that creates a structured folder:

```
<ProjectName>-export/
├── <ProjectName>.sceneripper    # Project JSON (paths rewritten)
├── sources/                     # Source video files (optional)
│   ├── interview.mp4
│   └── broll.mp4
└── frames/                      # Extracted frame images (if any)
    └── frame_0001.png
```

Key behaviors (see brainstorm: `docs/brainstorms/2026-02-28-export-project-brainstorm.md`):
- **In-memory serialization**: Exports the current project state, not the on-disk file. Works even with unsaved changes or never-saved projects.
- **Include videos option**: User chooses per-export. Full bundle copies source videos; lightweight bundle omits them.
- **Path rewriting**: All `file_path` fields rewritten to point into bundle subdirectories. `_absolute_path` fallback fields are **stripped** from the exported JSON to ensure true portability.
- **Seamless open**: Opening the bundle's `.sceneripper` file resolves paths naturally via existing relative-path mechanism.
- **Lightweight re-link**: When opening a lightweight bundle, `missing_source_callback` fires to prompt re-linking of missing source videos.
- **No project.path change**: Export is a copy operation. The original project's save path is unchanged.

## Technical Considerations

### Path handling

Sources and frames store `file_path` relative to the project file's parent directory (`Source.to_dict(base_path=...)` at `models/clip.py:95`). The export rewrites paths to point into `sources/` and `frames/` subdirectories, then calls the same `save_project()` serialization with the bundle directory as `base_path`.

**Critical: Strip `_absolute_path`**. Both `Source.from_dict()` (`models/clip.py:150`) and `Frame.from_dict()` (`models/frame.py:159`) fall back to `_absolute_path` when relative resolution fails. If left in the exported JSON, a bundle opened on the same machine would silently load from original locations instead of the bundle — defeating portability. The export must strip these fields.

### Lightweight export path strategy

Even when videos are excluded, source `file_path` fields in the JSON should point to `sources/<filename>`. This way:
- The `missing_source_callback` fires consistently on load
- A user can manually drop video files into `sources/` to "complete" a lightweight bundle

### Thumbnails are not bundled

Thumbnails are **not persisted** in the project JSON — `Source.from_dict()`, `Clip.from_dict()`, and `Frame.from_dict()` all set `thumbnail_path=None` ("Regenerate on load"). No thumbnail copying is needed. This significantly simplifies the bundle.

### Filename collision handling

Multiple sources can share a filename (e.g., `interview.mp4` from different directories). The export applies a numeric suffix: `interview.mp4`, `interview_2.mp4`, `interview_3.mp4`. The same strategy applies to frame files.

### Worker pattern

File copying (especially multi-GB videos) runs in a `CancellableWorker` (`ui/workers/base.py:14`) with:
- `progress(int, int)` signal for file-level progress (current file, total files)
- `is_cancelled()` checks between file copies
- Cleanup of partial bundle on cancellation

## Acceptance Criteria

### Functional

- [x] **File > Export Project Bundle...** menu action added after "Export EDL..." in File menu
- [x] Custom export dialog with folder picker and "Include source videos" checkbox
- [x] Dialog shows summary: N sources (X.X GB), M frames
- [x] Full export copies source videos into `sources/` and frames into `frames/`
- [x] Lightweight export (unchecked) skips video copying, still writes `sources/<filename>` paths in JSON
- [x] `_absolute_path` fields stripped from exported project JSON
- [x] Filename collisions resolved with numeric suffix (`name_2.mp4`, `name_3.mp4`)
- [x] Opening the exported `.sceneripper` file loads the project with all paths resolving
- [x] Opening a lightweight export triggers `missing_source_callback` for each missing source
- [x] Missing source/frame files at export time are skipped with a warning, shown in summary
- [x] Export runs in a background worker with progress dialog and cancel button
- [x] Cancellation cleans up partial bundle folder
- [x] Overwrite confirmation when destination already exists
- [x] `project.path` unchanged after export (export is not "Save As")
- [x] Works with never-saved projects (serializes in-memory state)

### Testing

- [x] Unit test: `export_project_bundle()` creates correct folder structure
- [x] Unit test: paths in exported JSON are relative to bundle directory
- [x] Unit test: `_absolute_path` stripped from exported JSON
- [x] Unit test: filename collision resolution produces unique names
- [x] Unit test: lightweight export writes `sources/<filename>` paths even without copying videos
- [x] Unit test: missing source files are skipped with logged warning
- [x] Unit test: missing frame files are skipped with logged warning
- [x] Integration test: export then load round-trip preserves all project data (sources, clips, sequence, frames, metadata)
- [x] Integration test: lightweight export then load triggers missing source callback

## Implementation Plan

### Phase 1: Core export logic (`core/project_export.py`)

New module with the bundle export function. Follows the same patterns as `save_project()` in `core/project.py`.

**`core/project_export.py`**

```python
def export_project_bundle(
    project: Project,
    dest_dir: Path,
    include_videos: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> ExportResult:
```

Steps:
1. Validate `dest_dir` doesn't exist (or confirm overwrite upstream)
2. Create `dest_dir/`, `dest_dir/sources/`, `dest_dir/frames/`
3. Build filename mapping for collision resolution (sources and frames)
4. Copy frame files to `frames/` (with collision-safe names)
5. If `include_videos`: copy source videos to `sources/` (with collision-safe names), checking `cancel_check()` between files
6. Create temporary copies of Source/Frame objects with rewritten `file_path` pointing into bundle subdirectories
7. Call `save_project()` with `filepath=dest_dir/<ProjectName>.sceneripper` and the rewritten objects
8. Post-process: strip `_absolute_path` from the saved JSON (read, filter, rewrite)
9. Return `ExportResult` with stats (files copied, skipped, total size)

**Filename collision resolution** (`_build_filename_map`):
```python
# Input: [("/a/video.mp4", "sources"), ("/b/video.mp4", "sources")]
# Output: {"/a/video.mp4": "sources/video.mp4", "/b/video.mp4": "sources/video_2.mp4"}
```

**Files:**
- `core/project_export.py` — new module
- `tests/test_project_export.py` — new test file

### Phase 2: Export worker (`ui/workers/export_worker.py`)

Background worker inheriting from `CancellableWorker`:

```python
class ExportWorker(CancellableWorker):
    progress = Signal(int, int, str)  # current, total, filename
    export_completed = Signal(object)  # ExportResult
    error = Signal(str)
```

Follows the same pattern as `DetectionWorker` (`ui/main_window.py:101`) and `SequenceExportWorker` (`ui/main_window.py:395`).

**Files:**
- `ui/workers/export_worker.py` — new worker

### Phase 3: Export dialog (`ui/dialogs/export_bundle_dialog.py`)

Custom `QDialog` (not `QFileDialog` — need the checkbox). Layout:

- Folder picker (QLineEdit + Browse button)
- "Include source videos" checkbox (default: checked)
- Summary label: "3 sources (2.4 GB), 12 frames"
- Size updates when checkbox toggles (omit video size when unchecked)
- Export / Cancel buttons

Follows existing dialog patterns in the codebase (e.g., `ui/settings_dialog.py`).

**Files:**
- `ui/dialogs/export_bundle_dialog.py` — new dialog

### Phase 4: Menu integration (`ui/main_window.py`)

Add to `_create_menu_bar()` after the Export EDL action (around line 1146):

```python
self.export_bundle_action = QAction("Export Project &Bundle...", self)
self.export_bundle_action.triggered.connect(self._on_export_bundle_click)
file_menu.addAction(self.export_bundle_action)
```

Add `_on_export_bundle_click()` handler that:
1. Shows the export dialog
2. On accept, creates and starts `ExportWorker`
3. Shows progress dialog
4. On completion, shows success/summary dialog
5. On error, shows error message

Wire up worker signals following the existing pattern of `self.export_worker` reference stored on `MainWindow`.

**Files:**
- `ui/main_window.py` — add menu action, handler, worker management

## Error Handling

| Scenario | Behavior |
|---|---|
| Source video missing at export time | Skip, log warning, include in summary |
| Frame file missing at export time | Skip, log warning, include in summary |
| Destination already exists | Confirmation dialog before overwriting |
| Destination is read-only / permission denied | Error dialog with specific message |
| Disk space exhausted mid-copy | Error dialog, partial bundle left on disk with warning |
| Export cancelled by user | Delete partial bundle directory |
| Export of empty project (0 sources) | Allowed — creates bundle with just the project file |

## Dependencies & Risks

- **No new dependencies**: Uses only `shutil.copy2`, `pathlib`, `json` — all stdlib
- **Large file risk**: Video files can be multi-GB. Progress feedback and cancellation are essential.
- **Filesystem edge cases**: Case-insensitive filename collisions on macOS/Windows (two files differing only in case). The numeric suffix strategy handles this since it checks the destination filesystem.

## Sources & References

- **Origin brainstorm:** [docs/brainstorms/2026-02-28-export-project-brainstorm.md](docs/brainstorms/2026-02-28-export-project-brainstorm.md) — Key decisions: plain folder (not ZIP), subdirectory structure, option to exclude videos, seamless re-open, prompt-to-relink for missing sources
- Project save/load: `core/project.py` — `save_project()` at line 76, `load_project()` at line 263
- Path serialization: `models/clip.py:95` (`Source.to_dict`), `models/frame.py:66` (`Frame.to_dict`)
- `_absolute_path` fallback: `models/clip.py:150`, `models/frame.py:159`
- Worker base: `ui/workers/base.py:14` (`CancellableWorker`)
- File menu: `ui/main_window.py:1088` (`_create_menu_bar`)
- Export EDL pattern: `ui/main_window.py:1141`
